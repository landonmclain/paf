"""Shell capability that provides command execution for agents.

Originally copied from pydantic-ai-harness, MIT.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opentelemetry import trace
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ApprovalRequired, SkipToolExecution
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from paf.approvals import approvals_available, current_channel
from paf.capabilities._sandbox import SandboxedExecutor

from ._analyze import analyze
from ._command_policy import CommandPolicy, evaluate
from ._preview import preview
from ._toolset import ShellToolset

# Tools whose `command` argument the command policy inspects.
_COMMAND_TOOLS = ("run_command", "start_command")
POLICY_SPAN = "paf.shell.command_policy"
_TRACER = trace.get_tracer("paf")

COMMAND_POLICY_INSTRUCTIONS = (
    "Some shell commands (deleting files, discarding git changes, force-pushing, "
    "writing to disks) ask the user for approval before they run. When the task "
    "needs one, run the correct, direct command: the user will be shown it and "
    "asked. Do not substitute a workaround to avoid the prompt (find -delete, a "
    "Python one-liner, moving files elsewhere): those ask too, and are harder for "
    "the user to review. If a command is denied or refused, do not retry it or "
    "work around it; ask the user how to proceed."
)
_BLOCKED = (
    "This shell command was refused by the agent's command policy before it ran: "
    "{reason}. Do not retry it or work around it; tell the user what you wanted "
    "to do."
)
_UNATTENDED = (
    "This shell command needs the user's approval ({reason}), and no one is "
    "attached to this turn to give it, so it was refused before it ran. Do not "
    "retry it or work around it; tell the user what you wanted to do."
)

_DEFAULT_DENIED_COMMANDS: tuple[str, ...] = (
    "rm",
    "rmdir",
    "mkfs",
    "dd",
    "format",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init",
)


LLM_API_KEY_ENV_PATTERNS: tuple[str, ...] = (
    "ANTHROPIC_*",
    "GATEWAY_*",
    "GEMINI_*",
    "GOOGLE_*",
    "OPENAI_*",
    "OPENROUTER_*",
    "PYDANTIC_AI_GATEWAY_API_KEY",
)
"""Glob patterns for common LLM provider credentials, for `denied_env_patterns`.

Pass these when an agent runs untrusted commands that must not read the host's
LLM API keys. Covers provider prefixes only -- not other host secrets, and the
prefixes are coarse (`GOOGLE_*` also strips `GOOGLE_APPLICATION_CREDENTIALS`),
so treat it as a starting point. Not a default: stripping env silently would
break agents that rely on inherited credentials, so opt in explicitly.
"""


@dataclass
class Shell(AbstractCapability[AgentDepsT]):
    """Shell command execution for agents.

    Commands execute in a subprocess rooted at `cwd`. Use `allowed_commands`
    or `denied_commands` to control what the agent can invoke.
    """

    cwd: str | Path = "."
    """Working directory for command execution."""

    allowed_commands: Sequence[str] = field(default_factory=list[str])
    """If non-empty, only these command names may be executed (allowlist)."""

    denied_commands: Sequence[str] = _DEFAULT_DENIED_COMMANDS
    """These command names are always rejected (denylist).

    Defaults to blocking destructive commands (rm, dd, shutdown, etc.).
    Set to an empty list to disable.
    """

    denied_operators: Sequence[str] = field(default_factory=list[str])
    """Shell operators that are blocked (e.g. '>', '>>', '|' for restrictive mode)."""

    default_timeout: float = 30.0
    """Default timeout in seconds for command execution."""

    max_output_chars: int = 50_000
    """Maximum characters of output returned to the model. Must be positive."""

    persist_cwd: bool = False
    """If True, track cd commands and adjust the working directory for subsequent calls."""

    allow_interactive: bool = False
    """If True, allow interactive commands (vi, nano, ssh, etc.). Blocked by default."""

    env: Mapping[str, str] | None = None
    """Explicit environment for spawned subprocesses, replacing inheritance.

    When `None` (default) the subprocess inherits the parent environment. Set
    this to a fixed mapping to start subprocesses with exactly these variables
    and nothing else -- a hard boundary that keeps host secrets (LLM API keys,
    tokens) out of commands the agent runs.
    """

    denied_env_patterns: Sequence[str] = field(default_factory=list[str])
    """Glob patterns for environment variable names to strip before spawning.

    Follows the `denied_*` naming convention but matches by glob (`fnmatch`,
    e.g. `OPENAI_*`), since env secrets cluster by prefix -- unlike
    `denied_commands`, which matches executable names exactly. Names matching
    any pattern are removed from the base environment; applied on top of `env`
    when both are set, so patterns filter an explicit `env` too. See
    `LLM_API_KEY_ENV_PATTERNS` for a ready-made provider-credential denylist.
    """

    executor: SandboxedExecutor | None = None
    """The execution backend for spawned commands.

    When `None` (default) a `LocalProcessExecutor` runs commands directly on the
    host (no sandbox). Set this to a `DockerExecutor` to run commands inside a
    locked-down container. The model cannot change this at runtime; it is fixed
    at construction, per 'the model proposes, the runtime executes'.
    """

    processes_root: str | Path | None = None
    """Root where durable background stores live (``data/processes``).

    When set, shell background commands are persisted to
    ``<processes_root>/<agent>/runs.jsonl`` (UI feed) and
    ``<processes_root>/<agent>/shell_cmds.jsonl`` (rich shell records) and run
    through the ``ShellBackgroundQueue`` as detached subprocesses. When `None`
    (default) shell keeps the legacy in-memory background-process behavior.
    """

    agent: str = ""
    """Agent name used to scope durable stores. Empty when not persisting."""

    on_complete: Callable[[str, str | None, bool, str], Awaitable[None]] | None = None
    """Completion-notification hook, injected by the embedding layer (the SSE
    transport). When set, the toolset's queue fires it when a
    ``notify_on_completion`` process finishes; the callback is expected to run a
    notify turn through the originating session. ``None`` disables the hook.
    """

    command_policy: CommandPolicy | None = None
    """Per-command allow / approve / block (``_command_policy``). When set it
    replaces the first-word deny list: a destructive command reaches the person
    for approval instead of being refused. Paths are resolved against ``cwd``
    (a ``cd`` earlier in the turn is not tracked here when ``persist_cwd`` is on)."""

    def __post_init__(self) -> None:
        """Resolve the built-in denylist according to the selected policy."""
        if self.denied_commands is _DEFAULT_DENIED_COMMANDS:
            replaced = self.allowed_commands or self.command_policy is not None
            self.denied_commands = [] if replaced else list(_DEFAULT_DENIED_COMMANDS)

    def get_instructions(self) -> list[str] | None:
        return (
            [COMMAND_POLICY_INSTRUCTIONS] if self.command_policy is not None else None
        )

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply the command policy before a shell command runs."""
        policy = self.command_policy
        if policy is None or call.tool_name not in _COMMAND_TOOLS:
            return args
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return args
        cwd = Path(self.cwd).expanduser()
        decision = evaluate(policy, analyze(command), cwd)
        if decision.action == "allow":
            return args
        self._trace(call, command, decision)
        if decision.action == "block":
            raise SkipToolExecution(_BLOCKED.format(reason=decision.reason))
        if ctx.tool_call_approved:
            return args
        if not approvals_available():
            raise SkipToolExecution(_UNATTENDED.format(reason=decision.reason))
        channel = current_channel()
        if channel is not None:
            channel.annotate(
                call.tool_call_id,
                reason=decision.reason,
                preview=preview(decision.deciding, cwd),
            )
        raise ApprovalRequired

    @staticmethod
    def _trace(call: ToolCallPart, command: str, decision: Any) -> None:
        try:
            from paf.capabilities.guardrails.secret_handles import scrub_secrets

            with _TRACER.start_as_current_span(POLICY_SPAN) as span:
                span.set_attribute("tool", call.tool_name)
                span.set_attribute("action", decision.action)
                span.set_attribute("command", scrub_secrets(command)[:500])
                span.set_attribute("reason", decision.reason[:500])
        except Exception:  # noqa: BLE001 - observability must never fail the run
            pass

    def get_toolset(self) -> ShellToolset[AgentDepsT]:
        """Build and return the shell toolset."""
        return ShellToolset[AgentDepsT](
            cwd=Path(self.cwd),
            allowed_commands=self.allowed_commands,
            denied_commands=self.denied_commands,
            denied_operators=self.denied_operators,
            default_timeout=self.default_timeout,
            max_output_chars=self.max_output_chars,
            persist_cwd=self.persist_cwd,
            allow_interactive=self.allow_interactive,
            env=self.env,
            denied_env_patterns=self.denied_env_patterns,
            executor=self.executor,
            processes_root=self.processes_root,
            agent=self.agent,
            on_complete=self.on_complete,
        )
