"""Shell toolset -- gives agents the ability to run commands.

Originally copied from pydantic-ai-harness, MIT.

The toolset is a thin execution seam: it owns the model-facing tools
(``run_command`` / ``start_command`` / ``check_command`` / ``stop_command``),
the allow/deny filtering, cwd tracking, output labelling and truncation, and
the ``command_id -> BackgroundHandle`` bookkeeping. The actual process spawning
and sandbox policy live in a :class:`SandboxedExecutor` backend, injected at
construction, so the execution boundary can be swapped (local process now,
Docker later) without changing the tool surface.
"""

from __future__ import annotations

import fnmatch
import functools
import os
import re
import shlex
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Concatenate, ParamSpec

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool

from paf.background_processes._events import (
    ProcessEvent,
    ProcessEventType,
    get_agent_emitter,
)
from paf.capabilities._sandbox import (
    BackgroundHandle,
    BackgroundStatus,
    LocalProcessExecutor,
    SandboxedExecutor,
)

from ._background import ShellBackgroundQueue, ShellRun
from ._output import truncate_tail
from ._store import ShellCommandStore

_P = ParamSpec("_P")


def _recoverable(  # noqa: UP047  # ParamSpec/Concatenate signature-preserving decorator; PEP 695 inline type params can't express it
    fn: Callable[Concatenate[ShellToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[ShellToolset, _P], Awaitable[str]]:
    """Convert model-correctable errors into `ModelRetry`.

    pydantic-ai only feeds `ModelRetry` back to the model as a retry prompt; any
    other exception propagates and aborts the whole run. A denied command is
    something the model can recover from (pick an allowed one), so surface it as
    a retry instead of crashing the agent. paf also recovers spawn failures (a
    missing command binary raises `FileNotFoundError`/`OSError`), so a typo'd
    command returns to the model instead of aborting the run.
    """

    @functools.wraps(fn)
    async def wrapper(self: ShellToolset, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except OSError as e:
            raise ModelRetry(str(e)) from e

    return wrapper


def _is_interactive_command(command: str) -> bool:
    """Detect commands that typically require interactive input."""
    interactive_patterns = [
        r"^(vi|vim|nano|emacs|less|more|top|htop|man)\b",
        r"^sudo\s",
        r"^passwd\b",
        r"^ssh\b",
        r"^telnet\b",
        r"^ftp\b",
    ]
    return any(re.match(p, command.strip()) for p in interactive_patterns)


class ShellToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent the ability to execute shell commands.

    Supports synchronous execution (run_command) and background processes
    (start_command / check_command / stop_command). Output is streamed,
    truncated to fit model context, and labelled with stdout/stderr/exit code.

    Optionally tracks the working directory across calls so ``cd`` persists.
    """

    def __init__(
        self,
        *,
        cwd: Path,
        allowed_commands: Sequence[str],
        denied_commands: Sequence[str],
        denied_operators: Sequence[str],
        default_timeout: float,
        max_output_chars: int,
        persist_cwd: bool,
        allow_interactive: bool,
        env: Mapping[str, str] | None = None,
        denied_env_patterns: Sequence[str] = (),
        executor: SandboxedExecutor | None = None,
        processes_root: str | Path | None = None,
        agent: str = "",
        conversation_id: str | None = None,
        on_complete: (
            Callable[[str, str | None, bool, str], Awaitable[None]] | None
        ) = None,
    ) -> None:
        super().__init__()
        self._cwd = cwd.resolve()
        # The configured starting directory, never mutated by persist_cwd, so
        # `for_run` can hand each run a fresh instance rooted back here.
        self._initial_cwd = self._cwd
        self._conversation_id = conversation_id
        # Completion notification hook, injected by the embedding layer (the
        # SSE transport). When None, completed processes just persist/feed and
        # never trigger a notify turn.
        self._on_complete = on_complete
        self._allowed_commands = list(allowed_commands)
        self._denied_commands = list(denied_commands)
        self._denied_operators = list(denied_operators)
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars
        self._persist_cwd = persist_cwd
        self._allow_interactive = allow_interactive
        self._env = dict(env) if env is not None else None
        self._denied_env_patterns = list(denied_env_patterns)
        self._executor = executor if executor is not None else LocalProcessExecutor()
        self._background: dict[str, BackgroundHandle] = {}
        # Durable stores + background queue. Only wired when a `processes_root`
        # is given (the shell capability's default), so callers that just want
        # in-memory background processes keep behaving exactly as before.
        self._processes_root = (
            Path(processes_root) if processes_root is not None else None
        )
        self._agent = agent
        self._shell_cmds: ShellCommandStore | None = None
        self._bg_queue: ShellBackgroundQueue | None = None
        if self._processes_root is not None:
            self._shell_cmds = ShellCommandStore.open(
                str(self._processes_root), agent, conversation_id=self._conversation_id
            )
            self._bg_queue = ShellBackgroundQueue(
                executor=self._executor,
                cwd=self._initial_cwd,
                agent=agent,
                shell_cmds=self._shell_cmds,
                on_complete=on_complete,
            )

        if self._allowed_commands and self._denied_commands:
            raise ValueError("Specify allowed_commands or denied_commands, not both.")
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be a positive integer.")

        self.add_function(self.run_command, name="run_command")
        self.add_function(self.start_command, name="start_command")
        self.add_function(self.check_command, name="check_command")
        self.add_function(self.stop_command, name="stop_command")

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh instance per run so cwd and background processes are isolated.

        `get_toolset` builds one shared instance at agent construction (see
        `AbstractToolset.for_run`, which defaults to returning `self`). This
        toolset holds mutable per-run state (`_cwd`, `_background`), so without
        an override two concurrent runs would corrupt each other's cwd and kill
        each other's background processes.
        """
        return ShellToolset(
            cwd=self._initial_cwd,
            allowed_commands=self._allowed_commands,
            denied_commands=self._denied_commands,
            denied_operators=self._denied_operators,
            default_timeout=self._default_timeout,
            max_output_chars=self._max_output_chars,
            persist_cwd=self._persist_cwd,
            allow_interactive=self._allow_interactive,
            env=self._env,
            denied_env_patterns=self._denied_env_patterns,
            executor=self._executor,
            processes_root=self._processes_root,
            agent=self._agent,
            conversation_id=getattr(ctx, "conversation_id", None),
            on_complete=self._on_complete,
        )

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        """Enforce the model-visible output cap at the tool dispatch seam.

        Tools place control metadata (status, exit code, `start_command`'s ID
        line) at the end of their responses, so keeping the tail preserves it
        without any per-tool cases here. Only `str` results are capped; a
        future tool returning rich content (e.g. `ToolReturn`) needs this seam
        extended.
        """
        result = await super().call_tool(name, tool_args, ctx, tool)
        if not isinstance(result, str):
            return result
        return truncate_tail(result, self._max_output_chars)

    def _resolve_env(self) -> dict[str, str] | None:
        """Compute the environment passed to spawned subprocesses.

        Returns `None` -- meaning the subprocess inherits the parent env -- only
        when neither `env` nor `denied_env_patterns` is configured, so the
        default behavior is unchanged. An explicit `env` replaces inheritance
        entirely; `denied_env_patterns` then strips matching names (glob, via
        `fnmatch`) from whichever base applies, so the two compose: patterns
        filter an explicit `env` just as they filter the inherited environment.
        """
        if self._env is None and not self._denied_env_patterns:
            return None
        base = dict(self._env) if self._env is not None else dict(os.environ)
        if not self._denied_env_patterns:
            return base
        return {
            name: value
            for name, value in base.items()
            if not any(
                fnmatch.fnmatchcase(name, pattern)
                for pattern in self._denied_env_patterns
            )
        }

    async def __aexit__(self, *args: Any) -> None:
        """Terminate live in-memory processes; stop the supervisor cleanly.

        Detached background processes are NOT killed here (they survive the
        toolset and loop by design); we only stop the queue's supervisor tasks
        so their subprocess transports aren't GC'd after loop close.
        """
        for handle in self._background.values():
            await self._executor.stop(handle)
        self._background.clear()
        if self._bg_queue is not None:
            await self._bg_queue.shutdown()

    def _first_denied_operator(self, command: str) -> str | None:
        """Return the first denied operator found in command, or None."""
        return next((op for op in self._denied_operators if op in command), None)

    def _check_command(self, command: str) -> None:
        """Validate command against allow/deny lists.

        These checks are best-effort and are not a security boundary -- a
        sufficiently motivated agent can bypass them. Use OS-level isolation
        (containers, sandboxes) for hard enforcement.
        """
        if not self._allow_interactive and _is_interactive_command(command):
            raise PermissionError(
                f"Interactive commands are not allowed. Command: {command!r}"
            )

        matched_op = self._first_denied_operator(command)
        if matched_op:
            raise PermissionError(f"Shell operator {matched_op!r} is not allowed.")

        try:
            tokens = shlex.split(command)
        except ValueError:
            return
        if not tokens:
            return
        executable = tokens[0]

        if self._denied_commands and executable in self._denied_commands:
            raise PermissionError(f"Command {executable!r} is denied.")
        if self._allowed_commands and executable not in self._allowed_commands:
            raise PermissionError(f"Command {executable!r} is not in the allowed list.")

    def _build_cwd_capture(self, command: str) -> tuple[str, Path | None]:
        """Wrap a command to record its final working directory out-of-band.

        `pwd` is written to a private temp file whose random path the agent's
        command can't address, so command output can never spoof the tracked
        cwd -- unlike parsing a sentinel out of stdout, where any command that
        prints the sentinel string (or one using `;` to skip success-gating)
        could redirect the cwd. Returns the wrapped command plus the temp-file
        path, or the command unchanged and `None` when cwd tracking is off.
        """
        if not self._persist_cwd:
            return command, None
        fd, name = tempfile.mkstemp(prefix="harness_cwd_")
        os.close(fd)
        wrapped = (
            f"{command}\n__harness_ec=$?\npwd > {shlex.quote(name)}\nexit $__harness_ec"
        )
        return wrapped, Path(name)

    def _apply_captured_cwd(self, cwd_file: Path) -> None:
        """Update the persistent cwd from the capture file, ignoring junk."""
        try:
            recorded = cwd_file.read_text(encoding="utf-8").strip()
        except OSError:  # pragma: no cover
            return
        if not recorded:
            return
        candidate = Path(recorded)
        if candidate.is_dir():
            self._cwd = candidate

    @_recoverable
    async def run_command(
        self, command: str, *, timeout_seconds: float | None = None
    ) -> str:
        """Execute a shell command and return its output.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: 30).

        Returns:
            Labeled stdout/stderr output with exit code on non-zero exit.
        """
        self._check_command(command)
        timeout = (
            timeout_seconds if timeout_seconds is not None else self._default_timeout
        )

        actual_command, cwd_file = self._build_cwd_capture(command)
        try:
            result = await self._executor.run(
                actual_command,
                cwd=self._cwd,
                env=self._resolve_env(),
                timeout=timeout,
            )
            if result.timed_out:
                return f"[Command timed out after {timeout}s]"

            parts: list[str] = []
            if result.stdout:
                parts.append(f"[stdout]\n{result.stdout}")
            if result.stderr:
                parts.append(f"[stderr]\n{result.stderr}")
            output = "\n".join(parts) if parts else "(no output)"

            if cwd_file is not None and result.exit_code == 0:
                self._apply_captured_cwd(cwd_file)

            if result.exit_code != 0:
                output = f"{output}\n[exit code: {result.exit_code}]"
            return output
        finally:
            if cwd_file is not None:
                cwd_file.unlink(missing_ok=True)

    @_recoverable
    async def start_command(
        self, command: str, notify_on_completion: bool = False
    ) -> str:
        """Start a long-running command in the background (e.g. a server or watcher).

        The command runs as a DETACHED subprocess via the background queue, and
        its lifecycle (status, pid, exit code) is logged durably to both the
        run-event feed and the shell command store. Callers make a later
        ``check_command`` / ``stop_command`` with this returned run id.

        Args:
            command: The shell command to run in the background.

        Returns:
            A message containing the unique run id for later check/stop calls.
        """
        self._check_command(command)
        cid = self._conversation_id or "global"
        seq = self._next_seq(cid)
        run_id = f"shell.{cid}.{seq}"
        # Emit the "running" event synchronously so the live feed sees the
        # command instantly; the queue worker then spawns, records pid, and
        # supervises to completion.
        await get_agent_emitter(self._agent).emit_process(
            ProcessEvent(
                id=run_id,
                label=f"shell: {command}",
                status=ProcessEventType.running,
                detail=f"started: {command}",
                lane="conversation",
                scope=self._conversation_id,
            )
        )
        if self._bg_queue is not None:
            self._bg_queue.enqueue(
                run_id,
                ShellRun(
                    run_id=run_id,
                    command=command,
                    conversation_id=self._conversation_id,
                    cwd=str(self._cwd),
                    env=self._resolve_env(),
                    notify_on_completion=notify_on_completion,
                ),
            )
        else:
            handle = await self._executor.start(
                command,
                cwd=self._cwd,
                env=self._resolve_env(),
            )
            self._background[handle.command_id] = handle
            return f"Started background command: {command!r}\nID: {handle.command_id}"
        return f"Started background command: {command!r}\nID: {run_id}"

    def _next_seq(self, cid: str) -> int:
        """Next per-conversation sequence for ``shell.{cid}.{seq}`` run ids.

        Derived from the durable store (scans existing ``shell.{cid}.*`` rich
        lines) so the id stays unique across turns/restarts — the per-run
        toolset has no in-memory counter to rely on.
        """
        if self._shell_cmds is None:
            return 1
        prefix = f"shell.{cid}."
        max_seq = 0
        for rec in self._shell_cmds.read_all():
            rid = rec.get("run_id", "")
            if not rid.startswith(prefix):
                continue
            tail = rid[len(prefix) :]
            if tail.isdigit():
                max_seq = max(max_seq, int(tail))
        return max_seq + 1

    @_recoverable
    async def check_command(self, command_id: str) -> str:
        """Check the status and recent output of a background command.

        Works for both live (in-memory) handles and detached survivors: when no
        live executor handle is present, status is derived from the durable
        shell store (join key = run id) plus a pid-liveness check. If the last
        durable line still says ``running`` but the process is gone (e.g. the
        supervisor died with the turn before writing the terminal line), this
        RESOLVES the staleness by appending a terminal ``done``/``error`` line
        (append-only — never mutates a prior line) and reports the finished
        status.

        Args:
            command_id: The id returned by start_command.

        Returns:
            Status and available output/exit code of the background command.
        """
        # Fastest, richest path: a live executor handle in this instance.
        handle = self._background.get(command_id)
        if handle is not None:
            return self._format_check(await self._executor.check(handle))

        # Durable path: read the rich shell store; resolve stale `running`.
        lines = self._shell_cmds.all_for(command_id) if self._shell_cmds else []
        if not lines:
            return f"[Error: unknown command ID {command_id!r}]"
        last = lines[-1]
        status, exit_code = self._derive_status(last, command_id)
        if status.status == "finished" and last.get("status") in ("running", None):
            # Supervisor never wrote the terminal line; journal it now.
            await self._append_terminal(command_id, status)
        return self._format_check(status)

    @_recoverable
    async def stop_command(self, command_id: str) -> str:
        """Stop a background command and return its final output.

        Stops a live in-memory handle via the executor; otherwise, for a
        detached survivor, terminates its process group by pid, appends a
        terminal ``stopped`` line to the durable store, and reports it.

        Args:
            command_id: The id returned by start_command.

        Returns:
            Final status of the stopped command.
        """
        handle = self._background.get(command_id)
        if handle is not None:
            status = await self._executor.stop(handle)
            del self._background[command_id]
            return self._format_stop(status)

        if not self._shell_cmds:
            return f"[Error: unknown command ID {command_id!r}]"
        lines = self._shell_cmds.all_for(command_id)
        if not lines:
            return f"[Error: unknown command ID {command_id!r}]"
        last = lines[-1]
        pid = last.get("pid")
        if pid is not None and self._pid_alive(pid):
            self._kill_pid(pid)
        await self._append_terminal(
            command_id,
            BackgroundStatus(status="finished", stdout="", stderr="", exit_code=0),
            detail=f"stopped: {last.get('command', '')}",
        )
        return self._format_stop(
            BackgroundStatus(status="finished", stdout="", stderr="", exit_code=0)
        )

    def _derive_status(
        self, last: dict[str, Any], run_id: str
    ) -> tuple[BackgroundStatus, int | None]:
        """(BackgroundStatus, exit_code) resolved from the last rich line."""
        st = last.get("status")
        if st in ("done", "error"):
            return (
                BackgroundStatus(
                    status="finished",
                    stdout="",
                    stderr="",
                    exit_code=last.get("exit_code"),
                ),
                last.get("exit_code"),
            )
        pid = last.get("pid")
        # Running line but process is gone → finished (never-mutated; resolve by append).
        if pid is not None and not self._pid_alive(pid):
            return BackgroundStatus(
                status="finished", stdout="", stderr="", exit_code=None
            ), None
        return BackgroundStatus(status="running", stdout="", stderr=""), None

    async def _append_terminal(
        self, run_id: str, status: BackgroundStatus, *, detail: str = ""
    ) -> None:
        """Append a terminal rich line and emit a live terminal event."""
        command = run_id
        if self._shell_cmds is not None:
            lines = self._shell_cmds.all_for(run_id)
            if not lines:
                return
            base = lines[0]
            command = base.get("command", run_id)
            exit_code = status.exit_code
            # exit_code None (unknown, vanished process) → treat as error, not done.
            terminal_status = "done" if exit_code == 0 else "error"
            self._shell_cmds.append(
                {
                    "run_id": run_id,
                    "command": command,
                    "agent": self._agent,
                    "conversation_id": base.get(
                        "conversation_id", self._conversation_id
                    ),
                    "timestamp": self._queue_now(),
                    "status": terminal_status,
                    "exit_code": exit_code,
                }
            )
        exit_code = status.exit_code
        terminal_status = "done" if exit_code == 0 else "error"
        await get_agent_emitter(self._agent).emit_process(
            ProcessEvent(
                id=run_id,
                label=f"shell: {command}",
                status=ProcessEventType(terminal_status),
                detail=detail or f"exit {exit_code}",
                lane="conversation",
                scope=self._conversation_id,
            )
        )

    @staticmethod
    def _format_check(status: BackgroundStatus) -> str:
        parts = [f"[status: {status.status}]"]
        if status.stdout:
            parts.append(f"[stdout]\n{status.stdout}")
        if status.stderr:
            parts.append(f"[stderr]\n{status.stderr}")
        if status.status == "finished" and status.exit_code is not None:
            parts.append(f"[exit code: {status.exit_code}]")
        if len(parts) == 1:
            parts.insert(0, "(no output yet)")
        return "\n".join(parts)

    def _format_stop(self, status: BackgroundStatus) -> str:
        parts = ["[stopped]"]
        if status.stdout:
            parts.append(f"[{status.stdout}]")
        if status.exit_code is not None:
            parts.append(f"[exit code: {status.exit_code}]")
        return "\n".join(parts)

    @staticmethod
    def _queue_now() -> str:
        from datetime import UTC, datetime

        return datetime.now(UTC).replace(microsecond=0).isoformat()

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    @staticmethod
    def _kill_pid(pid: int) -> None:
        """SIGTERM the process group of pid (best-effort, for detached re-attach)."""
        import signal

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
