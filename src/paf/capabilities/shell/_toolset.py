"""Shell toolset -- gives agents the ability to run commands.

paf-owned shell toolset (copied verbatim from pydantic-ai-harness, MIT).

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

from paf.capabilities._sandbox import (
    BackgroundHandle,
    LocalProcessExecutor,
    SandboxedExecutor,
)

from ._output import truncate_tail

_P = ParamSpec('_P')


def _recoverable(  # noqa: UP047  # paf-hardened: also recovers spawn failures (upstream 0.26.0)
    fn: Callable[Concatenate[ShellToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[ShellToolset, _P], Awaitable[str]]:
    """Convert model-correctable errors into `ModelRetry`.

    pyai only feeds `ModelRetry` back to the model as a retry prompt; any other
    exception propagates and aborts the whole run. A denied command is something
    the model can recover from (pick an allowed one), so surface it as a retry
    instead of crashing the agent. paf also recovers spawn failures (a missing
    command binary raises `FileNotFoundError`/`OSError`), so a typo'd command
    returns to the model instead of aborting the run.
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
        r'^(vi|vim|nano|emacs|less|more|top|htop|man)\b',
        r'^sudo\s',
        r'^passwd\b',
        r'^ssh\b',
        r'^telnet\b',
        r'^ftp\b',
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
    ) -> None:
        super().__init__()
        self._cwd = cwd.resolve()
        # The configured starting directory, never mutated by persist_cwd, so
        # `for_run` can hand each run a fresh instance rooted back here.
        self._initial_cwd = self._cwd
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

        if self._allowed_commands and self._denied_commands:
            raise ValueError('Specify allowed_commands or denied_commands, not both.')
        if max_output_chars <= 0:
            raise ValueError('max_output_chars must be a positive integer.')

        self.add_function(self.run_command, name='run_command')
        self.add_function(self.start_command, name='start_command')
        self.add_function(self.check_command, name='check_command')
        self.add_function(self.stop_command, name='stop_command')

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
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in self._denied_env_patterns)
        }

    async def __aexit__(self, *args: Any) -> None:
        """Terminate all remaining background processes and clean up."""
        for handle in self._background.values():
            await self._executor.stop(handle)
        self._background.clear()

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
            raise PermissionError(f'Interactive commands are not allowed. Command: {command!r}')

        matched_op = self._first_denied_operator(command)
        if matched_op:
            raise PermissionError(f'Shell operator {matched_op!r} is not allowed.')

        try:
            tokens = shlex.split(command)
        except ValueError:
            return
        if not tokens:
            return
        executable = tokens[0]

        if self._denied_commands and executable in self._denied_commands:
            raise PermissionError(f'Command {executable!r} is denied.')
        if self._allowed_commands and executable not in self._allowed_commands:
            raise PermissionError(f'Command {executable!r} is not in the allowed list.')

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
        fd, name = tempfile.mkstemp(prefix='harness_cwd_')
        os.close(fd)
        wrapped = f'{command}\n__harness_ec=$?\npwd > {shlex.quote(name)}\nexit $__harness_ec'
        return wrapped, Path(name)

    def _apply_captured_cwd(self, cwd_file: Path) -> None:
        """Update the persistent cwd from the capture file, ignoring junk."""
        try:
            recorded = cwd_file.read_text(encoding='utf-8').strip()
        except OSError:  # pragma: no cover
            return
        if not recorded:
            return
        candidate = Path(recorded)
        if candidate.is_dir():
            self._cwd = candidate

    @_recoverable
    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str:
        """Execute a shell command and return its output.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: 30).

        Returns:
            Labeled stdout/stderr output with exit code on non-zero exit.
        """
        self._check_command(command)
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout

        actual_command, cwd_file = self._build_cwd_capture(command)
        try:
            result = await self._executor.run(
                actual_command,
                cwd=self._cwd,
                env=self._resolve_env(),
                timeout=timeout,
            )
            if result.timed_out:
                return f'[Command timed out after {timeout}s]'

            parts: list[str] = []
            if result.stdout:
                parts.append(f'[stdout]\n{result.stdout}')
            if result.stderr:
                parts.append(f'[stderr]\n{result.stderr}')
            output = '\n'.join(parts) if parts else '(no output)'

            if cwd_file is not None and result.exit_code == 0:
                self._apply_captured_cwd(cwd_file)

            if result.exit_code != 0:
                output = f'{output}\n[exit code: {result.exit_code}]'
            return output
        finally:
            if cwd_file is not None:
                cwd_file.unlink(missing_ok=True)

    @_recoverable
    async def start_command(self, command: str) -> str:
        """Start a long-running command in the background (e.g. a server or watcher).

        Callers MUST call `stop_command(command_id)` when done to terminate the
        process and clean up temporary output files.

        Args:
            command: The shell command to run in the background.

        Returns:
            A message containing the unique command ID for later check/stop calls.
        """
        self._check_command(command)
        handle = await self._executor.start(
            command,
            cwd=self._cwd,
            env=self._resolve_env(),
        )
        self._background[handle.command_id] = handle
        return f'Started background command: {command!r}\nID: {handle.command_id}'

    async def check_command(self, command_id: str) -> str:
        """Check the status and recent output of a background command.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Status and recent output of the background command.
        """
        handle = self._background.get(command_id)
        if handle is None:
            return f'[Error: unknown command ID {command_id!r}]'

        status = await self._executor.check(handle)

        output_sections: list[str] = []
        if status.stdout:
            output_sections.append(f'[stdout]\n{status.stdout}')
        if status.stderr:
            output_sections.append(f'[stderr]\n{status.stderr}')
        parts = ['\n'.join(output_sections) if output_sections else '(no output yet)', f'[status: {status.status}]']
        if status.status == 'finished' and status.exit_code is not None:
            parts.append(f'[exit code: {status.exit_code}]')
        return '\n'.join(parts)

    async def stop_command(self, command_id: str) -> str:
        """Stop a background command and return its final output.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Final output and exit status of the stopped command.
        """
        handle = self._background.get(command_id)
        if handle is None:
            return f'[Error: unknown command ID {command_id!r}]'

        status = await self._executor.stop(handle)
        del self._background[command_id]

        output_sections: list[str] = []
        if status.stdout:
            output_sections.append(f'[stdout]\n{status.stdout}')
        if status.stderr:
            output_sections.append(f'[stderr]\n{status.stderr}')
        parts = ['\n'.join(output_sections) if output_sections else '(no output)', '[stopped]']
        if status.exit_code is not None:
            parts.append(f'[exit code: {status.exit_code}]')
        return '\n'.join(parts)
