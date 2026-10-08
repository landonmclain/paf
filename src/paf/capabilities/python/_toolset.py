"""Python toolset: one ``run_python`` tool over the sandbox seam.

The code is shipped to the interpreter as a base64 argv value and compiled under
the filename ``<run_python>``, so nothing is written to disk (the sandbox root
and workspace are read-only) and there is no shell quoting for the model to get
wrong. ``python -I`` (isolated mode) ignores ``PYTHON*`` environment variables
and the user site. The run is wrapped in ``timeout`` so the interpreter is
killed inside the sandbox, not just abandoned by the host.
"""

from __future__ import annotations

import base64
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from paf.capabilities._sandbox import LocalProcessExecutor, SandboxedExecutor
from paf.capabilities.shell._output import truncate_tail

# Exit codes ``timeout(1)`` and the kernel produce when the run is cut short.
_EXIT_TIMEOUT = 124
_EXIT_KILLED = 137

# Runs the base64 program in argv[1]. The traceback drops this bootstrap frame
# so the model only sees frames from its own code, with source lines.
_BOOTSTRAP = """\
import base64, linecache, sys, traceback
src = base64.b64decode(sys.argv[1]).decode()
linecache.cache['<run_python>'] = (len(src), None, src.splitlines(True), '<run_python>')
try:
    exec(compile(src, '<run_python>', 'exec'), {'__name__': '__main__'})
except SystemExit:
    raise
except BaseException:
    t, v, tb = sys.exc_info()
    traceback.print_exception(t, v, tb.tb_next)
    sys.exit(1)
"""


class PythonToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent a sandboxed ``run_python`` tool."""

    def __init__(
        self,
        *,
        cwd: Path,
        executor: SandboxedExecutor,
        default_timeout: float = 30.0,
        max_timeout: float = 120.0,
        max_output_chars: int = 20_000,
        max_code_chars: int = 60_000,
        env: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        if min(default_timeout, max_timeout, max_output_chars, max_code_chars) <= 0:
            raise ValueError("timeouts and size caps must be positive.")
        self._cwd = cwd
        self._executor = executor
        self._default_timeout = default_timeout
        self._max_timeout = max_timeout
        self._max_output_chars = max_output_chars
        self._max_code_chars = max_code_chars
        self._env = dict(env) if env is not None else None
        self.add_function(self.run_python, name="run_python")

    def _command(self, code: str, timeout: float) -> str:
        interpreter = (
            shlex.quote(sys.executable)
            if isinstance(self._executor, LocalProcessExecutor)
            else "python"
        )
        payload = base64.b64encode(code.encode("utf-8")).decode("ascii")
        return (
            f"timeout -k 2 {int(timeout)} {interpreter} -I "
            f"-c {shlex.quote(_BOOTSTRAP)} {payload}"
        )

    async def run_python(self, code: str, timeout_seconds: float | None = None) -> str:
        """Run a Python 3 program in an isolated sandbox and return its output.

        Each call is a fresh interpreter: no variables or imports carry over, so
        put everything one run needs in `code`. Only the standard library is
        available. There is no network. The working directory is the workspace,
        which is read-only except where the deployment allows writes; `/tmp` is
        scratch space that is discarded after the call. Print what you want to
        see: only stdout, stderr and the exit code come back.

        Args:
            code: The complete Python program to run.
            timeout_seconds: Maximum seconds to wait (default 30, capped by the
                deployment).

        Returns:
            Labeled stdout/stderr, plus the exit code when it is non-zero. A
            failing program returns its traceback.
        """
        if not code.strip():
            raise ModelRetry("`code` is empty; pass the Python program to run.")
        if len(code) > self._max_code_chars:
            raise ModelRetry(
                f"`code` is {len(code)} chars; the limit is {self._max_code_chars}. "
                "Split the work into smaller programs."
            )
        timeout = min(
            max(timeout_seconds or self._default_timeout, 1.0), self._max_timeout
        )
        result = await self._executor.run(
            self._command(code, timeout),
            cwd=self._cwd,
            env=self._env,
            # Backstop only: ``timeout`` inside the sandbox fires first.
            timeout=timeout + 10.0,
        )
        if result.timed_out or result.exit_code == _EXIT_TIMEOUT:
            return truncate_tail(
                self._format(result.stdout, result.stderr)
                + f"\n[timed out after {int(timeout)}s]",
                self._max_output_chars,
            )
        output = self._format(result.stdout, result.stderr)
        if result.exit_code == _EXIT_KILLED:
            output += "\n[killed: timeout or memory limit exceeded]"
        elif result.exit_code != 0:
            output += f"\n[exit code: {result.exit_code}]"
        return truncate_tail(output, self._max_output_chars)

    @staticmethod
    def _format(stdout: str, stderr: str) -> str:
        parts = []
        if stdout:
            parts.append(f"[stdout]\n{stdout}")
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        return "\n".join(parts) if parts else "(no output)"
