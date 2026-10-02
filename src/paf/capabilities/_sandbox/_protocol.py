"""The sandbox seam: :class:`SandboxedExecutor` and its result/handle types.

A capability that runs untrusted code (shell, ``run_tests``, ``install_package``,
git, codebase_map) talks to this seam -- never to a concrete backend -- so the
execution boundary can be swapped (local process now, Docker / Windows sandbox
later) without touching the capability's tool surface. Modeled on
``capabilities/_storage``, which is the same shape: a protocol seam plus
swappable backends that many capabilities import.

The seam is deliberately small: four async methods. ``run`` is the synchronous
path; ``start`` / ``check`` / ``stop`` are the background-process lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ExecResult:
    """Outcome of a synchronous ``run``."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


@dataclass
class BackgroundHandle:
    """Opaque handle to a background process, owned by the backend.

    The toolset keeps ``command_id -> BackgroundHandle`` bookkeeping and passes
    the handle back to ``check`` / ``stop``. The backend stores whatever it
    needs to find the process in ``state``; the toolset never inspects it.
    """

    command_id: str
    state: Any = field(repr=False, default=None)


@dataclass
class BackgroundStatus:
    """Snapshot of a background process at ``check`` time."""

    status: Literal["running", "finished"]
    stdout: str
    stderr: str
    exit_code: int | None = None


class SandboxedExecutor(Protocol):
    """The execution boundary every code-running tool routes through.

    Implementations are responsible for spawning the process, applying the
    sandbox policy, and the background-process lifecycle. ``env`` of ``None``
    means the subprocess inherits the parent environment.
    """

    async def run(
        self,
        command: str,
        *,
        cwd: Path,
        env: dict[str, str] | None,
        timeout: float,
    ) -> ExecResult: ...

    async def start(
        self,
        command: str,
        *,
        cwd: Path,
        env: dict[str, str] | None,
        output_path: Path | None = None,
    ) -> BackgroundHandle:
        """Start a detached process.

        With ``output_path`` its combined stdout+stderr goes to that file, which
        the caller owns (``check``/``stop`` read it; ``stop`` never deletes it),
        so the output outlives the process and this server. Without it the
        backend keeps output wherever it likes and cleans it up on ``stop``.
        """
        ...

    async def check(self, handle: BackgroundHandle) -> BackgroundStatus: ...

    async def stop(self, handle: BackgroundHandle) -> BackgroundStatus:
        """Terminate the process and return its final status.

        ``stop`` both forces termination and reports the final output/exit code,
        so a caller (e.g. the shell toolset's ``stop_command``) can surface the
        process's last state in one call instead of racing ``check`` against the
        cleanup.
        """
        ...
