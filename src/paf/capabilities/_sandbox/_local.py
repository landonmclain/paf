"""LocalProcessExecutor: the current host behavior, no sandbox.

Wraps the shell toolset's existing ``anyio.open_process`` execution so the seam
can be introduced with zero behavior change. Used for read-only tools, or when
no sandbox is configured.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import uuid
from pathlib import Path

import anyio
import anyio.abc

from paf.capabilities._sandbox._protocol import (
    BackgroundHandle,
    BackgroundStatus,
    ExecResult,
)

_IO_DRAIN_TIMEOUT: float = 2.0
_KILL_GRACE_PERIOD: float = 2.0


class _LocalBackgroundState:
    """State for a background process using temp files for output."""

    __slots__ = ("proc", "stdout_path", "stderr_path", "finished", "exit_code")

    def __init__(
        self, proc: anyio.abc.Process, stdout_path: str, stderr_path: str | None
    ) -> None:
        self.proc = proc
        self.stdout_path = stdout_path
        # None = stderr is merged into a caller-owned stdout file (never deleted).
        self.stderr_path = stderr_path
        self.finished = False
        self.exit_code: int | None = None


class LocalProcessExecutor:
    """Runs commands as local subprocesses on the host (no sandbox)."""

    async def run(
        self,
        command: str,
        *,
        cwd: Path,
        env: dict[str, str] | None,
        timeout: float,
    ) -> ExecResult:
        proc = await anyio.open_process(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=env,
        )
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        try:
            assert proc.stdout is not None
            assert proc.stderr is not None

            async def _read_stdout() -> None:
                assert proc.stdout is not None
                async for chunk in proc.stdout:
                    stdout_chunks.append(chunk)

            async def _read_stderr() -> None:
                assert proc.stderr is not None
                async for chunk in proc.stderr:
                    stderr_chunks.append(chunk)

            with anyio.fail_after(timeout):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(_read_stdout)
                    tg.start_soon(_read_stderr)
                await proc.wait()
        except TimeoutError:
            await self._kill_process_group(proc)
            with anyio.CancelScope(shield=True):
                await proc.wait()
                await self._drain_with_timeout(stdout_chunks, stderr_chunks, proc)
            return ExecResult(stdout="", stderr="", exit_code=-1, timed_out=True)
        finally:
            await proc.aclose()

        stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        exit_code = proc.returncode if proc.returncode is not None else 0
        return ExecResult(stdout=stdout, stderr=stderr, exit_code=exit_code)

    async def start(
        self,
        command: str,
        *,
        cwd: Path,
        env: dict[str, str] | None,
        output_path: Path | None = None,
    ) -> BackgroundHandle:
        command_id = uuid.uuid4().hex[:12]
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as out:
                proc = await anyio.open_process(
                    command,
                    cwd=cwd,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
            state = _LocalBackgroundState(proc, str(output_path), None)
            return BackgroundHandle(command_id=command_id, state=state)
        stdout_file = tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f"harness_{command_id}_out_", delete=False
        )
        stderr_file = tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f"harness_{command_id}_err_", delete=False
        )
        try:
            proc = await anyio.open_process(
                command,
                cwd=cwd,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
                env=env,
            )
        except BaseException:
            stdout_file.close()
            stderr_file.close()
            os.unlink(stdout_file.name)
            os.unlink(stderr_file.name)
            raise

        stdout_file.close()
        stderr_file.close()
        state = _LocalBackgroundState(proc, stdout_file.name, stderr_file.name)
        return BackgroundHandle(command_id=command_id, state=state)

    async def check(self, handle: BackgroundHandle) -> BackgroundStatus:
        state: _LocalBackgroundState = handle.state
        if not state.finished and state.proc.returncode is not None:
            state.exit_code = state.proc.returncode
            state.finished = True
        stdout, stderr = self._read_bg_output(state)
        status = "finished" if state.finished else "running"
        return BackgroundStatus(
            status=status, stdout=stdout, stderr=stderr, exit_code=state.exit_code
        )

    async def stop(self, handle: BackgroundHandle) -> BackgroundStatus:
        state: _LocalBackgroundState = handle.state
        if not state.finished:
            await self._kill_process_group(state.proc)
            with anyio.CancelScope(shield=True):
                await state.proc.wait()
            state.exit_code = state.proc.returncode
            state.finished = True
        stdout, stderr = self._read_bg_output(state)
        self._cleanup_bg_files(state)
        await state.proc.aclose()
        return BackgroundStatus(
            status="finished", stdout=stdout, stderr=stderr, exit_code=state.exit_code
        )

    async def _kill_process_group(self, proc: anyio.abc.Process) -> None:
        """SIGTERM the process group, escalating to SIGKILL after the grace period."""
        pid = proc.pid
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            return

        with anyio.move_on_after(_KILL_GRACE_PERIOD):
            await proc.wait()
            return

        # Still alive after grace period -- hard kill
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    async def _drain_with_timeout(
        self,
        stdout_chunks: list[bytes],
        stderr_chunks: list[bytes],
        proc: anyio.abc.Process,
    ) -> None:
        """Drain remaining pipe data after kill (grandchildren may still hold the pipe)."""

        async def _drain_stdout() -> None:
            if proc.stdout is None:
                return
            try:
                async for chunk in proc.stdout:
                    stdout_chunks.append(chunk)
            except (anyio.ClosedResourceError, anyio.BrokenResourceError):
                pass

        async def _drain_stderr() -> None:
            if proc.stderr is None:
                return
            try:
                async for chunk in proc.stderr:
                    stderr_chunks.append(chunk)
            except (anyio.ClosedResourceError, anyio.BrokenResourceError):
                pass

        with anyio.move_on_after(_IO_DRAIN_TIMEOUT):
            async with anyio.create_task_group() as tg:
                tg.start_soon(_drain_stdout)
                tg.start_soon(_drain_stderr)

    def _read_bg_output(self, state: _LocalBackgroundState) -> tuple[str, str]:
        try:
            stdout = Path(state.stdout_path).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            stdout = ""
        if state.stderr_path is None:
            return stdout, ""
        try:
            stderr = Path(state.stderr_path).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            stderr = ""
        return stdout, stderr

    def _cleanup_bg_files(self, state: _LocalBackgroundState) -> None:
        if state.stderr_path is None:
            return  # caller-owned output file: kept
        try:
            os.unlink(state.stdout_path)
        except OSError:
            pass
        try:
            os.unlink(state.stderr_path)
        except OSError:
            pass
