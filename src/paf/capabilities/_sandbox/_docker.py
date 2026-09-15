"""DockerExecutor: the sandbox backend (Linux / macOS / Windows via WSL2).

Runs commands inside a locked-down Docker container: no network, read-only
filesystem, non-root user, resource limits. Two modes:

- ``mode="exec"`` (default): one long-lived container, ``docker exec`` per
  command (~60ms). State persists between commands.
- ``mode="run"``: fresh ``docker run --rm`` per command (~250ms). Clean slate
  each time, strongest isolation.

Background processes map to detached containers; ``check`` / ``stop`` become
``docker logs`` / ``docker stop``.
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import anyio

from paf.capabilities._sandbox._protocol import (
    BackgroundHandle,
    BackgroundStatus,
    ExecResult,
)


@dataclass
class _DockerBackgroundState:
    """State for a background process: the detached container's name/id."""

    container_id: str


class DockerExecutor:
    """Runs commands inside a locked-down Docker container."""

    def __init__(
        self,
        *,
        image: str = 'python:3.12-slim',
        mode: str = 'exec',
        network: str = 'none',
        user: str = '1000:1000',
        read_only: bool = True,
        memory: str = '1g',
        cpus: str = '1.0',
        pids_limit: int = 256,
        workspace: str = '/workspace',
        container_name: str | None = None,
    ) -> None:
        if mode not in ('exec', 'run'):
            raise ValueError("mode must be 'exec' or 'run'")
        self._image = image
        self._mode = mode
        self._network = network
        self._user = user
        self._read_only = read_only
        self._memory = memory
        self._cpus = cpus
        self._pids_limit = pids_limit
        self._workspace = workspace
        self._container_name = container_name
        self._container_id: str | None = None

    def _base_run_args(self) -> list[str]:
        args = [
            '--network', self._network,
            '--user', self._user,
            '--memory', self._memory,
            '--cpus', self._cpus,
            '--pids-limit', str(self._pids_limit),
        ]
        if self._read_only:
            args.append('--read-only')
        return args

    async def _docker(self, args: list[str], *, timeout: float = 60.0) -> tuple[int, str, str]:
        """Run a docker CLI command and return (exit_code, stdout, stderr)."""
        proc = await anyio.open_process(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
        finally:
            await proc.aclose()

        stdout = b''.join(stdout_chunks).decode('utf-8', errors='replace')
        stderr = b''.join(stderr_chunks).decode('utf-8', errors='replace')
        exit_code = proc.returncode if proc.returncode is not None else 0
        return exit_code, stdout, stderr

    async def _ensure_container(self, cwd: Path) -> None:
        """Create the long-lived container for ``mode='exec'`` on first use."""
        if self._container_id is not None:
            return
        name = self._container_name or f'paf-sandbox-{uuid.uuid4().hex[:8]}'
        args = [
            'docker', 'run', '-d', '--name', name,
            *self._base_run_args(),
            '-v', f'{cwd}:{self._workspace}', '-w', self._workspace,
            self._image, 'sleep', 'infinity',
        ]
        code, out, err = await self._docker(args)
        if code != 0:
            raise RuntimeError(
                f'Failed to start sandbox container: {err.strip() or out.strip()}'
            )
        self._container_id = out.strip()
        self._container_name = name

    async def run(
        self,
        command: str,
        *,
        cwd: Path,
        env: dict[str, str] | None,
        timeout: float,
    ) -> ExecResult:
        if self._mode == 'exec':
            await self._ensure_container(cwd)
            args = ['docker', 'exec', self._container_id, 'sh', '-c', command]
        else:
            args = [
                'docker', 'run', '--rm',
                *self._base_run_args(),
                '-v', f'{cwd}:{self._workspace}', '-w', self._workspace,
                self._image, 'sh', '-c', command,
            ]
        code, out, err = await self._docker(args, timeout=timeout)
        return ExecResult(stdout=out, stderr=err, exit_code=code)

    async def start(
        self,
        command: str,
        *,
        cwd: Path,
        env: dict[str, str] | None,
    ) -> BackgroundHandle:
        command_id = uuid.uuid4().hex[:12]
        name = f'paf-bg-{command_id}'
        args = [
            'docker', 'run', '-d', '--name', name,
            *self._base_run_args(),
            '-v', f'{cwd}:{self._workspace}', '-w', self._workspace,
            self._image, 'sh', '-c', command,
        ]
        code, out, err = await self._docker(args)
        if code != 0:
            raise RuntimeError(
                f'Failed to start background container: {err.strip() or out.strip()}'
            )
        state = _DockerBackgroundState(container_id=name)
        return BackgroundHandle(command_id=command_id, state=state)

    async def check(self, handle: BackgroundHandle) -> BackgroundStatus:
        state: _DockerBackgroundState = handle.state
        _, running_out, _ = await self._docker(
            ['docker', 'inspect', '-f', '{{.State.Running}}', state.container_id]
        )
        is_running = running_out.strip() == 'true'
        _, logs_out, logs_err = await self._docker(['docker', 'logs', state.container_id])
        status = 'running' if is_running else 'finished'
        exit_code: int | None = None
        if not is_running:
            _, code_out, _ = await self._docker(
                ['docker', 'inspect', '-f', '{{.State.ExitCode}}', state.container_id]
            )
            try:
                exit_code = int(code_out.strip())
            except ValueError:
                exit_code = None
        return BackgroundStatus(
            status=status, stdout=logs_out, stderr=logs_err, exit_code=exit_code
        )

    async def stop(self, handle: BackgroundHandle) -> BackgroundStatus:
        state: _DockerBackgroundState = handle.state
        await self._docker(['docker', 'stop', state.container_id])
        _, logs_out, logs_err = await self._docker(['docker', 'logs', state.container_id])
        _, code_out, _ = await self._docker(
            ['docker', 'inspect', '-f', '{{.State.ExitCode}}', state.container_id]
        )
        await self._docker(['docker', 'rm', state.container_id])
        try:
            exit_code = int(code_out.strip())
        except ValueError:
            exit_code = None
        return BackgroundStatus(
            status='finished', stdout=logs_out, stderr=logs_err, exit_code=exit_code
        )
