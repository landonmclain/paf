"""Unit tests for the shared sandbox seam (paf.capabilities._sandbox).

Covers the two backends: :class:`LocalProcessExecutor` (the current host
behavior, no sandbox) and :class:`DockerExecutor` (the sandbox backend, skipped
when Docker is unavailable). Follows the repo convention: sync ``def test_*``
functions that drive async scenarios via ``asyncio.run(...)``.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest

from paf.capabilities._sandbox import (
    BackgroundHandle,
    BackgroundStatus,
    DockerExecutor,
    ExecResult,
    LocalProcessExecutor,
)


def _run(coro):
    return asyncio.run(coro)


# --- LocalProcessExecutor: run -------------------------------------------------
def test_local_run_captures_stdout_stderr_and_exit_code(tmp_path):
    async def scenario() -> None:
        ex = LocalProcessExecutor()
        res = await ex.run(
            "echo hello; echo oops >&2; exit 3",
            cwd=tmp_path,
            env=None,
            timeout=10.0,
        )
        assert isinstance(res, ExecResult)
        assert res.stdout.strip() == "hello"
        assert res.stderr.strip() == "oops"
        assert res.exit_code == 3
        assert res.timed_out is False

    _run(scenario())


def test_local_run_timeout_kills_process_group(tmp_path):
    async def scenario() -> None:
        ex = LocalProcessExecutor()
        res = await ex.run("sleep 30", cwd=tmp_path, env=None, timeout=0.2)
        assert res.timed_out is True
        assert res.exit_code == -1

    _run(scenario())


def test_local_run_inherits_env_when_none(tmp_path):
    async def scenario() -> None:
        ex = LocalProcessExecutor()
        res = await ex.run("echo $HOME", cwd=tmp_path, env=None, timeout=10.0)
        assert res.stdout.strip()  # inherited env present

    _run(scenario())


def test_local_run_explicit_env(tmp_path):
    async def scenario() -> None:
        ex = LocalProcessExecutor()
        res = await ex.run("echo $FOO", cwd=tmp_path, env={"FOO": "bar"}, timeout=10.0)
        assert res.stdout.strip() == "bar"

    _run(scenario())


# --- LocalProcessExecutor: background lifecycle --------------------------------
def test_local_background_start_check_stop(tmp_path):
    async def scenario() -> None:
        ex = LocalProcessExecutor()
        handle = await ex.start("echo hi; sleep 0.2; echo done", cwd=tmp_path, env=None)
        assert isinstance(handle, BackgroundHandle)
        assert handle.command_id

        # Immediately after start it should be running (or already finished).
        status = await ex.check(handle)
        assert isinstance(status, BackgroundStatus)
        assert status.status in ("running", "finished")

        # Wait for it to finish, then confirm output + exit code.
        for _ in range(50):
            status = await ex.check(handle)
            if status.status == "finished":
                break
            await asyncio.sleep(0.05)
        assert status.status == "finished"
        assert "hi" in status.stdout
        assert "done" in status.stdout
        assert status.exit_code == 0

        # stop on a finished process is a no-op that cleans up.
        await ex.stop(handle)

    _run(scenario())


def test_local_background_stop_terminates_running(tmp_path):
    async def scenario() -> None:
        ex = LocalProcessExecutor()
        handle = await ex.start("sleep 30", cwd=tmp_path, env=None)
        status = await ex.check(handle)
        assert status.status == "running"
        await ex.stop(handle)
        status = await ex.check(handle)
        assert status.status == "finished"

    _run(scenario())


# --- Protocol conformance ------------------------------------------------------
def test_local_executor_conforms_to_sandboxed_executor():
    ex = LocalProcessExecutor()
    for attr in ("run", "start", "check", "stop"):
        assert callable(getattr(ex, attr, None)), attr


# --- Overlay socket guard (pure unit test, no Docker needed) -------------------
def test_docker_overlay_refuses_host_socket(tmp_path):
    """The overlay must never mount the host Docker socket into the sandbox.

    A sandboxed ``docker`` command must not reach the host daemon, so even if an
    exclude pattern resolves to the socket (e.g. via a symlink in the workspace)
    the overlay refuses to build the mount.
    """
    sock = tmp_path / "dockersock"
    try:
        sock.symlink_to("/var/run/docker.sock")
    except OSError:
        pytest.skip("cannot create symlink")
    ex = DockerExecutor(exclude=["dockersock"])
    with pytest.raises(RuntimeError, match="Docker socket"):
        ex._overlay_mounts(tmp_path)


# --- DockerExecutor (skipped when Docker is unavailable) -----------------------
def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
        return True
    except (subprocess.SubprocessError, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="Docker daemon not available"
)


def test_docker_run_captures_output(tmp_path):
    async def scenario() -> None:
        ex = DockerExecutor(mode="run")
        res = await ex.run("echo hello", cwd=tmp_path, env=None, timeout=30.0)
        assert res.exit_code == 0
        assert "hello" in res.stdout

    _run(scenario())


def test_docker_exec_mode_persists_state(tmp_path):
    async def scenario() -> None:
        ex = DockerExecutor(mode="exec")
        await ex.run(
            "echo one > /workspace/state.txt", cwd=tmp_path, env=None, timeout=30.0
        )
        res = await ex.run(
            "cat /workspace/state.txt", cwd=tmp_path, env=None, timeout=30.0
        )
        assert res.exit_code == 0
        assert "one" in res.stdout

    _run(scenario())


def test_docker_background_start_check_stop(tmp_path):
    async def scenario() -> None:
        ex = DockerExecutor(mode="run")
        handle = await ex.start("echo bg; sleep 0.2", cwd=tmp_path, env=None)
        for _ in range(50):
            status = await ex.check(handle)
            if status.status == "finished":
                break
            await asyncio.sleep(0.1)
        assert status.status == "finished"
        assert "bg" in status.stdout
        await ex.stop(handle)

    _run(scenario())
