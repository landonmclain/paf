"""Security tests for the Docker sandbox backend.

These assert the *containment properties* of the sandbox, not just the plumbing:
the locked-down flags (``--network=none``, ``--read-only``, ``--user 1000:1000``)
must actually hold at runtime. Each test drives a fresh ``mode="run"`` container
(the strongest isolation) and checks that a hostile action is contained.

Follows the repo convention: sync ``def test_*`` functions that drive async
scenarios via ``asyncio.run(...)``. Skipped when Docker is unavailable.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest

from paf.capabilities._sandbox import DockerExecutor


def _run(coro):
    return asyncio.run(coro)


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


def test_docker_network_dropped(tmp_path):
    """A network attempt inside the sandbox must fail (--network=none)."""

    async def scenario() -> None:
        ex = DockerExecutor(mode="run")
        res = await ex.run(
            "python -c \"import urllib.request; "
            "urllib.request.urlopen('https://example.com', timeout=5)\"",
            cwd=tmp_path,
            env=None,
            timeout=30.0,
        )
        assert res.exit_code != 0, "sandbox unexpectedly reached the network"

    _run(scenario())


def test_docker_root_read_only(tmp_path):
    """Writing outside the workspace must fail (--read-only)."""

    async def scenario() -> None:
        ex = DockerExecutor(mode="run")
        res = await ex.run("touch /etc/security_probe.txt", cwd=tmp_path, env=None, timeout=30.0)
        assert res.exit_code != 0, "sandbox unexpectedly wrote to a read-only path"

    _run(scenario())


def test_docker_workspace_writable(tmp_path):
    """The bind-mounted workspace is the one writable spot, and it maps to the host."""

    async def scenario() -> None:
        ex = DockerExecutor(mode="run")
        res = await ex.run("touch /workspace/security_probe.txt", cwd=tmp_path, env=None, timeout=30.0)
        assert res.exit_code == 0
        assert (tmp_path / "security_probe.txt").exists(), (
            "workspace write did not land on the host bind mount"
        )

    _run(scenario())


def test_docker_non_root(tmp_path):
    """The sandbox runs as the non-root user 1000:1000."""

    async def scenario() -> None:
        ex = DockerExecutor(mode="run")
        res = await ex.run("id -u", cwd=tmp_path, env=None, timeout=30.0)
        assert res.exit_code == 0
        assert res.stdout.strip() == "1000", f"expected uid 1000, got {res.stdout.strip()!r}"

    _run(scenario())


def test_docker_malicious_command_contained(tmp_path):
    """A destructive command is contained: it fails and the sandbox still works."""

    async def scenario() -> None:
        ex = DockerExecutor(mode="run")
        # rm -rf /etc is destructive on a host; on the read-only root it must fail.
        res = await ex.run("rm -rf /etc", cwd=tmp_path, env=None, timeout=30.0)
        assert res.exit_code != 0, "destructive command unexpectedly succeeded"

        # The sandbox is still functional afterward (proves it did not escape/break).
        res2 = await ex.run("echo still-alive", cwd=tmp_path, env=None, timeout=30.0)
        assert res2.exit_code == 0
        assert "still-alive" in res2.stdout

    _run(scenario())
