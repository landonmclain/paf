"""Security tests for the Docker sandbox backend.

These assert the *containment properties* of the sandbox, not just the plumbing:
the locked-down flags (``--network=none``, ``--read-only``, ``--user <host uid:gid>``, never root)
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

from paf.capabilities._sandbox import DockerExecutor, default_sandbox_user


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
            'python -c "import urllib.request; '
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
        res = await ex.run(
            "touch /etc/security_probe.txt", cwd=tmp_path, env=None, timeout=30.0
        )
        assert res.exit_code != 0, "sandbox unexpectedly wrote to a read-only path"

    _run(scenario())


@pytest.mark.live
def test_docker_workspace_writable(tmp_path):
    """The bind-mounted workspace is the one writable spot, and it maps to the host."""

    async def scenario() -> None:
        ex = DockerExecutor(mode="run")
        res = await ex.run(
            "touch /workspace/security_probe.txt", cwd=tmp_path, env=None, timeout=30.0
        )
        assert res.exit_code == 0
        assert (tmp_path / "security_probe.txt").exists(), (
            "workspace write did not land on the host bind mount"
        )

    _run(scenario())


def test_docker_non_root(tmp_path):
    """The sandbox runs as the host's (non-root) user, so the workspace is readable."""
    expected = default_sandbox_user().split(":")[0]
    assert expected != "0"

    async def scenario() -> None:
        ex = DockerExecutor(mode="run")
        res = await ex.run("id -u", cwd=tmp_path, env=None, timeout=30.0)
        assert res.exit_code == 0
        assert res.stdout.strip() == expected, (
            f"expected uid {expected}, got {res.stdout.strip()!r}"
        )

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


@pytest.mark.live
def test_docker_exclude_overlays_env(tmp_path):
    """An excluded .env is empty inside the container; the host file is untouched."""
    (tmp_path / ".env").write_text("SECRET=super-secret-value\n")

    async def scenario() -> None:
        ex = DockerExecutor(mode="run", exclude=[".env"])
        res = await ex.run("cat /workspace/.env", cwd=tmp_path, env=None, timeout=30.0)
        assert res.exit_code == 0
        assert "super-secret-value" not in res.stdout, "secret leaked into the sandbox"
        assert res.stdout.strip() == "", (
            "excluded .env should be empty in the container"
        )

    _run(scenario())
    # The real host file is never touched by the overlay.
    assert (tmp_path / ".env").read_text() == "SECRET=super-secret-value\n"


@pytest.mark.live
def test_docker_exclude_write_hits_overlay(tmp_path):
    """A write to /workspace/.env lands on the overlay, not the host file."""
    (tmp_path / ".env").write_text("SECRET=original\n")

    async def scenario() -> None:
        ex = DockerExecutor(mode="run", exclude=[".env"])
        res = await ex.run(
            "echo 'SECRET=compromised' > /workspace/.env",
            cwd=tmp_path,
            env=None,
            timeout=30.0,
        )
        assert res.exit_code == 0

    _run(scenario())
    assert (tmp_path / ".env").read_text() == "SECRET=original\n", (
        "host .env was modified"
    )


@pytest.mark.live
def test_docker_exclude_overlays_git_dir(tmp_path):
    """A .git directory is shadowed by an empty dir inside the container."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    async def scenario() -> None:
        ex = DockerExecutor(mode="run", exclude=[".git"])
        res = await ex.run(
            "ls -A /workspace/.git", cwd=tmp_path, env=None, timeout=30.0
        )
        assert res.exit_code == 0
        assert res.stdout.strip() == "", (
            "excluded .git should be empty in the container"
        )

    _run(scenario())
    assert (tmp_path / ".git" / "HEAD").read_text() == "ref: refs/heads/main\n"
