"""Container lifecycle: labels, reaping of orphans, close(), atexit cleanup.

Exec mode keeps one long-lived container per executor. Left alone, every process
start orphaned one (39 stopped husks had piled up). Now every container is
labeled, `close()`/`atexit` remove what a process made, and the first use in a
process reaps what a dead process left behind, without ever touching a container
whose owner is alive.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

import paf.capabilities._sandbox._docker as dk
from paf.capabilities._sandbox import DockerExecutor, reap_stale_containers
from paf.capabilities._sandbox._docker import docker_available, select_reapable


def _run(coro):
    return asyncio.run(coro)


# --- selection logic (no Docker) ---------------------------------------------------
def _alive_only(*pids: int):
    return lambda pid: pid in pids


@pytest.mark.parametrize("state", ["exited", "dead"])
def test_a_stopped_container_is_a_husk_only_when_its_owner_is_gone(state):
    """A finished background container still holds its logs: its owner reads them
    before calling ``stop()``. A sweep from a different process must not delete
    output that owner has not read yet."""
    rows = [
        ("husk", state, "exec", "7"),  # owner gone
        ("legacy", state, "", ""),  # pre-label husk: nobody owns it
        ("junk", state, "exec", "notanumber"),
        ("finished-bg", state, "bg", "42"),  # live owner: logs still pending
        ("live-exec", state, "exec", "42"),  # live owner removes it itself
    ]
    assert select_reapable(rows, _alive_only(42)) == ["husk", "legacy", "junk"]


def test_a_created_container_with_a_live_owner_is_mid_start_and_spared():
    """``docker run -d`` shows ``created`` until the container is running; another
    process's sweep in that window must not delete it out from under its owner."""
    rows = [
        ("starting", "created", "exec", "42"),  # live owner: about to start it
        ("crashed", "created", "exec", "7"),  # owner died between create and start
        ("ownerless", "created", "", ""),  # nobody will ever start it
    ]
    assert select_reapable(rows, _alive_only(42)) == ["crashed", "ownerless"]


def test_running_container_with_a_dead_owner_is_orphaned():
    rows = [("a", "running", "exec", "4242"), ("b", "running", "run", "4242")]
    assert select_reapable(rows, _alive_only()) == ["a", "b"]


def test_running_container_with_a_live_owner_is_never_touched():
    """Another process (the CLI next to paf-sse) may own it."""
    rows = [("a", "running", "exec", "7"), ("b", "running", "run", "7")]
    assert select_reapable(rows, _alive_only(7)) == []


def test_running_background_and_unreadable_owner_are_left_alone():
    rows = [
        ("bg", "running", "bg", "4242"),  # background jobs may outlive their starter
        ("nopid", "running", "exec", ""),
        ("junk", "running", "exec", "notanumber"),
        ("other", "running", "", ""),
    ]
    assert select_reapable(rows, _alive_only()) == []


def test_paused_and_removing_states_are_not_reaped():
    rows = [("a", "paused", "exec", "4242"), ("b", "removing", "exec", "4242")]
    assert select_reapable(rows, _alive_only()) == []


def test_pid_alive_helper():
    assert dk._pid_alive(os.getpid())
    done = subprocess.Popen(["true"])
    done.wait()
    assert not dk._pid_alive(done.pid)


def test_reaping_never_raises_when_docker_is_unavailable(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(dk.subprocess, "run", boom)
    assert reap_stale_containers() == []


def test_first_use_reaps_once_per_process(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dk, "reap_stale_containers", lambda *a, **k: calls.append(1) or []
    )
    monkeypatch.setattr(dk, "_reaped", False)

    async def scenario():
        await dk._reap_once()
        await dk._reap_once()
        await dk._reap_once()

    _run(scenario())
    assert calls == [1]


# --- Docker ---------------------------------------------------------------------------
needs_docker = pytest.mark.skipif(not docker_available(), reason="Docker not available")


def _ids(*filters: str) -> set[str]:
    out = subprocess.run(
        ["docker", "ps", "-a", "-q", *filters], capture_output=True, text=True
    ).stdout
    return set(out.split())


def _exists(name: str) -> bool:
    return bool(_ids("--filter", f"name=^{name}$"))


def _inspect(name: str) -> dict:
    out = subprocess.run(
        ["docker", "inspect", name], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out)[0]


def _mk(name: str, *, labels: dict[str, str], cmd: list[str]) -> None:
    args = ["docker", "run", "-d", "--name", name]
    for k, v in labels.items():
        args += ["--label", f"{k}={v}"]
    subprocess.run(
        [*args, "python:3.12-slim", *cmd], capture_output=True, check=True, timeout=60
    )


def _dead_pid() -> int:
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


@pytest.fixture
def docker_cleanup():
    names: list[str] = []
    yield names
    for name in names:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@needs_docker
def test_docker_every_container_kind_is_labeled_with_its_owner(
    tmp_path, docker_cleanup
):
    async def scenario() -> dict[str, dict]:
        ex = DockerExecutor(
            mode="exec", container_name=f"paf-test-{uuid.uuid4().hex[:6]}"
        )
        await ex.run("true", cwd=tmp_path, env=None, timeout=30)
        exec_name = ex._container_name  # noqa: SLF001
        docker_cleanup.append(exec_name)
        handle = await ex.start("sleep 30", cwd=tmp_path, env=None)
        bg_name = handle.state.container_id
        docker_cleanup.append(bg_name)
        seen = {
            "exec": _inspect(exec_name)["Config"]["Labels"],
            "bg": _inspect(bg_name)["Config"]["Labels"],
        }
        await ex.stop(handle)
        await ex.close()
        return seen

    seen = _run(scenario())
    for kind, labels in seen.items():
        assert labels["paf.sandbox"] == "1"
        assert labels["paf.sandbox.kind"] == kind
        assert labels["paf.sandbox.pid"] == str(os.getpid())


@needs_docker
def test_docker_exec_mode_reuses_one_container_then_close_removes_it(tmp_path):
    async def scenario() -> tuple[str, str, bool]:
        ex = DockerExecutor(
            mode="exec", container_name=f"paf-test-{uuid.uuid4().hex[:6]}"
        )
        await ex.run(
            "echo one > /tmp/x 2>/dev/null; true", cwd=tmp_path, env=None, timeout=30
        )
        first = ex._container_id  # noqa: SLF001
        await ex.run("true", cwd=tmp_path, env=None, timeout=30)
        second = ex._container_id  # noqa: SLF001
        name = ex._container_name  # noqa: SLF001
        assert _exists(name)
        await ex.close()
        gone = not _exists(name)
        await ex.close()  # idempotent
        return first, second, gone

    first, second, gone = _run(scenario())
    assert first == second  # reused within the process
    assert gone


@needs_docker
def test_docker_close_without_starting_anything_is_a_noop(tmp_path):
    _run(DockerExecutor(mode="exec").close())


@needs_docker
def test_docker_close_removes_the_exec_overlay_dir(tmp_path):
    (tmp_path / ".env").write_text("s")
    before = set(Path("/tmp").glob("paf-overlay-*"))

    async def scenario() -> set[Path]:
        ex = DockerExecutor(
            mode="exec",
            exclude=[".env"],
            container_name=f"paf-test-{uuid.uuid4().hex[:6]}",
        )
        await ex.run("true", cwd=tmp_path, env=None, timeout=30)
        during = set(Path("/tmp").glob("paf-overlay-*")) - before
        await ex.close()
        return during

    during = _run(scenario())
    assert len(during) == 1  # held while the container lives
    assert set(Path("/tmp").glob("paf-overlay-*")) - before == set()


@needs_docker
def test_docker_atexit_hook_removes_what_this_process_made(tmp_path):
    async def scenario() -> str:
        ex = DockerExecutor(
            mode="exec", container_name=f"paf-test-{uuid.uuid4().hex[:6]}"
        )
        await ex.run("true", cwd=tmp_path, env=None, timeout=30)
        return ex._container_name  # noqa: SLF001

    name = _run(scenario())
    assert _exists(name) and name in dk._owned  # noqa: SLF001
    dk._close_owned_sync()  # noqa: SLF001 - what atexit runs
    assert not _exists(name) and name not in dk._owned  # noqa: SLF001


@needs_docker
def test_docker_reap_removes_orphans_and_husks_but_spares_the_living(docker_cleanup):
    tag = uuid.uuid4().hex[:6]
    mine = str(os.getpid())
    dead = str(_dead_pid())
    cases = {
        # name: (labels, cmd, should_be_reaped)
        f"paf-test-husk-{tag}": (
            {"paf.sandbox": "1", "paf.sandbox.kind": "exec", "paf.sandbox.pid": dead},
            ["true"],
            True,
        ),
        f"paf-test-bgdone-{tag}": (
            {"paf.sandbox": "1", "paf.sandbox.kind": "bg", "paf.sandbox.pid": mine},
            ["true"],
            False,  # finished, but its owner has not read the logs yet
        ),
        f"paf-test-orphan-{tag}": (
            {"paf.sandbox": "1", "paf.sandbox.kind": "exec", "paf.sandbox.pid": dead},
            ["sleep", "300"],
            True,
        ),
        f"paf-test-orphanrun-{tag}": (
            {"paf.sandbox": "1", "paf.sandbox.kind": "run", "paf.sandbox.pid": dead},
            ["sleep", "300"],
            True,
        ),
        f"paf-test-alive-{tag}": (
            {"paf.sandbox": "1", "paf.sandbox.kind": "exec", "paf.sandbox.pid": mine},
            ["sleep", "300"],
            False,
        ),
        f"paf-test-bg-{tag}": (
            {"paf.sandbox": "1", "paf.sandbox.kind": "bg", "paf.sandbox.pid": dead},
            ["sleep", "300"],
            False,
        ),
        f"paf-test-nopid-{tag}": (
            {"paf.sandbox": "1", "paf.sandbox.kind": "exec"},
            ["sleep", "300"],
            False,
        ),
        f"unrelated-{tag}": ({}, ["true"], False),  # exited, but not ours
        f"paf-sandbox-legacy{tag}": ({}, ["true"], True),  # pre-label husk
    }
    for name, (labels, cmd, _) in cases.items():
        docker_cleanup.append(name)
        _mk(name, labels=labels, cmd=cmd)
    subprocess.run(
        [
            "docker",
            "wait",
            f"paf-test-husk-{tag}",
            f"paf-test-bgdone-{tag}",
            f"unrelated-{tag}",
            f"paf-sandbox-legacy{tag}",
        ],
        capture_output=True,
        timeout=60,
    )

    reap_stale_containers()

    # Judge by what is left, not by what this call returned: under pytest-xdist,
    # another worker's first-use sweep may legitimately reap these orphans first.
    for name, (_, _, reaped) in cases.items():
        assert _exists(name) is (not reaped), f"{name}: reaped={reaped}"


@needs_docker
def test_docker_another_process_cannot_steal_a_finished_backgrounds_logs(
    tmp_path, docker_cleanup
):
    """The race the owner check closes: a background job's logs are read *after*
    it exits, so a sweep from a second process must not delete it first."""

    async def scenario() -> tuple[str, str]:
        ex = DockerExecutor(
            mode="exec", container_name=f"paf-test-{uuid.uuid4().hex[:6]}"
        )
        docker_cleanup.append(ex._container_name)  # noqa: SLF001
        handle = await ex.start("echo bg-output", cwd=tmp_path, env=None)
        name = handle.state.container_id
        docker_cleanup.append(name)

        # A second process sweeps while this one still owns the finished job.
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from paf.capabilities._sandbox import reap_stale_containers; "
                "reap_stale_containers()",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        status = await ex.check(handle)
        await ex.close()
        return status.stdout, name

    stdout, name = _run(scenario())
    assert "bg-output" in stdout, "the sweep deleted the container before it was read"
    assert _exists(name)  # still there: its owner is alive
