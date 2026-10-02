"""Shell background-command durability + queue wiring tests.

Covers the Phase 6 shell changes:
- ``ShellCommandStore`` (``capabilities/shell/_store.py``) round-trips on the
  shared append-only seam, scoped by agent in the same ``data/processes`` root.
- Live events: ``start_command`` emits a ``ProcessEvent`` (running) through the
  per-agent ``ProcessEventEmitter`` (the feed path) and writes a RICH row
  (pid/status) to ``ShellCommandStore``, keyed on the same ``run_id``.
- ``start_command`` fires IMMEDIATELY through ``ShellBackgroundQueue`` (no
  turn-end / coordinator wait) and returns a run id.
- Detached-process re-attach: after the toolset's live cache is gone, a fresh
  toolset over the same root can still ``check_command`` by id (durable pid).
- Backward compat: with no ``processes_root``, the legacy in-memory executor
  path (executor.start + command id) still works.

The repo runs the default suite with ``-m 'not live'`` and uses sync
``def test_*`` driving async via ``asyncio.run`` (see ``tests/test_background.py``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from paf.background_processes._events import (
    ProcessEvent,
    ProcessEventType,
    clear_agent_emitters,
    get_agent_emitter,
)
from paf.capabilities.shell._store import ShellCommandStore
from paf.capabilities.shell._toolset import ShellToolset


@pytest.fixture(autouse=True)
def _isolate_emitters() -> None:
    """Clear the process-wide per-agent emitters so captured events are isolated."""
    clear_agent_emitters()
    yield
    clear_agent_emitters()


def _toolset(
    root: str | Path,
    *,
    agent: str = "demo_assistant",
    conversation_id: str | None = None,
) -> ShellToolset:
    return ShellToolset(
        cwd=Path("."),
        allowed_commands=[],
        denied_commands=[],
        denied_operators=[],
        default_timeout=30.0,
        max_output_chars=50_000,
        persist_cwd=False,
        allow_interactive=False,
        processes_root=Path(root) / "processes",
        agent=agent,
        conversation_id=conversation_id,
    )


def _processes(tmp_path: Path) -> Path:
    return tmp_path / "processes"


def test_shell_store_append_read_and_recent(tmp_path: Path) -> None:
    store = ShellCommandStore.open(str(_processes(tmp_path)), "demo_assistant")
    store.append({"run_id": "r1", "command": "true", "pid": 123, "status": "running"})
    store.append({"run_id": "r2", "command": "false", "status": "error"})

    assert len(store.read_all()) == 2
    assert store.recent("r1")["pid"] == 123
    assert store.recent("r2")["command"] == "false"
    assert store.recent("nope") is None


def test_shell_store_restores_after_reopen(tmp_path: Path) -> None:
    root = tmp_path / "processes"
    ShellCommandStore.open(str(root), "demo_assistant").append(
        {"run_id": "r1", "command": "ls", "status": "done", "exit_code": 0}
    )
    reopened = ShellCommandStore.open(str(root), "demo_assistant")
    assert reopened.read_all()[0]["exit_code"] == 0


def test_shell_store_conversation_scoped_root(tmp_path: Path) -> None:
    """ShellCommandStore.open with conversation_id roots at the hashed
    per-conversation folder, colocated with the basic runs.jsonl."""
    import hashlib

    root = tmp_path / "processes"
    agent = "demo_assistant"
    store = ShellCommandStore.open(str(root), agent, conversation_id="c1")
    store.append(
        {"run_id": "shell.c1.1", "command": "ls", "status": "running", "pid": 1}
    )

    # Rich store lands in the hashed conversation folder.
    conv_path = (
        root
        / agent
        / "conversations"
        / hashlib.sha256(b"c1").hexdigest()[:24]
        / "shell_cmds.jsonl"
    )
    assert conv_path.exists()
    assert store.read_all()[0]["run_id"] == "shell.c1.1"

    # The agent-level store (no conversation_id) is a separate file.
    agent_store = ShellCommandStore.open(str(root), agent)
    assert agent_store.read_all() == []


def test_start_command_emits_running_and_writes_rich_store(tmp_path: Path) -> None:
    async def scenario():
        events: list[ProcessEvent] = []
        get_agent_emitter("demo_assistant").on(ProcessEventType.running, events.append)
        async with _toolset(tmp_path) as t:
            out = await t.start_command("echo hello-background")
            run_id = out.split("ID: ")[1]
            # A live "running" event was emitted (the feed path), not a runs.jsonl row.
            assert any(e.id == run_id and e.status == "running" for e in events)
            # The rich record is written by the async worker after it spawns the
            # detached subprocess — wait for it (with pid) rather than asserting
            # synchronously right after start_command.
            shell = ShellCommandStore.open(str(_processes(tmp_path)), "demo_assistant")
            rich = None
            for _ in range(40):
                rich = shell.recent(run_id)
                if rich is not None and rich.get("pid") is not None:
                    break
                await asyncio.sleep(0.05)
            assert rich is not None and rich["command"] == "echo hello-background"
            assert rich.get("pid") is not None, "worker should record pid"
            return run_id

    run_id = asyncio.run(scenario())
    assert run_id.startswith("shell.global.")


def test_run_id_prepends_shell_and_conversation_and_stamps_conversation_id(
    tmp_path: Path,
) -> None:
    """run_id is shell.<conversation_id>.<seq>; conversation_id lands in the event scope + rich store."""

    async def scenario():
        events: list[ProcessEvent] = []
        get_agent_emitter("demo_assistant").on(ProcessEventType.running, events.append)
        async with _toolset(tmp_path, conversation_id="c1") as t:
            out = await t.start_command("echo convo")
            run_id = out.split("ID: ")[1]
            assert run_id == "shell.c1.1", run_id
            # The emitted running event carries the conversation scope + lane.
            ev = next(e for e in events if e.id == run_id)
            assert ev.scope == "c1"
            assert ev.lane == "conversation"
            # rich row (written by worker) also carries it
            sc = ShellCommandStore.open(
                str(_processes(tmp_path)), "demo_assistant", conversation_id="c1"
            )
            for _ in range(40):
                if sc.recent(run_id) is not None:
                    break
                await asyncio.sleep(0.05)
            assert sc.recent(run_id)["conversation_id"] == "c1"
            # second start in the same conversation advances seq
            out2 = await t.start_command("echo convo2")
            assert out2.split("ID: ")[1] == "shell.c1.2"
            return run_id

    assert asyncio.run(scenario()) == "shell.c1.1"


def test_check_command_reattaches_by_pid_after_cache_gone(tmp_path: Path) -> None:
    """A fresh toolset (no live handle) can still check a started command."""

    async def scenario():
        async with _toolset(tmp_path) as t1:
            out = await t1.start_command("sh -c 'sleep 5'")
            run_id = out.split("ID: ")[1]
            # Wait for the worker to record the pid in the durable store.
            shell = ShellCommandStore.open(str(_processes(tmp_path)), "demo_assistant")
            for _ in range(40):
                rec = shell.recent(run_id)
                if rec is not None and rec.get("pid") is not None:
                    break
                await asyncio.sleep(0.05)
            assert rec is not None and rec.get("pid") is not None
            # Simulate a restart: a brand-new toolset, empty live cache.
            async with _toolset(tmp_path) as t2:
                check = await t2.check_command(run_id)
                assert "[status: running]" in check
                # Clean up the detached process.
                await t2.stop_command(run_id)
            return run_id

    run_id = asyncio.run(scenario())
    assert run_id


def test_check_resolves_stale_running_when_process_gone(tmp_path: Path) -> None:
    """A journal line stuck on `running` (supervisor died) resolves to finished.

    Simulates a process that finished but whose supervisor never wrote the
    terminal line: the durable store has a `running` line with a NOW-DEAD pid.
    check_command should derive finished from pid-gone and APPEND a terminal
    line (append-only), not keep reporting running.
    """

    async def scenario():
        # The toolset below opens conversation-scoped stores, so the hand-written
        # line must go into the same conversation-scoped store.
        sc = ShellCommandStore.open(
            str(_processes(tmp_path)), "demo_assistant", conversation_id="c1"
        )
        # Hand-write a `running` line with a dead pid (624242 unlikely to exist).
        sc.append(
            {
                "run_id": "shell.c1.9",
                "command": "sleep 1",
                "agent": "demo_assistant",
                "conversation_id": "c1",
                "timestamp": "2026-09-22T00:00:00+00:00",
                "status": "running",
                "pid": 624242,
            }
        )
        async with _toolset(tmp_path, conversation_id="c1") as t:
            check = await t.check_command("shell.c1.9")
            assert "[status: finished]" in check, check
            # The staleness was resolved by APPENDING a terminal line. A vanished
            # process with no exit code is reported as error, not done.
            lines = sc.all_for("shell.c1.9")
            assert lines[-1]["status"] == "error"
            assert len(lines) == 2  # running + resolved error
        return True

    assert asyncio.run(scenario()) is True


def test_legacy_in_memory_path_without_processes_root(tmp_path: Path) -> None:
    """With no processes_root, background uses executor.start + command id."""

    async def scenario():
        async with ShellToolset(
            cwd=Path("."),
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=30.0,
            max_output_chars=50_000,
            persist_cwd=False,
            allow_interactive=False,
        ) as t:
            out = await t.start_command("echo legacy")
            assert "background command" in out
            check = await t.check_command(out.split("ID: ")[1])
            assert "[status:" in check
            return out

    out = asyncio.run(scenario())
    assert "ID:" in out


def test_two_store_join_key_is_run_id(tmp_path: Path) -> None:
    """The same run_id keys a live emitted event and a rich shell row."""

    async def scenario():
        events: list[ProcessEvent] = []
        get_agent_emitter("demo_assistant").on(ProcessEventType.running, events.append)
        async with _toolset(tmp_path) as t:
            out = await t.start_command("echo join-check")
            run_id = out.split("ID: ")[1]
            # A live running event was emitted for this run_id (the feed path).
            assert any(e.id == run_id for e in events)
            # The rich row rejoins by the same run_id (the ShellCommandStore.recent
            # lookup *is* the join). The rich row is written by the worker, so wait
            # for it; prove it resolves to the command we started.
            rich_store = ShellCommandStore.open(
                str(_processes(tmp_path)), "demo_assistant"
            )
            for _ in range(40):
                if rich_store.recent(run_id) is not None:
                    break
                await asyncio.sleep(0.05)
            assert rich_store.recent(run_id)["command"] == "echo join-check"
            return True

    assert asyncio.run(scenario()) is True


def test_background_output_is_durable_paged_and_notified(tmp_path: Path) -> None:
    """Output lands in shell_outputs/<run_id>.log (not /tmp), the notify carries
    the tail with a pointer, and check_command pages the file."""
    from paf.capabilities._sandbox import LocalProcessExecutor
    from paf.capabilities.shell._background import ShellBackgroundQueue, ShellRun

    notes: list[tuple[str, bool, str]] = []

    async def on_complete(run_id, cid, ok, detail):
        notes.append((run_id, ok, detail))

    store = ShellCommandStore.open(
        str(_processes(tmp_path)), "demo_assistant", conversation_id="c1"
    )
    queue = ShellBackgroundQueue(
        executor=LocalProcessExecutor(),
        cwd=tmp_path,
        agent="demo_assistant",
        shell_cmds=store,
        on_complete=on_complete,
    )
    cmd = "seq 1 2000; echo boom >&2"
    run = ShellRun(
        run_id="shell.c1.1", command=cmd, conversation_id="c1",
        notify_on_completion=True,
    )  # fmt: skip
    asyncio.run(queue.process_job("s", run, None))

    path = store.output_path("shell.c1.1")
    assert path is not None and path.parent.name == "shell_outputs"
    text = path.read_text()
    assert text.startswith("1\n2\n") and text.rstrip().endswith("boom")
    last = store.recent("shell.c1.1")
    assert last["status"] == "done" and last["output_chars"] == len(text)

    (run_id, ok, msg) = notes[0]
    assert ok and msg.startswith("[system] Background command shell.c1.1 (exit 0)")
    assert "boom" in msg and "check_command('shell.c1.1', offset=0)" in msg
    assert "\n1\n2\n" not in msg  # the head is not in the excerpt

    async def check():
        async with _toolset(tmp_path, conversation_id="c1") as t:
            tail = await t.check_command("shell.c1.1", max_chars=100)
            head = await t.check_command("shell.c1.1", offset=0, max_chars=10)
            return tail, head

    tail, head = asyncio.run(check())
    assert "[status: finished]" in tail and "[exit code: 0]" in tail
    assert tail.rstrip().endswith("boom") and "offset=" in tail
    assert "1\n2\n3\n4\n5" in head and "more chars; check_command" in head
