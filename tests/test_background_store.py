"""Tests for the background-process event store (``background_processes/_store.py``).

Covers the dumb append-only contract: events are stored in append order and
never rewritten, the store does not interpret records (any status string is
accepted), per-agent scoping, and restoration — a fresh store over the same
root reads back the same events, which is the "survives restart" property the
durable registry needs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from paf.background_processes._store import RunEventStore


def _store(tmp_path: Path, agent: str = "demo_assistant") -> RunEventStore:
    return RunEventStore.open(str(tmp_path / "processes"), agent)


def _event(status: str = "running", **extra) -> dict:
    return {"run_id": "r1", "process_id": "consolidation", "status": status, **extra}


def test_append_stores_event_and_read_all_returns_it(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.append(_event(status="running", detail="triggered"))

    records = s.read_all()
    assert len(records) == 1
    assert records[0]["status"] == "running"
    assert records[0]["detail"] == "triggered"


def test_read_all_returns_append_order(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.append(_event(status="running"))
    s.append(_event(status="done"))
    s.append(_event(status="skipped"))

    assert [r["status"] for r in s.read_all()] == ["running", "done", "skipped"]


def test_append_only_never_rewrites_prior_events(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.append(_event(status="running"))
    s.append(_event(status="done"))
    before = s.read_all()

    # A later event must not mutate the earlier ones.
    s.append(_event(status="running", run_id="r2"))
    s.append(_event(status="error", run_id="r2", detail="boom"))

    after = s.read_all()
    assert after[:2] == before
    assert len(after) == 4


def test_store_does_not_interpret_status(tmp_path: Path) -> None:
    """The store is dumb: any status string is accepted, no enum gating."""
    s = _store(tmp_path)
    for status in ("running", "done", "error", "skipped", "queued", "anything"):
        s.append(_event(status=status))

    assert [r["status"] for r in s.read_all()] == [
        "running",
        "done",
        "error",
        "skipped",
        "queued",
        "anything",
    ]


def test_restoration_reads_back_same_events(tmp_path: Path) -> None:
    root = tmp_path / "processes"
    s1 = RunEventStore.open(str(root), "demo_assistant")
    s1.append(_event(status="running", detail="triggered"))
    s1.append(_event(status="done"))

    # A fresh store over the same root is the "after restart" view.
    s2 = RunEventStore.open(str(root), "demo_assistant")
    records = s2.read_all()
    assert len(records) == 2
    assert records[0]["status"] == "running"
    assert records[1]["status"] == "done"


def test_agents_are_scoped_apart(tmp_path: Path) -> None:
    root = tmp_path / "processes"
    RunEventStore.open(str(root), "agent_a").append(_event())
    RunEventStore.open(str(root), "agent_b").append(_event())

    assert len(RunEventStore.open(str(root), "agent_a").read_all()) == 1
    assert len(RunEventStore.open(str(root), "agent_b").read_all()) == 1


def test_empty_store_reads_empty(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.read_all() == []


def test_arbitrary_event_fields_are_preserved(tmp_path: Path) -> None:
    """The store does not care what is in the event — it stores it verbatim."""
    s = _store(tmp_path)
    event = {
        "run_id": "r1",
        "process_id": "consolidation",
        "status": "done",
        "detail": "3 records folded",
        "agent": "demo_assistant",
        "conversation_id": "p:c1",
        "timestamp": "2026-09-17T00:00:00+00:00",
        "extra_field": {"nested": [1, 2, 3]},
    }
    s.append(event)
    assert s.read_all() == [event]


def test_coordinator_publishes_each_transition_to_journal(tmp_path: Path) -> None:
    """Step 2 wiring: every ``_publish_task`` call appends one event to the journal.

    The coordinator is the writer; the store stays dumb. Each status transition
    (queued -> running -> done) lands as its own lossless record.
    """
    from paf.background import (
        DONE,
        PROCESSING,
        QUEUED,
        BackgroundCoordinator,
        BackgroundTask,
    )

    class _StubStore:
        def list_runs(self, conversation_id=None):
            return []

    class _Profile:
        capability_names = ("Memory",)

    coord = BackgroundCoordinator(
        _StubStore(),
        agent_name="assistant",
        profile=_Profile(),
        processes_root=str(tmp_path / "processes"),
    )

    task = BackgroundTask(
        id="consolidation.conv-1.1", label="Consolidating memory (conv-1)"
    )
    asyncio.run(coord._publish_task(task))  # queued
    task.status = PROCESSING
    asyncio.run(coord._publish_task(task))  # running
    task.status = DONE
    task.detail = "3 records folded"
    asyncio.run(coord._publish_task(task))  # done

    events = coord._run_events.read_all()
    assert [e["status"] for e in events] == [QUEUED, PROCESSING, DONE]
    assert events[0]["process_id"] == "consolidation"
    assert events[0]["conversation_id"] == "conv-1"
    assert events[0]["agent"] == "assistant"
    assert events[0]["run_id"] == "consolidation.conv-1.1"
    assert events[2]["detail"] == "3 records folded"


def test_conversation_scoped_store_roots_at_hashed_dir(tmp_path: Path) -> None:
    """A conversation-scoped store roots at
    ``<root>/<agent>/conversations/<hash>/runs.jsonl``, isolated from the agent
    journal and from other conversations.
    """
    import hashlib

    root = tmp_path / "processes"
    agent = "demo_assistant"
    cid = "p:c1"

    s = RunEventStore.open(str(root), agent, conversation_id=cid)
    s.append(_event(status="running"))

    # The journal lives under the hashed per-conversation dir.
    expected = (
        root
        / agent
        / "conversations"
        / hashlib.sha256(cid.encode("utf-8")).hexdigest()[:24]
        / "runs.jsonl"
    )
    assert expected.exists()
    assert len(s.read_all()) == 1

    # The agent journal is untouched.
    assert not (root / agent / "runs.jsonl").exists()

    # A different conversation is isolated.
    other = RunEventStore.open(str(root), agent, conversation_id="p:c2")
    assert other.read_all() == []


def test_safe_key_segment_passes_through_nested_safe_path() -> None:
    """Nested safe keys (``<agent>/conversations/<hash>``) map to a real nested
    directory, not a flattened+hashed single segment.
    """
    from paf.capabilities._storage._flat_file import _safe_key_segment

    key = "demo_assistant/conversations/abc123"
    assert _safe_key_segment(key) == key

    # Bare names still pass through; deep/absolute keys still hash.
    assert _safe_key_segment("demo_assistant") == "demo_assistant"
    assert _safe_key_segment("/deep/absolute/path") != "/deep/absolute/path"


def test_recent_can_narrow_to_a_status(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.append(_event(status="done", detail="5 records folded", conversation_id="c"))
    s.append(_event(status="skipped", detail="only 1", conversation_id="c"))

    assert s.recent("consolidation", "c")["status"] == "skipped"
    assert s.recent("consolidation", "c", status="done")["detail"] == "5 records folded"


def _memory_coordinator(tmp_path: Path):
    from paf.background import BackgroundCoordinator

    class _StubStore:
        def list_runs(self, conversation_id=None):
            return []

        def list_snapshots(self, run_id=None):
            return []

    class _Profile:
        capability_names = ("Memory",)

    return BackgroundCoordinator(
        _StubStore(),
        agent_name="assistant",
        profile=_Profile(),
        processes_root=str(tmp_path / "processes"),
        memory_directory=str(tmp_path / "memory"),
    )


def test_fold_finish_reports_review_summary_and_changes(tmp_path: Path) -> None:
    """The consolidation row finishes from the job's outcome: fold count, the
    reviewer's tally, and one line per change."""
    from paf.background import DONE, PROCESSING, BackgroundTask
    from paf.capabilities.memory._background import FoldOutcome
    from paf.capabilities.memory._journal import JournalStore
    from paf.capabilities.memory._pairwise import Action, ReviewReport

    coord = _memory_coordinator(tmp_path)
    journal = JournalStore(tmp_path / "memory")
    for fact in ("The user lives in Tulsa.", "The user lives in Denver."):
        journal.append(
            type="identity",
            fact=fact,
            tier="t1",
            born_on="2026-09-30T12:00:00+00:00",
            source_ref="r:1",
            scope="global",
            source="stated",
        )
    task = BackgroundTask(id="consolidation.assistant.1", label="Consolidating memory")
    task.status = PROCESSING
    coord._background_tasks[task.id] = task
    queue = coord._ensure_memory_queue()

    review = ReviewReport(
        reviewed=[2],
        actions=[Action("supersede", 1, 2, "update", "superseded by #2: moved")],
    )
    asyncio.run(queue.on_finished(FoldOutcome(records=2, review=review)))

    assert task.status == DONE
    assert task.detail == "2 records folded · reviewed 1: 1 superseded"
    assert task.lines == [
        'superseded "The user lives in Tulsa." by "The user lives in Denver." — moved'
    ]
    assert coord._run_events.read_all()[-1]["lines"] == task.lines


def test_idle_row_keeps_last_completed_run(tmp_path: Path) -> None:
    """After the hold window, and under a newer skip, the idle row still says
    what the last fold did and what it changed."""
    from paf.background import JobIdleStatus

    coord = _memory_coordinator(tmp_path)
    old = "2026-01-01T00:00:00+00:00"
    base = {"run_id": "consolidation.assistant.1", "process_id": "consolidation"}
    coord._run_events.append(
        {
            **base,
            "status": "done",
            "detail": "5 records folded · reviewed 5: 1 archived",
            "lines": ['archived "x" (same as "y") — dup'],
            "conversation_id": "assistant",
            "timestamp": old,
        }
    )
    row = JobIdleStatus(
        id="consolidation.c", label="Consolidating memory", detail="idle · 1/10"
    )

    idle = coord._reconcile_status(row, "consolidation", "assistant")
    assert idle.status == "idle" and idle.detail == "idle · 1/10"
    assert idle.last == "5 records folded · reviewed 5: 1 archived"
    assert idle.last_at == old
    assert idle.lines == ('archived "x" (same as "y") — dup',)

    from datetime import UTC, datetime

    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    coord._run_events.append(
        {
            **base,
            "run_id": "consolidation.assistant.2",
            "status": "skipped",
            "detail": "only 1 unsettled turn(s)",
            "conversation_id": "assistant",
            "timestamp": now,
        }
    )
    skipped = coord._reconcile_status(row, "consolidation", "assistant")
    assert skipped.status == "skipped"
    assert skipped.last == "5 records folded · reviewed 5: 1 archived"
