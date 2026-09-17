"""Tests for the background-process event store (``background_processes/_store.py``).

Covers the dumb append-only contract: events are stored in append order and
never rewritten, the store does not interpret records (any status string is
accepted), per-agent scoping, and restoration — a fresh store over the same
root reads back the same events, which is the "survives restart" property the
durable registry needs.
"""

from __future__ import annotations

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
    coord._publish_task(task)  # queued
    task.status = PROCESSING
    coord._publish_task(task)  # running
    task.status = DONE
    task.detail = "3 records folded"
    coord._publish_task(task)  # done

    events = coord._run_events.read_all()
    assert [e["status"] for e in events] == [QUEUED, PROCESSING, DONE]
    assert events[0]["process_id"] == "consolidation"
    assert events[0]["conversation_id"] == "conv-1"
    assert events[0]["agent"] == "assistant"
    assert events[0]["run_id"] == "consolidation.conv-1.1"
    assert events[2]["detail"] == "3 records folded"
