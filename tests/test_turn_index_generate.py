"""Tests for the turn-index generator (collector merge + cursor + stub summarizer)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_ai_harness.step_persistence import (
    EventKind,
    FileStepStore,
    RunRecord,
    StepEvent,
)

from paf.capabilities.turn_index import TurnIndexStore
from paf.capabilities.turn_index._generate import (
    CollectorStore,
    CursorStore,
    EventRecord,
    generated_turns,
)


def _mk_event(
    run_id: str,
    kind: EventKind,
    *,
    step_index: int = 0,
    conversation_id: str | None = "c1",
    tool_name: str | None = None,
    error: str | None = None,
) -> StepEvent:
    return StepEvent(
        run_id=run_id,
        kind=kind,
        step_index=step_index,
        conversation_id=conversation_id,
        agent_name="assistant",
        tool_name=tool_name,
        error=error,
    )


def test_generated_turns_merges_collector_and_writes_index(tmp_path: Path) -> None:
    store = FileStepStore(tmp_path / "store")

    async def scenario():
        await store.register_run(
            RunRecord(run_id="r1", conversation_id="c1", agent_name="assistant")
        )
        await store.append_event(_mk_event("r1", "run_started"))
        await store.append_event(_mk_event("r1", "tool_call_started", tool_name="read_turn"))
        await store.append_event(_mk_event("r1", "tool_call_completed", tool_name="read_turn"))
        await store.append_event(_mk_event("r1", "run_completed"))

        collector = CollectorStore(tmp_path / "turn-index")
        cursor_store = CursorStore(tmp_path / "turn-index")
        summary = TurnIndexStore(tmp_path / "turn-index")

        first = await generated_turns(store, collector, cursor_store, summary)
        assert len(first) == 1
        assert first[0]["run_id"] == "r1"
        assert first[0]["message_index"] == 1  # one tool_call_completed
        summary_text = first[0]["summary"]
        assert isinstance(summary_text, str) and "read_turn" in summary_text

        # Collector has all 4 events, deduped.
        events = collector.read()
        assert len(events) == 4
        assert len({e.key() for e in events}) == 4

        # Index has one entry.
        entries = summary.read_all()
        assert len(entries) == 1
        assert entries[0].run_id == "r1"

        # Cursor advanced.
        c = cursor_store.load("c1")
        assert c.run_id == "r1"

        # Re-run is idempotent: same events already merged, index already has the turn.
        second = await generated_turns(store, collector, cursor_store, summary)
        assert second == []
        assert len(collector.read()) == 4
        assert len(summary.read_all()) == 1

    asyncio.run(scenario())


def test_generated_turns_dedups_repeated_events(tmp_path: Path) -> None:
    store = FileStepStore(tmp_path / "store")

    async def scenario():
        await store.register_run(
            RunRecord(run_id="r2", conversation_id="c2", agent_name="assistant")
        )
        # Same (run_id, step_index, kind) twice -> dedup to one.
        await store.append_event(_mk_event("r2", "tool_call_started", step_index=3, tool_name="search"))
        await store.append_event(_mk_event("r2", "tool_call_started", step_index=3, tool_name="search"))

        collector = CollectorStore(tmp_path / "turn-index")
        summary = TurnIndexStore(tmp_path / "turn-index")
        await generated_turns(store, collector, CursorStore(tmp_path / "turn-index"), summary)
        events = collector.read()
        assert len(events) == 1

    asyncio.run(scenario())


def test_event_record_roundtrip() -> None:
    e = _mk_event("r9", "tool_call_failed", tool_name="search", error="boom")
    rec = EventRecord.from_step_event(e)
    assert rec.key() == ("r9", 0, "tool_call_failed")
    d = rec.to_dict()
    assert d["tool_name"] == "search"
    assert d["error"] == "boom"
    restored = EventRecord(**d)  # type: ignore[arg-type]  # dict[str,object] is a lossy json view
    assert restored == rec