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
from paf.capabilities.turn_index._capability import TurnIndexCapability
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


def test_generated_turns_scopes_each_run_to_its_conversation(tmp_path: Path) -> None:
    """Regression: multi-conversation runs must not cross-contaminate the index.

    Previously ``CollectorStore.read()`` ignored its ``conversation_id`` argument and
    returned every conversation's events, so the summary loop wrote each run into every
    conversation's ``index.jsonl``. Each run must be summarized once, into its own
    conversation's directory only.
    """
    store = FileStepStore(tmp_path / "store")

    async def scenario():
        for run_id, conv in (("rA", "cA"), ("rB", "cB"), ("rC", "cC")):
            await store.register_run(RunRecord(run_id=run_id, conversation_id=conv, agent_name="assistant"))
            await store.append_event(_mk_event(run_id, "run_started", conversation_id=conv))
            await store.append_event(_mk_event(run_id, "tool_call_started", tool_name="search", conversation_id=conv))
            await store.append_event(_mk_event(run_id, "tool_call_completed", tool_name="search", conversation_id=conv))
            await store.append_event(_mk_event(run_id, "run_completed", conversation_id=conv))

        collector = CollectorStore(tmp_path / "turn-index")
        cursor_store = CursorStore(tmp_path / "turn-index")
        summary = TurnIndexStore(tmp_path / "turn-index")

        written = await generated_turns(store, collector, cursor_store, summary)
        assert len(written) == 3  # one summary per run, no duplicates

        # Each conversation's index holds exactly its own run -- and only once.
        for run_id, conv in (("rA", "cA"), ("rB", "cB"), ("rC", "cC")):
            entries = summary.read_all(conv)
            assert len(entries) == 1, f"{conv} index contaminated: {[e.run_id for e in entries]}"
            assert entries[0].run_id == run_id

        # Re-run stays idempotent and scoped.
        again = await generated_turns(store, collector, cursor_store, summary)
        assert again == []
        for conv in ("cA", "cB", "cC"):
            assert len(summary.read_all(conv)) == 1

    asyncio.run(scenario())


def test_cursor_tracks_covered_runs_prevents_resummarize(tmp_path: Path) -> None:
    """Regression: a single-watermark cursor re-summarizes OLDER runs when a newer run
    sorts first in the iteration (out-of-write-order), producing duplicate index lines.

    The cursor must track the SET of covered run_ids so each `after_run`/`generated_turns`
    is a true no-op for already-indexed turns -- not just the most-recent run.
    """
    store = FileStepStore(tmp_path / "store")

    async def scenario():
        # Run A registered + summarized first.
        await store.register_run(RunRecord(run_id="rA", conversation_id="cLive", agent_name="assistant"))
        await store.append_event(_mk_event("rA", "run_started", conversation_id="cLive"))
        await store.append_event(_mk_event("rA", "model_request_completed", conversation_id="cLive"))
        await store.append_event(_mk_event("rA", "run_completed", conversation_id="cLive"))

        collector = CollectorStore(tmp_path / "turn-index")
        cursor_store = CursorStore(tmp_path / "turn-index")
        summary = TurnIndexStore(tmp_path / "turn-index")

        first = await generated_turns(store, collector, cursor_store, summary)
        assert len(first) == 1
        assert first[0]["run_id"] == "rA"
        c = cursor_store.load("cLive")
        assert c.is_covered("rA")

        # Now run B arrives. rB sorts BEFORE rA alphabetically? No: "rA" < "rB",
        # so rA sorts first. Force the out-of-order case by adding a run whose id
        # sorts before rA, e.g. r0a, after rA was already covered.
        await store.register_run(RunRecord(run_id="r0a", conversation_id="cLive", agent_name="assistant"))
        await store.append_event(_mk_event("r0a", "run_started", conversation_id="cLive"))
        await store.append_event(_mk_event("r0a", "tool_call_completed", tool_name="search", conversation_id="cLive"))
        await store.append_event(_mk_event("r0a", "run_completed", conversation_id="cLive"))

        again = await generated_turns(store, collector, cursor_store, summary)
        # Only the NEW run is written; rA is NOT re-summarized despite sorting last.
        assert sorted(str(w["run_id"]) for w in again) == ["r0a"]
        entries = summary.read_all("cLive")
        assert len(entries) == 2
        assert {e.run_id for e in entries} == {"rA", "r0a"}

        # Rerun is fully idempotent -- neither run rewritten.
        third = await generated_turns(store, collector, cursor_store, summary)
        assert third == []
        assert len(summary.read_all("cLive")) == 2

    asyncio.run(scenario())


def test_collector_read_scopes_to_conversation(tmp_path: Path) -> None:
    collector = CollectorStore(tmp_path / "turn-index")
    collector.append(EventRecord.from_step_event(_mk_event("r1", "run_started", conversation_id="c1")))
    collector.append(EventRecord.from_step_event(_mk_event("r2", "run_started", conversation_id="c2")))

    c1 = collector.read("c1")
    c2 = collector.read("c2")
    assert [e.run_id for e in c1] == ["r1"]
    assert [e.run_id for e in c2] == ["r2"]
    assert len(collector.read()) == 2  # unscoped = all conversations


def test_after_run_hook_populates_conversation_scoped_index(tmp_path: Path) -> None:
    """The `after_run` hook pulls events straight from the store and keys them by the
    run's `conversation_id` (the "pull from store on our hook" slice).

    A `cli-primary` run must land under `cli-primary` in the turn index -- never swept
    from a global `list_runs(None)` or cross-contaminated into another conversation's
    directory. The hook leaves `result` untouched and is a no-op when `store` is None.
    """
    from types import SimpleNamespace

    from pydantic_ai_harness.conversation_search import SnapshotHistorySource

    store = FileStepStore(tmp_path / "store")
    index = TurnIndexStore(tmp_path / "turn-index")
    cap = TurnIndexCapability(
        source=SnapshotHistorySource(store),
        index_store=index,
        store=store,
    )

    async def scenario():
        # A run under the friendly conversation id (mirrors live `cli-primary`).
        await store.register_run(
            RunRecord(run_id="r-live", conversation_id="cli-primary", agent_name="assistant")
        )
        await store.append_event(_mk_event("r-live", "run_started", conversation_id="cli-primary"))
        await store.append_event(
            _mk_event("r-live", "tool_call_started", tool_name="search", conversation_id="cli-primary")
        )
        await store.append_event(
            _mk_event("r-live", "tool_call_completed", tool_name="search", conversation_id="cli-primary")
        )
        await store.append_event(_mk_event("r-live", "run_completed", conversation_id="cli-primary"))

        ctx = SimpleNamespace(conversation_id="cli-primary")
        result = object()
        out = await cap.after_run(ctx, result=result)

        # Hook trivially returns the result unchanged.
        assert out is result
        # Index entry landed under cli-primary (this conversation only).
        entries = index.read_all("cli-primary")
        assert len(entries) == 1
        assert entries[0].run_id == "r-live"
        assert entries[0].conversation_id == "cli-primary"
        # No stray entry in a different conversation.
        assert index.read_all("other") == []

        # Second run in the same conversation appends (index is append-only, not a global sweep).
        await store.register_run(
            RunRecord(run_id="r-live-2", conversation_id="cli-primary", agent_name="assistant")
        )
        await store.append_event(_mk_event("r-live-2", "run_started", conversation_id="cli-primary"))
        await store.append_event(
            _mk_event("r-live-2", "tool_call_completed", tool_name="read_turn", conversation_id="cli-primary")
        )
        await store.append_event(_mk_event("r-live-2", "run_completed", conversation_id="cli-primary"))
        await cap.after_run(SimpleNamespace(conversation_id="cli-primary"), result=result)
        entries = index.read_all("cli-primary")
        assert len(entries) == 2
        assert {e.run_id for e in entries} == {"r-live", "r-live-2"}

    asyncio.run(scenario())


def test_after_run_hook_noop_without_store(tmp_path: Path) -> None:
    """A `from_spec`-style capability with no writer store is a no-op on `after_run`."""
    from types import SimpleNamespace

    from pydantic_ai_harness.conversation_search import SnapshotHistorySource

    index = TurnIndexStore(tmp_path / "turn-index")
    cap = TurnIndexCapability(
        source=SnapshotHistorySource(FileStepStore(tmp_path / "store")),
        index_store=index,
        store=None,
    )

    async def scenario():
        result = object()
        out = await cap.after_run(SimpleNamespace(conversation_id="cli-primary"), result=result)
        assert out is result
        assert index.read_all() == []

    asyncio.run(scenario())