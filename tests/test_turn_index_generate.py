"""Tests for the turn-index generator (collector merge + cursor + stub summarizer)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
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


def _question_messages(user_text: str = "what is my passport codename?") -> list[object]:
    """A realistic run's recovered messages: user ask -> tool call -> result -> answer.

    These are exactly what `SnapshotHistorySource.run_history` yields, and what
    `build_turn_context` renders into the summarizer's full-run context.
    """
    return [
        ModelRequest(parts=[UserPromptPart(content=user_text)]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="search_conversation_history",
                    args='{"query": "passport codename"}',
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="search_conversation_history",
                    content="found ZEPHYR-7741",
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content="Your passport codename is **ZEPHYR-7741**.")]),
    ]


async def _register_run(
    store: FileStepStore,
    run_id: str,
    conversation_id: str | None,
    *,
    tool: str | None = None,
    messages: list[object] | None = None,
) -> None:
    """Register a run + its step events + a real full-history snapshot.

    The summarizer reads `run_history` (recovered from snapshots), so tests must
    persist a snapshot carrying real messages -- bare `append_event` calls only
    feed the collector, not the LLM-turn context path.
    """
    await store.register_run(RunRecord(run_id=run_id, conversation_id=conversation_id, agent_name="assistant"))
    await store.append_event(_mk_event(run_id, "run_started", conversation_id=conversation_id))
    if tool:
        await store.append_event(
            _mk_event(run_id, "tool_call_started", tool_name=tool, conversation_id=conversation_id)
        )
        await store.append_event(
            _mk_event(run_id, "tool_call_completed", tool_name=tool, conversation_id=conversation_id)
        )
    await store.append_event(_mk_event(run_id, "run_completed", conversation_id=conversation_id))
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id=run_id,
            step_index=3,  # run_started, tool*2, run_completed
            messages=messages
            or (
                [
                    ModelRequest(parts=[UserPromptPart(content="hello")]),
                    ModelResponse(parts=[TextPart(content="hi")]),
                ]
            ),
            conversation_id=conversation_id,
            parent_run_id=None,
            agent_name="assistant",
            timestamp=datetime.now(timezone.utc),
            state="complete",
        )
    )


def test_generated_turns_merges_collector_and_writes_index(tmp_path: Path) -> None:
    store = FileStepStore(tmp_path / "store")

    async def scenario():
        await _register_run(
            store, "r1", "c1", tool="read_turn", messages=_question_messages()
        )

        collector = CollectorStore(tmp_path / "turn-index")
        cursor_store = CursorStore(tmp_path / "turn-index")
        summary = TurnIndexStore(tmp_path / "turn-index")

        first = await generated_turns(store, collector, cursor_store, summary)
        assert len(first) == 1
        assert first[0]["run_id"] == "r1"
        assert first[0]["message_index"] == 1  # one tool_call_completed
        summary_text = first[0]["summary"]
        # The one-liner is derived from the FULL run context (user prompt + tool + answer),
        # not just the trimmed collector: it captures the intent and the tool actually
        # called in the recovered messages.
        assert isinstance(summary_text, str)
        assert "search_conversation_history" in summary_text  # tool from the full-run context
        assert "passport codename" in summary_text  # from the full run, user prompt line

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
        await _register_run(
            store,
            "r2",
            "c2",
            messages=_question_messages(),
        )
        # Same (run_id, step_index, kind) twice -> dedup to one in the collector.
        await store.append_event(_mk_event("r2", "tool_call_started", step_index=3, tool_name="search"))
        await store.append_event(_mk_event("r2", "tool_call_started", step_index=3, tool_name="search"))

        collector = CollectorStore(tmp_path / "turn-index")
        summary = TurnIndexStore(tmp_path / "turn-index")
        await generated_turns(store, collector, CursorStore(tmp_path / "turn-index"), summary)
        events = collector.read()
        # The duplicate tool_call_started at the same key collapses to one.
        assert len(events) == len({e.key() for e in events})

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
            await _register_run(
                store, run_id, conv, tool="search", messages=_question_messages()
            )

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
        await _register_run(
            store, "rA", "cLive", messages=_question_messages()
        )

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
        await _register_run(
            store, "r0a", "cLive", tool="search", messages=_question_messages()
        )

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
        await _register_run(
            store,
            "r-live",
            "cli-primary",
            tool="search",
            messages=_question_messages(),
        )

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
        # The hook's summary came from the FULL run (seeded snapshot), not empty.
        assert "passport codename" in entries[0].summary
        # No stray entry in a different conversation.
        assert index.read_all("other") == []

        # Second run in the same conversation appends (index is append-only, not a global sweep).
        await _register_run(
            store,
            "r-live-2",
            "cli-primary",
            tool="read_turn",
            messages=_question_messages(),
        )
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