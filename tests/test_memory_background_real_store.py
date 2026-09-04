"""Real-store integration tests for the background memory-consolidation worker.

These exercise the three load-bearing behaviors the synthetic-seam unit tests
(:mod:`test_memory_background_trigger`) deliberately do NOT:
  1. **Persistence pull** — the worker resolves the unseen persisted span from a
     real ``FileStepStore`` via ``SnapshotHistorySource`` (list_runs ascending +
     run_history), honoring the ref watermark.
  2. **Watermark off it** — ``{run_id, turn_id}`` advances against the naturally
     sorted store: skip already-consumed turns within a run, include later runs.
  3. **Background queue trigger + fold** — a real store drives the worker's
     fold, advances the watermark, and renders, off the critical path.

The store fixtures mirror ``tests/test_search.py::_mk_store`` (register_run +
save_snapshot) and ``tests/test_runtime.py`` seeding.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    RunRecord,
)

from paf.capabilities.memory._background import (
    ConsolidationJob,
    MemoryConsolidationQueue,
    ResolvedSpan,
    resolve_unseen_span,
)
from paf.capabilities.memory._journal import JournalStore
from paf.capabilities.memory._watermark import MemoryWatermark


def _msg(*, user: str, assistant: str) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content=user)]),
        ModelResponse(parts=[TextPart(assistant)]),
    ]


def _store(tmp_path: Path, runs: dict[str, list[ModelMessage]], conversation_id: str = "c1") -> FileStepStore:
    """Write a FileStepStore with one run per key, in the given dict order."""
    store = FileStepStore(directory=tmp_path)
    for i, (run_id, msgs) in enumerate(runs.items()):
        asyncio.run(
            store.register_run(
                RunRecord(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    started_at=datetime(2026, 1, 1 + i, tzinfo=UTC),
                )
            )
        )
        asyncio.run(
            store.save_snapshot(
                ContinuableSnapshot(
                    run_id=run_id,
                    step_index=0,
                    conversation_id=conversation_id,
                    messages=msgs,
                )
            )
        )
    return store


def _turns(messages: list[ModelMessage]) -> list[str]:
    """User-prompt texts in the message list, in order (the turn anchors)."""
    out = []
    for m in messages:
        for p in m.parts:
            if isinstance(p, UserPromptPart):
                content = p.content
                text = content if isinstance(content, str) else str(content)
                out.append(text)
    return out


# ---------------------------------------------------------------------------
# 1. Persistence pull — resolve_unseen_span over a real store
# ---------------------------------------------------------------------------


def test_resolve_unseen_span_fresh_watermark_returns_all_complete_turns(tmp_path: Path) -> None:
    """Fresh watermark (nothing consumed) -> every complete turn, in order.

    ``third`` is the in-flight tip (tip=run-b:1), so it is excluded — the fold
    stops on the second-to-last turn to guarantee only settled turns fold.
    """
    store = _store(
        tmp_path,
        {
            "run-a": [*_msg(user="first", assistant="a1")],
            "run-b": [*_msg(user="second", assistant="b1"), *_msg(user="third", assistant="b2")],
        },
    )
    source = SnapshotHistorySource(store)
    wm = MemoryWatermark()  # fresh

    async def go():
        span = await resolve_unseen_span(source, wm, tip_run="run-b", tip_turn=1)
        # "third" is the tip -> excluded; only complete turns fold.
        assert _turns(span.messages) == ["first", "second"]
        # watermark advances to the last COMPLETE turn folded.
        assert span.watermark.run_id == "run-b"
        assert span.watermark.turn_id == 0

    asyncio.run(go())


def test_resolve_unseen_span_skips_consumed_tail_past_watermark(tmp_path: Path) -> None:
    """Watermark consumed run-a turn 0 -> run-a turn 1 + run-b complete turns."""
    store = _store(
        tmp_path,
        {
            "run-a": [_msg(user="consumed", assistant="a1")[0], _msg(user="mid", assistant="a2")[0]],
            "run-b": [_msg(user="new-1", assistant="b1")[0], _msg(user="new-2", assistant="b2")[0]],
        },
    )
    source = SnapshotHistorySource(store)
    wm = MemoryWatermark(run_id="run-a", turn_id=0)

    async def go():
        span = await resolve_unseen_span(source, wm, tip_run="run-b", tip_turn=1)
        # consumed(run-a:0) skipped; mid(run-a:1) + new-1(run-b:0) fold;
        # new-2(run-b:1) is the tip -> excluded.
        assert _turns(span.messages) == ["mid", "new-1"]
        assert span.watermark.run_id == "run-b"
        assert span.watermark.turn_id == 0

    asyncio.run(go())


def test_resolve_unseen_span_excludes_in_flight_tip(tmp_path: Path) -> None:
    """The in-flight tip turn is never folded (stop on the second-to-last turn)."""
    store = _store(
        tmp_path,
        {
            "run-a": [*_msg(user="only", assistant="a")],
        },
    )
    source = SnapshotHistorySource(store)
    wm = MemoryWatermark()  # fresh

    async def go():
        # tip is run-a:0 — the ONLY turn, and it's in flight -> fold nothing.
        span = await resolve_unseen_span(source, wm, tip_run="run-a", tip_turn=0)
        assert span.messages == []
        # fresh watermark is untouched (nothing folded).
        assert span.watermark.run_id == ""
        assert span.watermark.turn_id is None

    asyncio.run(go())


# ---------------------------------------------------------------------------
# 3. Session hook — combustion of trigger + real-store worker (live wiring)
# ---------------------------------------------------------------------------


def test_session_hook_enqueues_job_past_cadence(tmp_path: Path) -> None:
    """The Session turn hook fires the trigger and folds against a real store.

    Drives the LIVE wiring end to end: ``_maybe_trigger_consolidation``
    computes the tip, ``should_fire`` approves, the queue (constructed lazily,
    bound to ``self._store``) receives a job, and its self-contained worker
    resolves the persisted span and folds it into the journal — all without a
    live LLM (journal-fenced consolidate seam).
    """
    msgs: list[ModelMessage] = []
    for i in range(12):
        msgs.extend(_msg(user=f"t{i}", assistant=f"a{i}"))
    store = _store(tmp_path, {"run-a": msgs})
    memdir = tmp_path / "mem"
    memdir.mkdir(exist_ok=True)
    JournalStore(str(memdir))
    # Watermark at turn 1; 12 turns -> tip=11, 11-1=10 >= 10 cadence -> fires.
    MemoryWatermark(run_id="run-a", turn_id=1).save(str(memdir))

    async def fence(messages: list[ModelMessage], directory: str, wm: MemoryWatermark) -> int:
        j = JournalStore(directory)
        for m in messages:
            for p in m.parts:
                if isinstance(p, UserPromptPart):
                    j.append(
                        type="fact",
                        fact=str(p.content) if isinstance(p.content, str) else str(p.content),
                        tier="t3",
                        born_on=datetime.now(UTC),
                        source_ref="run-a:0",
                    )
        return j.count()

    from paf.runtime import Session

    with Session(
        "assistant",
        trace=False,
        resume=False,
        store=store,
        memory_directory=str(memdir),
        consolidate_span=fence,
    ) as session:
        for i in range(12):
            session.history.extend(_msg(user=f"t{i}", assistant=f"a{i}"))

        async def scenario():
            await session._maybe_trigger_consolidation("c1")
            q = session.memory_queue
            assert q is not None, "cadence reached, so the queue must be constructed"
            await q.drain("assistant")  # let the fire-and-forget worker finish

        asyncio.run(scenario())

    # The worker folded the persisted span of complete turns (turns 2..10:
    # turns 0,1 consumed; turn 11 is the in-flight tip) and advanced the
    # durable watermark to the last complete turn folded.
    wm = MemoryWatermark.load(str(memdir))
    assert wm.run_id == "run-a"
    assert wm.turn_id == 10
    journal = JournalStore(str(memdir))
    rows = journal.read_all()
    folded = sorted(str(r.fact) for _, r in rows)
    assert "t2" in folded and "t10" in folded
    # the in-flight tip was never folded
    assert not any(str(r.fact) == "t11" for _, r in rows)
    assert (memdir / "MEMORY.md").exists()


def test_session_hook_skips_below_cadence(tmp_path: Path) -> None:
    """Below the cadence threshold the hook enqueues nothing."""
    msgs: list[ModelMessage] = []
    for i in range(5):
        msgs.extend(_msg(user=f"t{i}", assistant=f"a{i}"))
    store = _store(tmp_path, {"run-a": msgs})
    memdir = tmp_path / "mem"
    memdir.mkdir(exist_ok=True)
    JournalStore(str(memdir))
    # Watermark consumed "just now"; only ~5 turns/seconds later the tip is well
    # UNDER the 30-minute cadence, so the hook must not fire.
    MemoryWatermark(run_id="run-a", consumed_ts=datetime.now(UTC)).save(str(memdir))

    from paf.runtime import Session

    with Session(
        "assistant",
        trace=False,
        resume=False,
        store=store,
        memory_directory=str(memdir),
    ) as session:
        for i in range(5):
            session.history.extend(_msg(user=f"t{i}", assistant=f"a{i}"))

        asyncio.run(session._maybe_trigger_consolidation("c1"))

        q = session.memory_queue
        # Below cadence -> the queue is never even constructed.
        assert q is None


# ---------------------------------------------------------------------------
# 2. Queue fold advances the ref watermark against a real store
# ---------------------------------------------------------------------------


def test_queue_folds_real_store_and_advances_watermark(tmp_path: Path) -> None:
    """Draining a job with a real store folds complete turns, advances, renders."""
    store = _store(
        tmp_path,
        {"run-a": [*_msg(user="hello", assistant="hey"), *_msg(user="tip", assistant="t")]},
    )
    source = SnapshotHistorySource(store)
    memdir = tmp_path / "mem"
    memdir.mkdir(exist_ok=True)
    JournalStore(str(memdir))  # create the empty journal

    q = MemoryConsolidationQueue(
        memory_directory=str(memdir),
    )

    async def span(run: str, turn: int) -> ResolvedSpan:
        wm = MemoryWatermark.load(str(memdir), run_id=run)
        return await resolve_unseen_span(source, wm, tip_run=run, tip_turn=turn)

    async def fake_consolidate(messages: list[ModelMessage], directory: str, watermark: MemoryWatermark) -> int:
        j = JournalStore(directory)
        for m in messages:
            for p in m.parts:
                if isinstance(p, UserPromptPart):
                    j.append(
                        type="fact",
                        fact=str(p.content),
                        tier="t3",
                        born_on=datetime.now(UTC),
                        source_ref="run-a:0",
                    )
        return j.count()

    q.span_source = span
    q.consolidate_span = fake_consolidate

    async def go():
        # job position = the in-flight tip (run-a:1).
        await q.process_job("sess", ConsolidationJob(run_id="run-a", turn_id=1), None)

    asyncio.run(go())

    wm = MemoryWatermark.load(str(memdir))
    # watermark advanced to the last COMPLETE turn folded = run-a:0 (turn 1 is tip).
    assert wm.run_id == "run-a"
    assert wm.turn_id == 0
    assert (memdir / "memory-watermark.json").exists()
    assert (memdir / "MEMORY.md").exists()
    # the folded (complete) turn is in the journal (via read-back)
    j = JournalStore(str(memdir))
    rows = j.read_all()
    assert any(str(r.fact) == "hello" for _, r in rows), rows
    # the in-flight tip turn was NOT folded
    assert not any(str(r.fact) == "tip" for _, r in rows), rows
