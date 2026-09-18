"""Offline tests for the conversation-scoped runtime thread trigger.

These drive the LIVE: ``_maybe_trigger_thread_summarize`` against a real
``FileStepStore`` + real ``Session`` wiring (with the ``thread_summarize_span``
seam injected so no LLM runs), mirroring
:mod:`test_memory_background_real_store`. They cover:

  - the trigger resolves THIS conversation's unseen turn-span and enqueues,
    gated by the ``thread_summarize_every`` cadence + single-flight;
  - the worker advances the per-conversation watermark on success;
  - two conversations never share a watermark or cross-summarize.
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
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    RunRecord,
)

from paf.background import BackgroundCoordinator
from paf.capabilities.memory._watermark import MemoryWatermark
from paf.capabilities.thread_summarizer._store import (
    THREAD_WATERMARK_FILENAME,
    ThreadSummaryStore,
)
from paf.capabilities.thread_summarizer._summarize import Beat, BeatsRecord
from paf.runtime import BackgroundTask, Session
from tests.framework import TEST_PROFILE


def _msg(*, user: str, assistant: str) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content=user)]),
        ModelResponse(parts=[TextPart(assistant)]),
    ]


def _store(
    tmp_path: Path, runs: dict[str, list[ModelMessage]], conversation_id: str
) -> FileStepStore:
    """Write a FileStepStore with one run per key, all under one conversation."""
    store = FileStepStore(directory=tmp_path / "store")
    for i, (run_id, msgs) in enumerate(runs.items()):
        # Stamp run_id on each message the way the persistence layer does, so
        # the worker can resolve {run_id}:{turn_id} refs from the stored span.
        for m in msgs:
            try:
                m.run_id = run_id
            except Exception:
                pass
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


async def _beats_from_span(span: list, cid: str) -> BeatsRecord:
    from paf.capabilities.thread_summarizer._summarize import _ref_for

    refs = [_ref_for(span, i) for i in range(len(span))]
    refs = [r for r in refs if r is not None]
    return BeatsRecord(
        turns_covered=refs,
        beats=[Beat(at=refs[0], what="did work")] if refs else [],
        summary="started the work and progressed",
    )


def _wm(store: ThreadSummaryStore) -> MemoryWatermark:
    return MemoryWatermark.load(store.directory, filename=THREAD_WATERMARK_FILENAME)


def test_trigger_off_does_not_touch_store(tmp_path: Path) -> None:
    """ThreadSummarizer not in the profile -> trigger is a No-Op."""
    store = _store(tmp_path, {"run-a": _msg(user="hi", assistant="hey")}, "profile:cid")

    with Session(
        "test", profile=TEST_PROFILE, trace=False, resume=False, store=store
    ) as session:
        session.background = BackgroundCoordinator(store=store, agent_name="test")
        session.background._thread_summarizer_enabled = False
        asyncio.run(session.background._maybe_trigger_thread_summarize("profile:cid"))
        assert session.background.thread_summarizer_queue is None
        assert session.background._background_tasks == {}


def test_trigger_enqueues_when_block_and_advances_watermark(tmp_path: Path) -> None:
    """>= BLOCK_TURNS complete turns -> trigger resolves span, worker folds, advances."""
    msgs: list[ModelMessage] = []
    for i in range(11):  # 10 complete turns (turn 10 is the in-flight tip)
        msgs.extend(_msg(user=f"t{i}", assistant=f"a{i}"))
    store = _store(tmp_path, {"run-a": msgs}, "profile:cid")
    capture_root = str(tmp_path / "summaries" / "assistant")
    # fresh watermark => whole backlog is new; 10 complete turns >= BLOCK_TURNS.
    wm_store = ThreadSummaryStore(capture_root, "profile:cid")
    wm_store.directory.mkdir(parents=True, exist_ok=True)

    with Session(
        "test",
        profile=TEST_PROFILE,
        trace=False,
        resume=False,
        store=store,
    ) as session:
        session.background = BackgroundCoordinator(
            store=store,
            agent_name="test",
            thread_summarize_span=_beats_from_span,
        )
        session.background._thread_capture_root = capture_root
        session.background._thread_summarizer_enabled = True

        async def scenario():
            await session.background._maybe_trigger_thread_summarize("profile:cid")
            q = session.background.thread_summarizer_queue
            assert q is not None, (
                ">= 10 complete turns -> the queue must be constructed"
            )
            await q.drain("profile:cid")

        asyncio.run(scenario())

    # watermark advanced to the last COMPLETE turn folded (turn 9; turn 10 is tip)
    wm = _wm(ThreadSummaryStore(capture_root, "profile:cid"))
    assert wm.run_id == "run-a"
    assert wm.turn_id == 9
    assert any(
        task.status == "done" for task in session.background._background_tasks.values()
    )

    # exactly one 10-turn block was written (sparse {at,what} beats + a summary)
    recs = ThreadSummaryStore(capture_root, "profile:cid").existing_summaries()
    assert len(recs) == 1
    assert len(recs[0].refs) == 10  # deduped to the 10 distinct covered refs
    assert recs[0].summary != ""


def test_trailing_partial_block_is_not_summarized(tmp_path: Path) -> None:
    """Regression: a resolved span with a non-multiple of BLOCK_TURNS leaves a
    trailing partial block that must NOT be summarized as a thin 1-9 turn record.

    A 12-complete-turn span (12 user/assistant pairs => 11 complete + in-flight
    tip) chunks as 10 + 1 leftover. Before the fix the worker summarized the
    leftover 1-turn block too (e.g. a record with n_refs=1 and a "only 1 turn in
    this block" summary — exactly the live miss). After the fix only the full
    10-turn block is written; the trailing turn is left to fold into a later
    >= BLOCK_TURNS block, while the watermark still advances past it.
    """
    msgs: list[ModelMessage] = []
    for i in range(12):  # 11 complete turns, turn 12 is the in-flight tip
        msgs.extend(_msg(user=f"t{i}", assistant=f"a{i}"))
    store = _store(tmp_path, {"run-a": msgs}, "profile:cid")
    capture_root = str(tmp_path / "summaries" / "assistant")
    wm_store = ThreadSummaryStore(capture_root, "profile:cid")
    wm_store.directory.mkdir(parents=True, exist_ok=True)

    with Session(
        "test",
        profile=TEST_PROFILE,
        trace=False,
        resume=False,
        store=store,
    ) as session:
        session.background = BackgroundCoordinator(
            store=store,
            agent_name="test",
            thread_summarize_span=_beats_from_span,
        )
        session.background._thread_capture_root = capture_root
        session.background._thread_summarizer_enabled = True

        async def scenario():
            await session.background._maybe_trigger_thread_summarize("profile:cid")
            q = session.background.thread_summarizer_queue
            assert q is not None, ">= 10 complete turns -> queue constructed"
            await q.drain("profile:cid")

        asyncio.run(scenario())

    # Only the FULL 10-turn block is written — no thin trailing (1-turn) record.
    recs = ThreadSummaryStore(capture_root, "profile:cid").existing_summaries()
    assert len(recs) == 1
    assert len(recs[0].refs) == 10

    # The watermark still advances over the trailing turn (turn 10), so it is
    # NOT re-summarized per turn — it folds into the next >= BLOCK_TURNS block.
    wm = _wm(ThreadSummaryStore(capture_root, "profile:cid"))
    assert wm.run_id == "run-a"
    assert wm.turn_id == 10
    assert any(
        task.status == "done" for task in session.background._background_tasks.values()
    )


def test_fresh_watermark_enqueues_across_multi_run_backlog(tmp_path: Path) -> None:
    """Regression: a brand-new conversation's WHOLE backlog must be seen as new.

    Reproduces the deadlock (2026-09-09): the production trigger loads the
    per-conversation watermark via ``MemoryWatermark.load(dir, run_id=tip_run)``.
    Before the fix, a missing watermark file returned a fresh watermark re-
    anchored at the TIP run, and ``resolve_unseen_span`` read that as
    *mid-history* — every run strictly older than the tip was treated as
    consumed. For a conversation whose >=10-turn backlog is spread across
    several runs (each run individually < BLOCK_TURNS), only the tip run's
    turns counted, the threshold was never reached, the watermark was never
    persisted, and it re-bootstrapped at the new tip every turn: the whole
    backlog was silently discarded and summarization never fired.

    Here: 6 turns in run-a + 6 in run-b (each < 10, but 12 total). A fresh
    watermark must treat the WHOLE backlog as new -> >=10 -> enqueues.
    """
    msgs_a = []
    for i in range(6):
        msgs_a.extend(_msg(user=f"a{i}", assistant=f"ra{i}"))
    msgs_b = []
    for i in range(6):
        msgs_b.extend(_msg(user=f"b{i}", assistant=f"rb{i}"))
    store = _store(tmp_path, {"run-a": msgs_a, "run-b": msgs_b}, "profile:cid")
    capture_root = str(tmp_path / "summaries" / "assistant")
    # Fresh watermark: ThreadSummaryStore dir exists but holds NO watermark file.
    ThreadSummaryStore(capture_root, "profile:cid").directory.mkdir(
        parents=True, exist_ok=True
    )

    with Session(
        "test",
        profile=TEST_PROFILE,
        trace=False,
        resume=False,
        store=store,
    ) as session:
        session.background = BackgroundCoordinator(
            store=store,
            agent_name="test",
            thread_summarize_span=_beats_from_span,
        )
        session.background._thread_capture_root = capture_root
        session.background._thread_summarizer_enabled = True

        async def scenario():
            await session.background._maybe_trigger_thread_summarize("profile:cid")
            q = session.background.thread_summarizer_queue
            assert q is not None, (
                ">=10 complete turns across the backlog -> must enqueue"
            )
            await q.drain("profile:cid")

        asyncio.run(scenario())

    # Whole backlog must be folded (up to the in-flight tip of run-b).
    wm = _wm(ThreadSummaryStore(capture_root, "profile:cid"))
    assert wm.turn_id is not None
    assert any(
        task.status == "done" for task in session.background._background_tasks.values()
    )
    recs = ThreadSummaryStore(capture_root, "profile:cid").existing_summaries()
    assert len(recs) >= 1
    refs = recs[0].refs
    # The fold must straddle BOTH runs — run-a's 6 turns AND run-b's lead turns.
    # (Before the fix, only the tip run counted and this never enqueued at all.)
    assert any(r.startswith("run-a:") for r in refs)
    assert any(r.startswith("run-b:") for r in refs)
    # 10 complete turns folded (12 total - in-flight tip - one trailing partial).
    assert len(refs) == 10


def test_trigger_skips_below_turn_threshold(tmp_path: Path) -> None:
    """< BLOCK_TURNS complete turns since the watermark -> never enqueues."""
    msgs: list[ModelMessage] = []
    for i in range(11):  # 10 total complete turns, but watermark consumed turn 8
        msgs.extend(_msg(user=f"t{i}", assistant=f"a{i}"))
    store = _store(tmp_path, {"run-a": msgs}, "profile:cid")
    capture_root = str(tmp_path / "summaries" / "assistant")
    # watermark at turn 8 => span = turns 9..9 = 1 turn < BLOCK_TURNS => skip.
    store_dir = ThreadSummaryStore(capture_root, "profile:cid").directory
    store_dir.mkdir(parents=True, exist_ok=True)
    MemoryWatermark(run_id="run-a", turn_id=8, consumed_ts=datetime.now(UTC)).save(
        store_dir, filename=THREAD_WATERMARK_FILENAME
    )

    with Session(
        "test",
        profile=TEST_PROFILE,
        trace=False,
        resume=False,
        store=store,
    ) as session:
        session.background = BackgroundCoordinator(
            store=store,
            agent_name="test",
            thread_summarize_span=_beats_from_span,
        )
        session.background._thread_capture_root = capture_root
        session.background._thread_summarizer_enabled = True
        asyncio.run(session.background._maybe_trigger_thread_summarize("profile:cid"))
        assert session.background.thread_summarizer_queue is None


def test_two_conversations_never_share_a_watermark(tmp_path: Path) -> None:
    """Conversation isolation: cA and cB keep independent watermarks."""
    _store(tmp_path, {"ra": _msg(user="a0", assistant="a1")}, "profile:cA")
    _store(tmp_path, {"rb": _msg(user="b0", assistant="b1")}, "profile:cB")
    capture_root = str(tmp_path / "summaries" / "assistant")

    # Simulate an existing per-conversation watermark for cA only (cB fresh).
    ca_store = ThreadSummaryStore(capture_root, "profile:cA")
    ca_store.directory.mkdir(parents=True, exist_ok=True)
    MemoryWatermark(run_id="ra", turn_id=0, consumed_ts=datetime.now(UTC)).save(
        ca_store.directory, filename=THREAD_WATERMARK_FILENAME
    )
    cb_store = ThreadSummaryStore(capture_root, "profile:cB")

    assert _wm(ca_store).run_id == "ra"
    # cB's watermark is untouched by cA's (still fresh, empty).
    wm_b = _wm(cb_store)
    assert wm_b.run_id == "" and wm_b.turn_id is None
    # distinct files
    from paf.capabilities.thread_summarizer._store import conversation_directory

    assert conversation_directory(capture_root, "profile:cA") != conversation_directory(
        capture_root, "profile:cB"
    )


def test_task_history_scopes_and_retains_error_rows(tmp_path: Path) -> None:
    """``task_history`` filters by conversation and keeps terminal error rows.

    The thread summarizer fails AFTER the turn stream closes, so its ERROR task
    must remain discoverable via ``task_history`` — including the error detail —
    for a UI to surface "background task failed" without relying on the live
    turn channel.
    """
    store = _store(tmp_path, {"ra": _msg(user="hi", assistant="hey")}, "profile:cA")

    with Session(
        "test", profile=TEST_PROFILE, trace=False, resume=False, store=store
    ) as session:
        session.background = BackgroundCoordinator(store=store, agent_name="test")
        session.background._thread_summarizer_enabled = True
        # Simulate a failed thread summarizer task for conversation cA.
        session.background._thread_summarizer_seq = 1
        session.background._publish_task(
            BackgroundTask(
                id="thread_summarize.profile:cA.1",
                label="Summarizing thread (profile:cA)",
                status="error",
                detail="summary run/turn anchor must increase",
            )
        )

        got_a = session.background.task_history("profile:cA")
        assert len(got_a) == 1
        assert got_a[0].status == "error"
        assert "anchor must increase" in got_a[0].detail

        # Conversation isolation: cB (different cid) sees nothing.
        assert session.background.task_history("profile:cB") == []
