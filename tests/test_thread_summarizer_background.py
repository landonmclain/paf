"""Tests for the conversation-scoped thread summarizer worker contract.

These exercise the deterministic mechanics of :class:`ThreadSummarizerQueue`
against the conversation-span + per-conversation ``MemoryWatermark`` model:

  1. A pre-resolved turn span is summarized (curated) and DUrable beats appended,
     then the per-conversation watermark advances to the span's target.
  2. An empty span is a successful pass: watermark still advances (nothing new).
  3. A thrown summarize op advances NOTHING so the span is retried next cadence.
  4. Retry after a partial failure is idempotent (anchor-based append).
  5. Session/job conversation mismatch is rejected.
  6. Queue status/drain single-flight shape.

The summarize seam is injected so no LLM is needed (mirrors the memory worker's
journal-fenced tests).
"""

from __future__ import annotations

import asyncio

import pytest

from paf.background import QueueState
from paf.capabilities.memory._watermark import MemoryWatermark
from paf.capabilities.thread_summarizer._background import (
    ThreadSummarizeJob,
    ThreadSummarizerQueue,
)
from paf.capabilities.thread_summarizer._store import (
    THREAD_WATERMARK_FILENAME,
    ThreadSummaryStore,
)
from paf.capabilities.thread_summarizer._summarize import Beat, BeatsRecord

CID = "profile:cid"


def _target(run: str = "run-a", turn: int = 1) -> MemoryWatermark:
    from datetime import UTC, datetime

    return MemoryWatermark(run_id=run, turn_id=turn, consumed_ts=datetime.now(UTC))


def _wm(store: ThreadSummaryStore) -> MemoryWatermark:
    return MemoryWatermark.load(store.directory, filename=THREAD_WATERMARK_FILENAME)


def _beats(run: str = "run-a", turn: int = 0) -> BeatsRecord:
    return BeatsRecord(
        turns_covered=[f"{run}:{turn}"],
        beats=[Beat(at=f"{run}:{turn}", what=f"did {run}:{turn}")],
        summary=f"started {run}:{turn} and progressed",
    )


def test_queue_summarizes_span_and_advances_watermark(tmp_path) -> None:
    root = tmp_path / "assistant"
    seen: list[MemoryWatermark] = []

    async def summarize(span, conversation_id):
        seen.append(conversation_id)
        return BeatsRecord(
            turns_covered=["run-a:0", "run-a:1"],
            beats=[Beat(at="run-a:0", what="hello")],
            summary="started and progressed",
        )

    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_span_fn=summarize)
    target = _target(turn=1)

    async def scenario():
        queue.enqueue(
            CID,
            ThreadSummarizeJob(
                conversation_id=CID,
                messages=_fake_span(2),
                watermark=target,
            ),
        )
        await queue.drain(CID)

    asyncio.run(scenario())

    store = ThreadSummaryStore(root, CID)
    assert seen == [CID]
    assert _wm(store).run_id == "run-a"
    assert _wm(store).turn_id == 1  # advanced to the target on success
    recs = store.existing_summaries()
    assert len(recs) == 1
    # record anchor === the watermark advance position (the span's end), not
    # the first beat ref
    assert recs[0].run_id == "run-a" and recs[0].turn_id == 1


def test_queue_advances_on_empty_span_without_writing(tmp_path) -> None:
    root = tmp_path / "assistant"
    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_span_fn=_beats)

    asyncio.run(
        queue.process_job(
            CID, ThreadSummarizeJob(conversation_id=CID, messages=[], watermark=_target(turn=0)), QueueState()
        )
    )

    store = ThreadSummaryStore(root, CID)
    assert _wm(store).turn_id == 0  # successful empty pass still advances
    assert not store.existing_summaries()


def test_queue_failure_does_not_advance_watermark(tmp_path) -> None:
    root = tmp_path / "assistant"

    async def fail(_span, _cid):
        raise RuntimeError("model unavailable")

    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_span_fn=fail)
    with pytest.raises(RuntimeError, match="model unavailable"):
        asyncio.run(
            queue.process_job(
                CID,
                ThreadSummarizeJob(
                    conversation_id=CID, messages=_fake_span(2), watermark=_target(turn=1)
                ),
                QueueState(),
            )
        )

    store = ThreadSummaryStore(root, CID)
    # watermark untouched -> span retried next cadence, never skipped
    wm = _wm(store)
    assert wm.run_id == "" and wm.turn_id is None


def test_queue_retry_is_idempotent_after_partial_failure(tmp_path) -> None:
    root = tmp_path / "assistant"
    calls = 0

    async def flaky(span, cid):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("first pass failed")
        return _beats(tmp_path / "not-used" if False else "run-a", 0)

    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_span_fn=flaky)
    job = ThreadSummarizeJob(
        conversation_id=CID, messages=_fake_span(2), watermark=_target(turn=1)
    )
    with pytest.raises(RuntimeError):
        asyncio.run(queue.process_job(CID, job, QueueState()))

    asyncio.run(queue.process_job(CID, job, QueueState()))

    store = ThreadSummaryStore(root, CID)
    assert _wm(store).turn_id == 1
    # idempotent: a retried pass over the same anchor writes exactly one record
    assert len(store.existing_summaries()) == 1


def test_queue_rejects_mismatched_session_and_job(tmp_path) -> None:
    queue = ThreadSummarizerQueue(capture_root=str(tmp_path))
    with pytest.raises(ValueError, match="session_id"):
        asyncio.run(
            queue.process_job(
                "profile:other", ThreadSummarizeJob(conversation_id=CID), QueueState()
            )
        )


def test_queue_status_and_drain_single_worker(tmp_path) -> None:
    root = tmp_path / "assistant"
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(span, cid):
        started.set()
        await release.wait()
        return _beats()

    async def scenario():
        queue = ThreadSummarizerQueue(capture_root=str(root), summarize_span_fn=slow)
        queue.enqueue(CID, ThreadSummarizeJob(conversation_id=CID, messages=_fake_span(2), watermark=_target()))
        await started.wait()
        assert queue.status(CID).running
        release.set()
        assert await queue.drain_with_timeout(CID, 1.0)
        assert queue.status(CID).queued == 0

    asyncio.run(scenario())


def _fake_span(n: int, run_id: str = "run-a") -> list:
    """A minimal persisted span with ``n`` complete turns (user + assistant).

    Messages carry the ``run_id`` stamp the store persists, so refs resolve.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    msgs: list = []
    for i in range(n):
        req = ModelRequest(parts=[UserPromptPart(content=f"u{i}")])
        resp = ModelResponse(parts=[TextPart(f"a{i}")])
        req.run_id = run_id
        resp.run_id = run_id
        msgs.append(req)
        msgs.append(resp)
    return msgs


def test_queue_chunks_span_into_10_turn_blocks(tmp_path) -> None:
    """A >10-turn span is folded as consecutive 10-turn blocks, each its own record.

    The watermark advances only past successfully-summarized blocks, so a 20-turn
    span yields exactly two 10-turn records and watermarks at the last turn covered.
    """
    import asyncio

    root = tmp_path / "assistant"
    seen_turns: list[int] = []

    async def summarize(span, conversation_id):
        from paf.capabilities.thread_summarizer._summarize import _ref_for

        seen_turns.append(len(_refs(span)))
        refs = _refs(span)
        return BeatsRecord(
            turns_covered=refs,
            beats=[Beat(at=refs[0], what="block beat") for _ in refs if refs],
            summary="a ten-turn block",
        )

    def _refs(msgs):
        from paf.capabilities.thread_summarizer._summarize import _ref_for

        return [r for r in (_ref_for(msgs, i) for i in range(len(msgs))) if r][::2]

    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_span_fn=summarize)
    target = _target(turn=19)

    async def scenario():
        queue.enqueue(
            CID,
            ThreadSummarizeJob(
                conversation_id=CID,
                messages=_fake_span(20, run_id="run-a"),
                watermark=target,
            ),
        )
        await queue.drain(CID)

    asyncio.run(scenario())

    store = ThreadSummaryStore(root, CID)
    recs = store.existing_summaries()
    assert len(recs) == 2  # 20 turns -> two 10-turn blocks
    assert seen_turns == [10, 10]  # the summarize fn saw two 10-turn slices
    assert {r.run_id for r in recs} == {"run-a"}
    assert sorted(r.turn_id for r in recs) == [9, 19]  # anchored at each block's last turn
    assert _wm(store).turn_id == 19  # advanced past the final block
