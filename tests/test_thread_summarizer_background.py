"""Tests for the deterministic thread summarizer worker contract."""

from __future__ import annotations

import asyncio

import pytest

from paf.background import QueueState
from paf.capabilities.thread_summarizer._background import (
    ThreadSummarizeJob,
    ThreadSummarizerQueue,
)
from paf.capabilities.thread_summarizer._capture import (
    CapturedMessage,
    EvictionBlock,
    append_eviction_block,
)
from paf.capabilities.thread_summarizer._store import ThreadSummaryStore
from paf.capabilities.thread_summarizer._summarize import Beat, BeatsRecord


def _block(seq: int, cid: str = "profile:cid") -> EvictionBlock:
    return EvictionBlock(
        seq=seq,
        conversation_id=cid,
        evicted_at="2026-09-08T00:00:00+00:00",
        keep_messages=2,
        messages=[CapturedMessage(ref=f"run:{seq}:0", role="user", text=f"turn {seq}")],
    )


def _beats(block: EvictionBlock) -> BeatsRecord:
    ref = block.messages[0].ref or "unanchored"
    return BeatsRecord(
        turns_covered=[ref],
        beats=[Beat(at=ref, what=f"did {block.seq}", why="grounded in block")],
        open=[],
        resumed_from=f"block {block.seq}",
    )


def test_queue_summarizes_pending_blocks_and_advances_watermark(tmp_path):
    root = tmp_path / "assistant"
    append_eviction_block(root, _block(0))
    append_eviction_block(root, _block(1))
    seen: list[int] = []

    async def summarize(block):
        seen.append(block.seq)
        return _beats(block)

    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_block_fn=summarize)

    async def scenario():
        queue.enqueue("profile:cid", ThreadSummarizeJob("profile:cid"))
        await queue.drain("profile:cid")

    asyncio.run(scenario())

    store = ThreadSummaryStore(root, "profile:cid")
    assert seen == [0, 1]
    assert store.load_watermark().eviction_seq == 1
    assert [record.eviction_seq for record in store.existing_summaries()] == [0, 1]


def test_queue_advances_on_zero_pending_without_writing(tmp_path):
    root = tmp_path / "assistant"
    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_block_fn=_beats)

    asyncio.run(
        queue.process_job(
            "profile:cid",
            ThreadSummarizeJob("profile:cid"),
            QueueState(),
        )
    )

    store = ThreadSummaryStore(root, "profile:cid")
    assert store.load_watermark().eviction_seq == -1
    assert not store.existing_summaries()


def test_queue_failure_does_not_advance_watermark(tmp_path):
    root = tmp_path / "assistant"
    append_eviction_block(root, _block(0))

    async def fail(_block):
        raise RuntimeError("model unavailable")

    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_block_fn=fail)
    with pytest.raises(RuntimeError, match="model unavailable"):
        asyncio.run(
            queue.process_job(
                "profile:cid",
                ThreadSummarizeJob("profile:cid"),
                QueueState(),
            )
        )

    assert ThreadSummaryStore(root, "profile:cid").load_watermark().eviction_seq == -1


def test_queue_retry_is_idempotent_after_partial_failure(tmp_path):
    root = tmp_path / "assistant"
    append_eviction_block(root, _block(0))
    append_eviction_block(root, _block(1))
    calls = 0

    async def flaky(block):
        nonlocal calls
        calls += 1
        if block.seq == 1 and calls == 2:
            raise RuntimeError("second block failed")
        return _beats(block)

    queue = ThreadSummarizerQueue(capture_root=str(root), summarize_block_fn=flaky)
    with pytest.raises(RuntimeError):
        asyncio.run(
            queue.process_job("profile:cid", ThreadSummarizeJob("profile:cid"), QueueState())
        )
    assert ThreadSummaryStore(root, "profile:cid").load_watermark().eviction_seq == -1

    asyncio.run(
        queue.process_job("profile:cid", ThreadSummarizeJob("profile:cid"), QueueState())
    )
    store = ThreadSummaryStore(root, "profile:cid")
    assert store.load_watermark().eviction_seq == 1
    assert [record.eviction_seq for record in store.existing_summaries()] == [0, 1]


def test_queue_rejects_mismatched_session_and_job(tmp_path):
    queue = ThreadSummarizerQueue(capture_root=str(tmp_path))
    with pytest.raises(ValueError, match="session_id"):
        asyncio.run(
            queue.process_job("profile:other", ThreadSummarizeJob("profile:cid"), QueueState())
        )


def test_queue_status_and_drain_single_worker(tmp_path):
    root = tmp_path / "assistant"
    append_eviction_block(root, _block(0))
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(block):
        started.set()
        await release.wait()
        return _beats(block)

    async def scenario():
        queue = ThreadSummarizerQueue(capture_root=str(root), summarize_block_fn=slow)
        queue.enqueue("profile:cid", ThreadSummarizeJob("profile:cid"))
        await started.wait()
        assert queue.status("profile:cid").running
        release.set()
        assert await queue.drain_with_timeout("profile:cid", 1.0)
        assert queue.status("profile:cid").queued == 0

    asyncio.run(scenario())
