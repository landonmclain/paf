"""Offline tests for the Phase-2 runtime thread trigger."""

from __future__ import annotations

import asyncio

from paf.capabilities.thread_summarizer._capture import (
    CapturedMessage,
    EvictionBlock,
    append_eviction_block,
)
from paf.capabilities.thread_summarizer._summarize import Beat, BeatsRecord
from paf.runtime import Session


def _block(seq: int, cid: str) -> EvictionBlock:
    return EvictionBlock(
        seq=seq,
        conversation_id=cid,
        evicted_at="2026-09-08T00:00:00+00:00",
        keep_messages=2,
        messages=[CapturedMessage(ref=f"run:{seq}:0", role="user", text="text")],
    )


def _bare_session(root, *, enabled: bool, summarize_block=None) -> Session:
    session = object.__new__(Session)
    session._background_tasks = {}
    session._task_subscribers = []
    session._thread_summarizer_enabled = enabled
    session._thread_capture_root = str(root)
    session._thread_queue = None
    session._thread_summarize_block = summarize_block
    session._thread_summarizer_seq = 0
    return session


async def _beats(_block):
    return BeatsRecord(
        turns_covered=[],
        beats=[Beat(at="run:0:0", what="did work", why="in block")],
        open=[],
        resumed_from="work",
    )


def test_off_trigger_does_not_touch_capture_store(tmp_path):
    session = _bare_session(tmp_path / "assistant", enabled=False)
    asyncio.run(session._maybe_trigger_thread_summarize("profile:cid"))

    assert session.thread_summarizer_queue is None
    assert not (tmp_path / "assistant").exists()
    assert session._background_tasks == {}


def test_on_trigger_enqueues_gap_and_worker_advances(tmp_path):
    root = tmp_path / "assistant"
    append_eviction_block(root, _block(0, "profile:cid"))
    session = _bare_session(root, enabled=True, summarize_block=_beats)

    async def scenario():
        await session._maybe_trigger_thread_summarize("profile:cid")
        assert session.thread_summarizer_queue is not None
        await session.thread_summarizer_queue.drain("profile:cid")

    asyncio.run(scenario())

    assert any(task.status == "done" for task in session._background_tasks.values())
    assert (root / "conversations").exists()


def test_trigger_single_flight_does_not_duplicate_jobs(tmp_path):
    root = tmp_path / "assistant"
    append_eviction_block(root, _block(0, "profile:cid"))
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(_block):
        started.set()
        await release.wait()
        return await _beats(_block)

    session = _bare_session(root, enabled=True, summarize_block=slow)

    async def scenario():
        await session._maybe_trigger_thread_summarize("profile:cid")
        await started.wait()
        await session._maybe_trigger_thread_summarize("profile:cid")
        release.set()
        assert session.thread_summarizer_queue is not None
        await session.thread_summarizer_queue.drain("profile:cid")

    asyncio.run(scenario())

    statuses = [task.status for task in session._background_tasks.values()]
    assert statuses.count("done") == 1
    assert "skipped" in statuses
