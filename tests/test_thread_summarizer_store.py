"""Deterministic tests for the Phase-2 thread summary store."""

from __future__ import annotations

import json

import pytest

from paf.capabilities.thread_summarizer._capture import (
    CapturedMessage,
    EvictionBlock,
    append_eviction_block,
)
from paf.capabilities.thread_summarizer._store import (
    SUMMARIES_FILENAME,
    WATERMARK_FILENAME,
    ThreadSummaryRecord,
    ThreadSummaryStore,
    ThreadWatermark,
    conversation_directory,
)


def _block(seq: int, cid: str) -> EvictionBlock:
    return EvictionBlock(
        seq=seq,
        conversation_id=cid,
        evicted_at=f"2026-09-08T00:00:0{seq}+00:00",
        keep_messages=2,
        messages=[CapturedMessage(ref=f"run:{seq}", role="user", text=f"turn {seq}")],
    )


def _record(seq: int, cid: str, thread: str = "thread") -> ThreadSummaryRecord:
    return ThreadSummaryRecord(
        eviction_seq=seq,
        conversation_id=cid,
        summarized_at="2026-09-08T00:00:00+00:00",
        thread=thread,
        refs=[f"run:{seq}"],
    )


def test_pending_blocks_filters_conversation_and_watermark(tmp_path):
    root = tmp_path / "assistant"
    for seq, cid in [(0, "p:c1"), (1, "p:c2"), (2, "p:c1"), (3, "p:c1")]:
        append_eviction_block(root, _block(seq, cid))

    store = ThreadSummaryStore(root, "p:c1")
    assert [block.seq for block in store.pending_blocks()] == [0, 2, 3]

    store.advance(2)
    assert [block.seq for block in store.pending_blocks()] == [3]
    assert ThreadWatermark.load(store.directory).eviction_seq == 2


def test_watermark_is_overwritten_atomically_not_appended(tmp_path):
    store = ThreadSummaryStore(tmp_path / "assistant", "p:c1")
    store.advance(4)
    store.advance(7)

    path = store.directory / WATERMARK_FILENAME
    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert json.loads(path.read_text(encoding="utf-8")) == {"eviction_seq": 7}
    assert not list(store.directory.glob(".watermark.json.staging"))


def test_corrupt_watermark_replays_from_wal(tmp_path):
    root = tmp_path / "assistant"
    append_eviction_block(root, _block(0, "p:c1"))
    store = ThreadSummaryStore(root, "p:c1")
    store.directory.mkdir(parents=True)
    (store.directory / WATERMARK_FILENAME).write_text("{broken", encoding="utf-8")

    assert [block.seq for block in store.pending_blocks()] == [0]


def test_summary_append_is_idempotent_by_eviction_seq(tmp_path):
    store = ThreadSummaryStore(tmp_path / "assistant", "p:c1")
    record = _record(2, "p:c1")
    store.append_summary(record)
    store.append_summary(record)

    assert store.existing_summaries() == [record]
    assert (store.directory / SUMMARIES_FILENAME).read_text(encoding="utf-8").count("\n") == 1


def test_summary_append_rejects_wrong_conversation_and_conflicting_duplicate(tmp_path):
    store = ThreadSummaryStore(tmp_path / "assistant", "p:c1")
    store.append_summary(_record(0, "p:c1"))

    with pytest.raises(ValueError, match="conversation_id"):
        store.append_summary(_record(1, "p:c2"))
    with pytest.raises(ValueError, match="different content"):
        store.append_summary(_record(0, "p:c1", thread="changed"))


def test_watermark_cannot_move_backward(tmp_path):
    store = ThreadSummaryStore(tmp_path / "assistant", "p:c1")
    store.advance(3)

    with pytest.raises(ValueError, match="backward"):
        store.advance(2)


def test_conversation_directories_are_stable_and_separate(tmp_path):
    root = tmp_path / "assistant"
    assert conversation_directory(root, "p:c1") == conversation_directory(root, "p:c1")
    assert conversation_directory(root, "p:c1") != conversation_directory(root, "p:c2")
