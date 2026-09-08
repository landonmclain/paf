"""Deterministic tests for the conversation-scoped thread summary store.

No eviction WAL here (resolved by the 2026-09-08 re-architecture): the store
owns one conversation's appended ``summaries.jsonl``, anchored by the
``{run_id, turn_id}`` watermark position each successful summarize pass advanced
to. These tests cover append idempotency (retry-safe), ordering, per-conversation
scoping, and the stable hashed conversation directory.
"""

from __future__ import annotations

import json

import pytest

from paf.capabilities.thread_summarizer._store import (
    SUMMARIES_FILENAME,
    ThreadSummaryRecord,
    ThreadSummaryStore,
    conversation_directory,
)


def _record(run: str = "run-a", turn: int = 0, cid: str = "p:c1", summary: str = "t") -> ThreadSummaryRecord:
    return ThreadSummaryRecord(
        conversation_id=cid,
        run_id=run,
        turn_id=turn,
        summarized_at="2026-09-08T00:00:00+00:00",
        summary=summary,
        refs=[f"{run}:{turn}"],
    )


def test_summary_append_is_idempotent_by_run_turn_anchor(tmp_path) -> None:
    store = ThreadSummaryStore(tmp_path / "assistant", "p:c1")
    record = _record(run="run-a", turn=2)
    store.append_summary(record)
    store.append_summary(record)  # duplicate pass over the same span -> no-op

    assert store.existing_summaries() == [record]
    # exactly one line in the file
    assert (store.directory / SUMMARIES_FILENAME).read_text(encoding="utf-8").count("\n") == 1


def test_summary_append_rejects_wrong_conversation_and_conflicting_duplicate(tmp_path) -> None:
    store = ThreadSummaryStore(tmp_path / "assistant", "p:c1")
    store.append_summary(_record(run="run-a", turn=0))

    with pytest.raises(ValueError, match="conversation_id"):
        store.append_summary(_record(run="run-b", turn=0, cid="p:c2"))
    with pytest.raises(ValueError, match="different content"):
        store.append_summary(_record(run="run-a", turn=0, summary="changed"))


def test_summary_append_requires_increasing_anchor(tmp_path) -> None:
    store = ThreadSummaryStore(tmp_path / "assistant", "p:c1")
    store.append_summary(_record(run="run-a", turn=5))

    with pytest.raises(ValueError, match="must increase"):
        store.append_summary(_record(run="run-a", turn=3))


def test_conversation_directories_are_stable_and_separate(tmp_path) -> None:
    root = tmp_path / "assistant"
    assert conversation_directory(root, "p:c1") == conversation_directory(root, "p:c1")
    assert conversation_directory(root, "p:c1") != conversation_directory(root, "p:c2")


def test_conversation_no_cross_read(tmp_path) -> None:
    """c1's summaries never appear in c2's store."""
    root = tmp_path / "assistant"
    ThreadSummaryStore(root, "p:c1").append_summary(_record(run="run-a", turn=0, cid="p:c1"))
    assert len(ThreadSummaryStore(root, "p:c2").existing_summaries()) == 0


def test_watermark_file_contract(tmp_path) -> None:
    """The per-conversation watermark is a distinct file co-located with summaries."""
    from paf.capabilities.memory._watermark import MemoryWatermark

    store = ThreadSummaryStore(tmp_path / "assistant", "p:c1")
    store.directory.mkdir(parents=True, exist_ok=True)
    MemoryWatermark(run_id="run-a", turn_id=9).save(store.directory, filename="thread-watermark.json")

    path = store.directory / "thread-watermark.json"
    assert path.read_text(encoding="utf-8").count("\n") == 1
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["run_id"] == "run-a" and row["turn_id"] == 9
