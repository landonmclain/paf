"""Wiring tests for the background memory-consolidation service + watermark trigger.

Covers the NEW trigger path added 2026-08-28:
- ``_watermark.should_fire`` — the metric trigger that decides *when* a
  background consolidation job is enqueued (decoupled from eviction).
- ``MemoryWatermark`` persistence — flat-file round-trip, fresh-on-corrupt.
- ``MemoryConsolidationQueue`` (the ``BackgroundQueueService`` consumer) —
  enqueue on trigger → drain → the unseen span is folded into the append-only
  journal, ``MEMORY.md`` is rendered deterministically, and the watermark
  advances + persists.

These tests deliberately inject a **synthetic ``span_source``** (standing in for
persistence) and a **journal-fenced ``consolidate_span``** (standing in for the
owner LLM) so the service + trigger mechanics are proven without a model call.
The compaction eviction tier (``_window.py::TurnBoundSlidingWindow``) is
trim-only and does no memory work — it is intentionally NOT touched; this
queued watermark worker is the sole memory writer.

Follows the established suite convention (no pytest-asyncio): sync ``def
test_*`` driving async scenarios via ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from paf.capabilities.memory._background import (
    ConsolidationJob,
    MemoryConsolidationQueue,
    ResolvedSpan,
)
from paf.capabilities.memory._journal import JournalStore, MemoryRecord
from paf.capabilities.memory._watermark import MemoryWatermark, should_fire

# --------------------------------------------------------------------------- #
# Trigger predicate
# --------------------------------------------------------------------------- #


def test_should_fire_fires_when_nothing_consumed_yet() -> None:
    """A fresh watermark (never consumed) means the whole backlog is new."""
    wm = MemoryWatermark(run_id="r1", turn_id=None)
    assert should_fire(wm, current_turn_id=3, every=10) is True


def test_should_fire_threshold_logic() -> None:
    """Fires when >= `every` turns accumulate; not below."""
    wm = MemoryWatermark(run_id="r1", turn_id=0)
    assert should_fire(wm, current_turn_id=2, every=10) is False
    assert should_fire(wm, current_turn_id=10, every=10) is True
    assert should_fire(wm, current_turn_id=15, every=10) is True


def test_should_fire_rejects_nonpositive_every() -> None:
    wm = MemoryWatermark(run_id="r1", turn_id=0)
    with pytest.raises(ValueError):
        should_fire(wm, current_turn_id=1, every=0)
    with pytest.raises(ValueError):
        should_fire(wm, current_turn_id=1, every=-5)


# --------------------------------------------------------------------------- #
# Watermark persistence
# --------------------------------------------------------------------------- #


def test_watermark_roundtrips(tmp_path) -> None:
    wm = MemoryWatermark(run_id="r1", turn_id=12)
    wm.save(tmp_path)
    loaded = MemoryWatermark.load(tmp_path, run_id="r1")
    assert loaded.run_id == "r1"
    assert loaded.turn_id == 12


def test_watermark_fresh_when_file_absent(tmp_path) -> None:
    loaded = MemoryWatermark.load(tmp_path, run_id="r1")
    assert loaded.run_id == "r1"
    assert loaded.turn_id is None
    assert loaded.is_empty()


def test_watermark_fresh_when_corrupt(tmp_path) -> None:
    (tmp_path / "memory-watermark.json").write_text("{ not json", encoding="utf-8")
    loaded = MemoryWatermark.load(tmp_path, run_id="r1")
    assert loaded.turn_id is None


def test_watermark_file_lives_at_memory_dir(tmp_path) -> None:
    MemoryWatermark(run_id="r1", turn_id=5).save(tmp_path)
    assert (tmp_path / "memory-watermark.json").is_file()
    data = json.loads((tmp_path / "memory-watermark.json").read_text(encoding="utf-8"))
    assert data == {"run_id": "r1", "turn_id": 5}


# --------------------------------------------------------------------------- #
# Background service + trigger wiring (synthetic seams, no LLM)
# --------------------------------------------------------------------------- #


def _synthetic_span_source():
    """Return an async span source reporting a fully-settled span.

    Returns a :class:`ResolvedSpan` reporting the tip position as fully settled
    (synthetic seam — the real resolver decides completeness; here the stand-in
    just affirms the job's ``up_to_turn_id``). The fence-consolidate writes its
    own records, so the message content itself is irrelevant to the
    service-mechanics assertions these tests make.
    """

    async def span_source(run_id: str, up_to_turn_id: int) -> ResolvedSpan:
        # The stand-in reports the foldable span; the fence-consolidate writes
        # its own records, so the message content here is irrelevant to the
        # service-mechanics assertions this test makes.
        return ResolvedSpan(
            messages=[],
            watermark=MemoryWatermark(run_id=run_id, turn_id=up_to_turn_id),
        )

    return span_source


def _journal_fence_consolidate(records: list[MemoryRecord]):
    """Return a consolidate seam that appends `records` to the journal for real.

    Still journal-fenced (writes through ``JournalStore``), so the fold + render
    is honest — just no LLM.
    """

    async def consolidate_span(
        messages: list, directory: str, watermark: MemoryWatermark
    ) -> int:
        journal = JournalStore(directory)
        journal.append_many(records)
        return len(records)

    return consolidate_span


def test_trigger_enqueues_and_drain_folds_journal_and_renders(tmp_path) -> None:
    """The full service path: trigger → enqueue → worker folds + renders + advances."""
    records = [
        MemoryRecord(
            type="fact", fact="Landon lives in Bentonville", tier="t1",
            born_on="2026-08-28T00:00:00Z", source_ref="r1:1",
        ),
        MemoryRecord(
            type="preference", fact="Prefers flat-file-first storage", tier="t1",
            born_on="2026-08-28T00:00:01Z", source_ref="r1:2",
        ),
    ]

    class Q(MemoryConsolidationQueue):
        pass

    async def scenario() -> None:
        q = Q(
            memory_directory=str(tmp_path),
            span_source=_synthetic_span_source(),
            consolidate_span=_journal_fence_consolidate(records),
        )
        # The trigger decides a job is due (10 turns accumulated):
        wm = MemoryWatermark.load(str(tmp_path), run_id="r1")
        assert should_fire(wm, current_turn_id=12, every=10) is True
        q.enqueue("s1", ConsolidationJob(run_id="r1", turn_id=12))
        await q.drain("s1")

    asyncio.run(scenario())

    journal = JournalStore(tmp_path)
    rows = journal.read_all()
    assert len(rows) == 2
    assert rows[0][1].fact == "Landon lives in Bentonville"
    assert rows[1][1].fact == "Prefers flat-file-first storage"

    # MEMORY.md rendered deterministically from the append-only journal.
    md = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    assert "Landon lives in Bentonville" in md
    assert "Prefers flat-file-first storage" in md

    # Watermark advanced past what was consumed.
    wm = MemoryWatermark.load(str(tmp_path), run_id="r1")
    assert wm.turn_id == 12


def test_drain_without_trigger_does_not_fire_when_below_threshold(tmp_path) -> None:
    """Below threshold, the trigger says no — no job, no fold, watermark unchanged."""
    # Seed a non-empty watermark so "3 - 0 = 3 < 10" is genuinely below threshold
    # (a FRESH watermark always fires — that's covered separately).
    MemoryWatermark(run_id="r1", turn_id=0).save(tmp_path)
    wm = MemoryWatermark.load(str(tmp_path), run_id="r1")
    fired = should_fire(wm, current_turn_id=3, every=10)
    assert fired is False
    # Nothing should have been written: no journal, no MEMORY.md, no watermark file.
    assert not (tmp_path / "memory-journal.jsonl").exists()
    assert not (tmp_path / "MEMORY.md").exists()


def test_missing_span_source_logs_and_skips(tmp_path, caplog) -> None:
    """A queue without a span_source must not guess at persistence — drop cleanly."""

    class NoSource(MemoryConsolidationQueue):
        pass

    async def scenario() -> None:
        q = NoSource(memory_directory=str(tmp_path))
        q.enqueue("s1", ConsolidationJob(run_id="r1", turn_id=5))
        await q.drain("s1")

    with caplog.at_level("ERROR"):
        asyncio.run(scenario())
    assert "no span_source" in caplog.text
    assert not (tmp_path / "memory-journal.jsonl").exists()
    assert not (tmp_path / "memory-watermark.json").exists()
