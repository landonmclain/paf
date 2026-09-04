"""Persistence tests for the time-anchored ``MemoryWatermark``.

The consolidated watermark is moving from a per-run ``{run_id, turn_id}`` pair
(only monotonic within one run) to a **time anchor** (``consumed_ts``) that is
monotonic across runs — the fix specified in
``Designs/Background-Memory-Watermark-Spec.md``. ``{run_id, turn_id}`` is kept
as the precise ``source_ref`` address, no longer as the ordering key.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from paf.capabilities.memory._watermark import MemoryWatermark

TS0 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
TS1 = datetime(2026, 9, 2, 9, 30, 0, tzinfo=UTC)


def _roundtrip(wm: MemoryWatermark, tmp_path) -> MemoryWatermark:
    wm.save(tmp_path)
    return MemoryWatermark.load(tmp_path, run_id=wm.run_id)


def test_roundtrips_consumed_ts(tmp_path) -> None:
    wm = MemoryWatermark(run_id="r1", turn_id=12, consumed_ts=TS1)
    loaded = _roundtrip(wm, tmp_path)
    assert loaded.run_id == "r1"
    assert loaded.turn_id == 12
    assert loaded.consumed_ts == TS1


def test_fresh_watermark_empty_when_no_ts(tmp_path) -> None:
    loaded = MemoryWatermark.load(tmp_path, run_id="r1")
    assert loaded.consumed_ts is None
    assert loaded.is_empty()


def test_legacy_file_backcompat_no_ts(tmp_path) -> None:
    """A pre-change watermark file (no ``consumed_ts``) loads without error."""
    (tmp_path / "memory-watermark.json").write_text(
        json.dumps({"run_id": "r1", "turn_id": 12}), encoding="utf-8"
    )
    loaded = MemoryWatermark.load(tmp_path, run_id="r1")
    assert loaded.run_id == "r1"
    assert loaded.turn_id == 12
    assert loaded.consumed_ts is None


def test_file_stores_iso_timestamp(tmp_path) -> None:
    MemoryWatermark(run_id="r1", turn_id=5, consumed_ts=TS0).save(tmp_path)
    data = json.loads(
        (tmp_path / "memory-watermark.json").read_text(encoding="utf-8")
    )
    assert data["consumed_ts"] == TS0.isoformat()


def test_fresh_when_corrupt(tmp_path) -> None:
    (tmp_path / "memory-watermark.json").write_text("{ not json", encoding="utf-8")
    loaded = MemoryWatermark.load(tmp_path, run_id="r1")
    assert loaded.consumed_ts is None


def test_bad_timestamp_treated_as_fresh(tmp_path) -> None:
    (tmp_path / "memory-watermark.json").write_text(
        json.dumps({"run_id": "r1", "turn_id": 1, "consumed_ts": "not-a-date"}),
        encoding="utf-8",
    )
    loaded = MemoryWatermark.load(tmp_path, run_id="r1")
    assert loaded.consumed_ts is None
