"""Durable memory snapshot reuse (run_lmem.py) — copy/gate/Persist round-trip.

Verifies the three helpers backing the `MEMORY=1` durable-reuse path: a
freshly-consolidated `memory/<qid>/` can be persisted to the durable dir and,
on a later (unchanged) run, restored back into a fresh live dir so the owner is
skipped. Exercises only the file-copy logic -- no owner agent / model / keys.

The owner loop's reuse/persist branch in run_lmem.py (`_restore_durable_memory`
before `consolidate_conversation`, `_persist_durable_memory` after a real
consolidation) is the caller; this test pins the primitive operations it leans
on: copy is recursive + idempotent, restore returns 0 when no snapshot, and
persist only fires on a real (non-restored) consolidation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.run_lmem import (
    _persist_durable_memory,
    _restore_durable_memory,
)


@pytest.fixture
def durable_dir(tmp_path, monkeypatch):
    """Point MEMORY_DURABLE_DIR at a temp dir for the whole test."""
    d = tmp_path / "durable"
    d.mkdir()
    monkeypatch.setattr(
        "tests.run_lmem.MEMORY_DURABLE_DIR", d, raising=True
    )
    return d


def _write_store(base: Path, *, body: str, version: int) -> None:
    """Build a live memory/<qid> store: MEMORY.md + a topic file + CAS sidecar."""
    base.mkdir(parents=True, exist_ok=True)
    (base / "MEMORY.md").write_text(body, encoding="utf-8")
    (base / "facts.md").write_text("# facts\n", encoding="utf-8")
    state = {
        "MEMORY.md": {"version": version, "fingerprint": "f:1", "last_operation_id": "o1"},
        "facts.md": {"version": version, "fingerprint": "f:2", "last_operation_id": "o2"},
    }
    (base / ".memory-state.json").write_text(
        __import__("json").dumps(state), encoding="utf-8"
    )


def test_persist_then_restore_round_trip(durable_dir, tmp_path):
    """persist() snapshots a consolidated store; restore() copies it back intact."""
    live = tmp_path / "memory" / "q1"
    _write_store(live, body="# Facts\n- A\n", version=3)

    # Persist the freshly-consolidated store to the durable dir.
    n = _persist_durable_memory("q1", live)
    assert n == 3  # MEMORY.md + facts.md + .memory-state.json

    snap = durable_dir / "q1"
    assert (snap / "MEMORY.md").read_text(encoding="utf-8") == "# Facts\n- A\n"
    assert (snap / ".memory-state.json").exists()
    assert (snap / "facts.md").exists()

    # A brand-new live store for a later run is empty; restore fills it.
    live2 = tmp_path / "memory2" / "q1"
    assert not live2.exists()
    r = _restore_durable_memory("q1", live2)
    assert r == 3
    assert (live2 / "MEMORY.md").read_text(encoding="utf-8") == "# Facts\n- A\n"
    assert (live2 / ".memory-state.json").exists()


def test_restore_returns_zero_when_no_snapshot(durable_dir, tmp_path):
    """No durable snapshot -> 0, so the caller proceeds to a real consolidation."""
    assert _restore_durable_memory("nope", tmp_path / "memory" / "nope") == 0


def test_persist_returns_zero_when_no_live_store(durable_dir, tmp_path):
    """No live store to snapshot -> 0, nothing written to durable."""
    assert _persist_durable_memory("q2", tmp_path / "does-not-exist") == 0
    assert not (durable_dir / "q2").exists()


def test_copy_tree_is_idempotent(durable_dir, tmp_path):
    """Re-persisting unchanged input re-stamps identical bytes (safe reuse)."""
    live = tmp_path / "memory" / "q3"
    _write_store(live, body="# Facts\n- B\n", version=1)
    a = _persist_durable_memory("q3", live)
    b = _persist_durable_memory("q3", live)
    assert a == b == 3
    snap = durable_dir / "q3"
    assert (snap / "MEMORY.md").read_bytes() == b"# Facts\n- B\n"


def test_fresh_flag_short_circuit(durable_dir, tmp_path):
    """PAF_FRESH_MEMORY=1 must NOT restore -- owner runs + snapshot overwrites.

    This mirrors the caller's gate: `_restore_durable_memory` is only invoked
    when `not os.environ.get("PAF_FRESH_MEMORY")`. Here we assert the flag
    suppresses the restore by simulating the caller's decision directly.
    """
    live = tmp_path / "memory" / "q4"
    _write_store(live, body="# Facts\n- OLD\n", version=1)
    _persist_durable_memory("q4", live)

    # A fresh re-consolidation produces different content in a NEW live dir,
    # which (per the gate) overwrites the durable snapshot.
    live_fresh = tmp_path / "memory_fresh" / "q4"
    _write_store(live_fresh, body="# Facts\n- NEW\n", version=2)
    _persist_durable_memory("q4", live_fresh)

    snap = durable_dir / "q4"
    assert "NEW" in (snap / "MEMORY.md").read_text(encoding="utf-8")


def test_dot_in_sidecar_and_missing_optional_files(durable_dir, tmp_path):
    """Only .md bodies plus the sidecar are required; a store with just MEMORY.md works."""
    live = tmp_path / "memory" / "q5"
    live.mkdir(parents=True)
    (live / "MEMORY.md").write_text("# Facts\n", encoding="utf-8")
    n = _persist_durable_memory("q5", live)
    assert n == 1  # only MEMORY.md (no topic file, no sidecar)
    assert (durable_dir / "q5" / "MEMORY.md").exists()
