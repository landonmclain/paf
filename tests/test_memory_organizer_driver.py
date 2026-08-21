"""Deterministic driver tests for ``run_organizer`` (no-LLM short-circuits).

The live LLM classification is exercised by ``tests/run_organizer_live.py``
(which costs tokens and is NOT part of the plain pytest run). These tests pin
the parts of the driver that never need a model: the empty-journal and
already-caught-up short-circuits, plus the journal-tail renderer.
"""

from __future__ import annotations

import asyncio

from paf.capabilities.memory import IndexStore
from paf.capabilities.memory._journal import JournalStore
from paf.capabilities.memory._organize import render_journal_tail, run_organizer


def _rec(type, fact, born_on="2026-08-17T00:00:00Z", source_ref="run:0"):
    return {"type": type, "fact": fact, "born_on": born_on, "source_ref": source_ref}


def test_run_organizer_empty_journal_returns_no_changes(tmp_path) -> None:
    journal = JournalStore(tmp_path)
    idx = IndexStore(tmp_path)
    assert asyncio.run(run_organizer(journal, index_store=idx)) == "no changes"
    assert not (tmp_path / "memory-index.json").exists()


def test_run_organizer_caught_up_returns_no_changes(tmp_path) -> None:
    # Index already progressed past every journal record -> nothing new.
    journal = JournalStore(tmp_path)
    journal.append(type="fact", fact="already indexed",
                   born_on="2026-08-17T00:00:00Z", source_ref="run:0")
    idx = IndexStore(tmp_path)
    # Advance the index watermark past line 1 without entries (simulate prior run).
    idx._save([], watermark=1, next_id=1)
    assert asyncio.run(run_organizer(journal, index_store=idx)) == "no changes"


def test_render_journal_tail_shows_only_new_lines() -> None:
    from paf.capabilities.memory._journal import MemoryRecord

    recs = [
        (1, MemoryRecord(type="fact", fact="old", born_on="2026-08-16T00:00:00Z", source_ref="r:0")),
        (2, MemoryRecord(type="fact", fact="new", born_on="2026-08-17T00:00:00Z", source_ref="r:1")),
    ]
    out = render_journal_tail(recs, watermark=1)
    assert "line 2" in out
    assert "new" in out
    assert "line 1" not in out


def test_render_journal_tail_empty_when_caught_up() -> None:
    assert render_journal_tail([], watermark=0) == "(no new records)"
