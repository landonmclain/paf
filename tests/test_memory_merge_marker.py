"""The tier render's ``[merged: N]`` marker on legacy merge rows."""

from __future__ import annotations

from paf.capabilities.memory._journal import JournalStore, MemoryRecord
from paf.capabilities.memory._render import render_journal_memory


def _rec(fact, *, src="run:0", tier="t1", typ="fact"):
    return MemoryRecord(
        type=typ, fact=fact, tier=tier, born_on="2026-08-17T00:00:00Z", source_ref=src
    )


def test_render_adds_merge_count_marker(tmp_path):
    store = JournalStore(tmp_path)
    store.append_record(_rec("merged fact", src="merge(#2,#3,#4)"))
    store.append_record(_rec("source a", src="run:2"))
    store.append_record(_rec("source b", src="run:4"))
    store.append_record(_rec("source c", src="run:8"))
    md = render_journal_memory(store)
    assert "[merged: 3]" in md


def test_render_no_marker_for_non_merged(tmp_path):
    store = JournalStore(tmp_path)
    store.append_record(_rec("plain fact", src="run:0"))
    md = render_journal_memory(store)
    assert "[merged:" not in md
