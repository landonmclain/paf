"""``expand_memory`` tool + the ``[merged: N]`` render marker.

Covers the recursive memory-line resolver (through nested merges to the leaf
source turns), the ``expand_memory`` read tool on ``OverrideToolset``, and the
render's merge-count marker — deterministically, no live LLM.
"""

from __future__ import annotations

import asyncio

from paf.capabilities.memory._journal import JournalStore, MemoryRecord
from paf.capabilities.memory._overrides import OverrideStore
from paf.capabilities.memory._render import render_journal_memory
from paf.capabilities.memory._toolset import (
    OverrideToolset,
    _merge_source_ids,
    _resolve_memory_leaves,
)


def _rec(fact, *, src="run:0", tier="t1", typ="fact"):
    return MemoryRecord(
        type=typ, fact=fact, tier=tier, born_on="2026-08-17T00:00:00Z", source_ref=src
    )


class _NullSpan:
    def is_recording(self):
        return False

    def set_attributes(self, attrs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return None


class _FakeCtx:
    def __init__(self):
        self.tracer = type(
            "T",
            (),
            {"start_as_current_span": staticmethod(lambda *a, **k: _NullSpan())},
        )()


def _by_line(tmp_path, records):
    store = JournalStore(tmp_path)
    for r in records:
        store.append_record(r)
    return {ln: rec for ln, rec in store.read_all()}


# ---------------------------------------------------------------------------
# resolver
# ---------------------------------------------------------------------------


def test_merge_source_ids_parses_merge_ref():
    assert _merge_source_ids("merge(#3,#8,#12)") == [3, 8, 12]


def test_merge_source_ids_non_merge_is_empty():
    assert _merge_source_ids("run:0") == []


def test_resolve_single_leaf_returns_itself(tmp_path):
    by_line = _by_line(tmp_path, [_rec("A fact", src="run:2")])
    assert _resolve_memory_leaves(by_line, 1) == [(1, by_line[1])]


def test_resolve_merged_expands_to_leaves(tmp_path):
    by_line = _by_line(
        tmp_path,
        [
            _rec("merged", src="merge(#2,#3)"),
            _rec("source a", src="run:2"),
            _rec("source b", src="run:4"),
        ],
    )
    leaves = _resolve_memory_leaves(by_line, 1)
    assert [(ln, r.source_ref) for ln, r in leaves] == [(2, "run:2"), (3, "run:4")]


def test_resolve_nested_merges(tmp_path):
    by_line = _by_line(
        tmp_path,
        [
            _rec("top", src="merge(#2)"),
            _rec("mid", src="merge(#3,#4)"),
            _rec("leaf a", src="run:2"),
            _rec("leaf b", src="run:4"),
        ],
    )
    leaves = _resolve_memory_leaves(by_line, 1)
    assert [(ln, r.source_ref) for ln, r in leaves] == [(3, "run:2"), (4, "run:4")]


def test_resolve_unknown_id_is_empty(tmp_path):
    by_line = _by_line(tmp_path, [_rec("only", src="run:0")])
    assert _resolve_memory_leaves(by_line, 999) == []


# ---------------------------------------------------------------------------
# tool
# ---------------------------------------------------------------------------


def test_override_toolset_exposes_archive_and_expand(tmp_path):
    ts = OverrideToolset(OverrideStore(tmp_path))
    assert {"memory_archive_suggestion", "expand_memory"} <= set(ts.tools.keys())


def test_expand_memory_echoes_single_leaf(tmp_path):
    store = JournalStore(tmp_path)
    store.append_record(_rec("Landon is married to Melanie", src="run:2"))
    ts = OverrideToolset(OverrideStore(tmp_path))
    result = asyncio.run(ts.expand_memory(_FakeCtx(), line_id=1))
    assert result == "turn run:2 — memory: Landon is married to Melanie"


def test_expand_memory_returns_all_source_turns(tmp_path):
    store = JournalStore(tmp_path)
    store.append_record(_rec("merged identity", src="merge(#2,#3,#4)"))
    store.append_record(_rec("Landon is a software engineer", src="run:2"))
    store.append_record(_rec("Landon is married to Melanie", src="run:4"))
    store.append_record(_rec("Landon has three children", src="run:8"))
    ts = OverrideToolset(OverrideStore(tmp_path))
    result = asyncio.run(ts.expand_memory(_FakeCtx(), line_id=1))
    assert result.startswith("#1 is a merge of 3 source lines")
    assert "turn run:2" in result and "software engineer" in result
    assert "turn run:4" in result and "married to Melanie" in result
    assert "turn run:8" in result and "three children" in result


def test_expand_memory_unknown_id(tmp_path):
    JournalStore(tmp_path)  # empty journal
    ts = OverrideToolset(OverrideStore(tmp_path))
    result = asyncio.run(ts.expand_memory(_FakeCtx(), line_id=1))
    assert result == "No memory found at journal line #1."


# ---------------------------------------------------------------------------
# render marker
# ---------------------------------------------------------------------------


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
