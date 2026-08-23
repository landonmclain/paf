"""Deterministic tests for the memory organizer index (`_organize.py`).

The index is the **curated** layer of the three-layer design
(journal -> *maintenance* -> index -> render): ``memory-index.json`` holds the
reconciled, deduped, supersession-aware view that ``render_index`` later turns
into ``MEMORY.md``. The consolidation *owner* writes the append-only journal;
the *organizer* reconciles new journal lines into this index.

This file pins the maintenance contracts WITH NO LLM — the organizer's
classification *brain* is a model, but the rules for applying a decision
(NEW / CONFIRM / SUPERSEDE) are deterministic and are what is unit-tested here,
mirroring how ``test_memory_journal.py`` pins the journal contract.

Covered:
1. File wrapper shape (`{version, watermark, entries}`) + atomic write.
2. Entry roundtrip + equality.
3. NEW — mints an entry (monotonic id, subject, source=[line], updated_on=born_on).
4. CONFIRM — grows ``source`` lineage, does NOT bump ``updated_on``.
5. SUPERSEDE — ``born_on`` preserved, ``updated_on`` bumps, ``source`` grows,
   fact/source_ref move to the new current value.
6. Prune — loser removed from the index entirely (no tombstone).
7. Subject grouping — reuse-existing-else-mint, exact-string match.
8. Watermark idempotence — re-running an overlapping tail is a no-op.
9. Atomic crash-safety — write is staging + ``os.replace``.
10. Determinism — reconcile is repeatable / idempotent, round-trip stable.
"""

from __future__ import annotations

import json

import pytest

from paf.capabilities.memory import (
    ConfirmDecision,
    IndexEntry,
    IndexStore,
    NewDecision,
    SupersedeDecision,
)


def _new(line, *, type="fact", subject="topic", fact="a fact",
         born_on="2026-08-17T00:00:00Z", source_ref="run:0") -> NewDecision:
    return NewDecision(
        line_no=line, type=type, subject=subject, fact=fact,
        born_on=born_on, source_ref=source_ref,
    )


def _store(tmp_path) -> IndexStore:
    return IndexStore(tmp_path)


# ---------------------------------------------------------------------------
# File wrapper shape + atomic write
# ---------------------------------------------------------------------------


def test_index_file_wrapper_shape(tmp_path) -> None:
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([_new(1)], end_line=1)
    raw = json.loads((tmp_path / "memory-index.json").read_text(encoding="utf-8"))
    assert raw["version"] == 4
    assert raw["watermark"] == 1
    assert isinstance(raw["next_id"], int)
    assert isinstance(raw["entries"], list)
    assert len(raw["entries"]) == 1
    # Entry carries the full locked schema.
    entry = raw["entries"][0]
    for key in ("id", "type", "subject", "fact", "born_on", "updated_on",
                "source_ref", "source"):
        assert key in entry


def test_index_filename_and_path(tmp_path) -> None:
    s = _store(tmp_path)
    assert s.filename == "memory-index.json"
    assert s.path == str(tmp_path / "memory-index.json")


def test_empty_index_loads_empty(tmp_path) -> None:
    s = _store(tmp_path)
    assert s.load() == []
    assert s.load_watermark() == 0
    assert s.count() == 0


# ---------------------------------------------------------------------------
# Entry roundtrip + equality
# ---------------------------------------------------------------------------


def test_entry_roundtrips_to_dict_and_back() -> None:
    e = IndexEntry(
        id="mem-7", type="preference", subject="storage",
        fact="Landon prefers flat-file, diffable storage",
        born_on="2026-08-17T00:00:00+00:00", updated_on="2026-08-18T00:00:00+00:00",
        source_ref="run-2:3", source=[12, 23],
    )
    assert IndexEntry.from_dict(e.to_dict()) == e


def test_entry_source_list_is_copied_not_aliased() -> None:
    src = [1, 2]
    e = IndexEntry(id="mem-1", type="fact", subject="s", fact="f",
                   born_on="b", updated_on="u", source_ref="r", source=src)
    e.source.append(3)
    # to_dict must not alias the internal list.
    assert src == [1, 2]


# ---------------------------------------------------------------------------
# NEW
# ---------------------------------------------------------------------------


def test_new_mints_entry_with_monotonic_ids(tmp_path) -> None:
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([_new(1, fact="one"), _new(2, fact="two", subject="other")], end_line=2)
    ids = [e.id for e in s.load()]
    assert ids == ["mem-1", "mem-2"]


def test_new_sets_updated_on_to_born_on_and_source_line(tmp_path) -> None:
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([_new(5, fact="flat-file diffable", born_on="2026-08-17T10:30:00Z")], end_line=5)
    e = s.load()[0]
    assert e.source == [5]
    assert e.updated_on == "2026-08-17T10:30:00+00:00"
    assert e.born_on == "2026-08-17T10:30:00+00:00"
    assert e.subject == "topic"
    assert e.source_ref == "run:0"


def test_new_preserves_type_and_source_ref(tmp_path) -> None:
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([_new(1, type="event", fact="deployed v2", source_ref="seed:7")], end_line=1)
    e = s.load()[0]
    assert e.type == "event"
    assert e.source_ref == "seed:7"


def test_new_after_existing_ids_continues_monotonic(tmp_path) -> None:
    """id allocator never reuses an id, even after a prune removes the high id."""
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([_new(1, fact="a"), _new(2, fact="b")], end_line=2)
    # Prune mem-2 (simulate a supersede loser).
    reconcile([SupersedeDecision(entry_id="mem-1", line_no=3, fact="a2",
                                 source_ref="run:3", updated_on="2026-08-18T00:00:00Z",
                                 prune=["mem-2"])], end_line=3)
    assert [e.id for e in s.load()] == ["mem-1"]
    # A brand-new entry must not reuse mem-2.
    reconcile([_new(4, fact="c", subject="fresh")], end_line=4)
    assert [e.id for e in s.load()] == ["mem-1", "mem-3"]


# ---------------------------------------------------------------------------
# CONFIRM
# ---------------------------------------------------------------------------


def test_confirm_grows_source_and_does_not_bump_updated_on(tmp_path) -> None:
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([_new(1, fact="prefers Terminal.app", born_on="2026-08-16T00:00:00Z")], end_line=1)
    reconcile([ConfirmDecision(entry_id="mem-1", line_no=2)], end_line=2)
    e = s.load()[0]
    assert e.source == [1, 2]
    assert e.updated_on == "2026-08-16T00:00:00+00:00"  # unchanged
    assert e.born_on == "2026-08-16T00:00:00+00:00"
    assert e.fact == "prefers Terminal.app"


def test_confirm_unknown_entry_raises(tmp_path) -> None:
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([_new(1)], end_line=1)
    with pytest.raises(ValueError, match="not in index"):
        reconcile([ConfirmDecision(entry_id="mem-99", line_no=2)], end_line=2)


# ---------------------------------------------------------------------------
# SUPERSEDE
# ---------------------------------------------------------------------------


def test_supersede_preserves_born_bumps_updated_grows_source(tmp_path) -> None:
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([_new(1, fact="lives in Boston", born_on="2026-08-10T00:00:00Z")], end_line=1)
    reconcile([
        SupersedeDecision(entry_id="mem-1", line_no=7, fact="lives in Austin",
                          source_ref="run:9", updated_on="2026-08-18T00:00:00Z")
    ], end_line=7)
    e = s.load()[0]
    assert e.id == "mem-1"          # winner keeps its id
    assert e.born_on == "2026-08-10T00:00:00+00:00"  # provenance preserved
    assert e.updated_on == "2026-08-18T00:00:00+00:00"  # recency bumps
    assert e.fact == "lives in Austin"
    assert e.source_ref == "run:9"
    assert e.source == [1, 7]


def test_supersede_updated_on_never_moves_backward(tmp_path) -> None:
    """Field contract: updated_on only ever moves forward."""
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([_new(1, fact="v1", born_on="2026-08-18T00:00:00Z")], end_line=1)
    # A supersede claiming an EARLIER updated_on than the current value is a no-op
    # for the recency axis (an out-of-order journal record).
    reconcile([
        SupersedeDecision(entry_id="mem-1", line_no=5, fact="v2",
                          source_ref="run:2", updated_on="2026-08-17T00:00:00Z")
    ], end_line=5)
    e = s.load()[0]
    assert e.updated_on == "2026-08-18T00:00:00+00:00"  # did NOT move backward


# ---------------------------------------------------------------------------
# Prune (MERGE is a special case of SUPERSEDE for v1)
# ---------------------------------------------------------------------------


def test_prune_removes_loser_entirely_no_tombstone(tmp_path) -> None:
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([_new(1, fact="contact: work email", subject="contact"),
               _new(2, fact="contact: email", subject="contact"),
               _new(3, fact="unrelated", subject="other")], end_line=3)
    # The two contact entries collide; mem-2 is the loser and must vanish.
    reconcile([
        SupersedeDecision(entry_id="mem-1", line_no=4, fact="contact: email",
                          source_ref="run:4", updated_on="2026-08-18T00:00:00Z",
                          prune=["mem-2"])
    ], end_line=4)
    ids = [e.id for e in s.load()]
    assert "mem-2" not in ids
    assert "mem-1" in ids and "mem-3" in ids
    # No tombstone field on disk for the pruned entry.
    raw = json.loads((tmp_path / "memory-index.json").read_text(encoding="utf-8"))
    assert all(e["id"] != "mem-2" for e in raw["entries"])


# ---------------------------------------------------------------------------
# Subject grouping (index-anchored vocabulary: reuse-else-mint, exact match)
# ---------------------------------------------------------------------------


def test_new_reuses_existing_subject_string(tmp_path) -> None:
    """Subject is index-anchored: a record about an existing subject reuses the
    exact string rather than minting a synonym, so grouping is by exact match."""
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([_new(1, subject="storage", fact="flat-file storage")], end_line=1)
    reconcile([_new(2, subject="storage", fact="atomic writes")], end_line=2)
    subjects = [e.subject for e in s.load()]
    assert subjects == ["storage", "storage"]


# ---------------------------------------------------------------------------
# Watermark idempotence
# ---------------------------------------------------------------------------


def test_overlapping_tail_rerun_is_idempotent(tmp_path) -> None:
    """Re-running reconcile over lines already processed does not double-append
    ``source`` nor move the watermark backward."""
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([_new(1, fact="prefers X", born_on="2026-08-17T00:00:00Z")], end_line=1)
    reconcile([ConfirmDecision(entry_id="mem-1", line_no=2)], end_line=2)
    e_before = s.load()[0]
    assert e_before.source == [1, 2]

    # Re-run with overlapping decisions for lines 1-3 (same confirmer again).
    reconcile([
        _new(1, fact="prefers X", born_on="2026-08-17T00:00:00Z"),
        ConfirmDecision(entry_id="mem-1", line_no=2),
    ], end_line=3)
    e_after = s.load()[0]
    # source is deduped (line 2 not appended again); watermark went to 3.
    assert e_after.source == [1, 2]
    assert s.load_watermark() == 3


def test_watermark_never_moves_backward(tmp_path) -> None:
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([_new(1)], end_line=5)
    assert s.load_watermark() == 5
    # A later run with an end_line behind the current cursor is a no-op.
    reconcile([], end_line=2)
    assert s.load_watermark() == 5


# ---------------------------------------------------------------------------
# Determinism / atomicity
# ---------------------------------------------------------------------------


def test_reconcile_is_deterministic_and_roundtrip_stable(tmp_path) -> None:
    """reconcile over the same decisions reproduces the same bytes."""
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([
        _new(1, subject="storage", fact="flat-file"),
        _new(2, subject="tools", fact="prefers CLI"),
        ConfirmDecision(entry_id="mem-1", line_no=3),
    ], end_line=3)
    first = (tmp_path / "memory-index.json").read_text(encoding="utf-8")

    # load => save must be byte-stable (no drift on re-persist).
    s2 = _store(tmp_path)
    entries = s2.load()
    wm = s2.load_watermark()
    s2._save(entries, wm)
    assert (tmp_path / "memory-index.json").read_text(encoding="utf-8") == first


def test_write_uses_atomic_staging(tmp_path) -> None:
    """The index is written via staging + os.replace (crash-safe: old or new)."""
    s = _store(tmp_path)
    reconcile = _make_reconcile(s)
    reconcile([_new(1)], end_line=1)
    assert not (tmp_path / ".memory-index.json.staging").exists()
    assert (tmp_path / "memory-index.json").is_file()
    # Value present post-write.
    assert s.load()[0].fact == "a fact"


def _make_reconcile(store: IndexStore):
    """Bound ``reconcile`` to a store for concise tests."""
    from paf.capabilities.memory._organize import reconcile as _reconcile

    def reconcile(decisions, *, end_line=None, watermark=None):
        return _reconcile(store, [], decisions, end_line=end_line, watermark=watermark)

    return reconcile


def test_index_new_rejects_composite_type_with_retry_guidance(tmp_path) -> None:
    """The organizer's `index_new` rejects `type/subject` compounds via ModelRetry
    (not a bare ValueError), teaching the model the type/subject split."""
    import asyncio

    from pydantic_ai import ModelRetry

    from paf.capabilities.memory._organize import _build_index_toolset

    toolset = _build_index_toolset()(IndexStore(tmp_path))
    # The prod bug: model emitted type="fact/paf" (topic folded into type).
    with pytest.raises(ModelRetry) as ei:
        asyncio.run(toolset.add_new(
            {}, line_no=27, type="fact/paf", subject="paf",
            fact="paf is a project", born_on="2026-08-22T00:00:00+00:00",
            source_ref="run:0",
        ))
    msg = str(ei.value)
    # Message must name the allowed set AND the type/subject correction.
    assert "expected one of" in msg and "fact" in msg
    assert "subject" in msg and "fact/paf" in msg


def test_index_new_accepts_bare_type_and_subject(tmp_path) -> None:
    """`index_new` accepts a valid bare `type` with the topic in `subject`."""
    import asyncio

    from paf.capabilities.memory._organize import _build_index_toolset

    toolset = _build_index_toolset()(IndexStore(tmp_path))
    out = asyncio.run(toolset.add_new(
        {}, line_no=27, type="fact", subject="paf",
        fact="paf is a project", born_on="2026-08-22T00:00:00+00:00",
        source_ref="run:0",
    ))
    assert out["status"] == "updated"
    entry = toolset._get_stage().entries[0]
    assert entry.type == "fact"
    assert entry.subject == "paf"


def test_index_result_returns_live_staged_view(tmp_path) -> None:
    """The tool result returns the last action AND a live view of every staged
    entry (not just a count), so the organizer can see — on each call — what it
    has already added this run and avoid re-adding the same fact."""
    import asyncio

    from paf.capabilities.memory._organize import _build_index_toolset

    toolset = _build_index_toolset()(IndexStore(tmp_path))
    out1 = asyncio.run(toolset.add_new(
        {}, line_no=40, type="fact", subject="paf",
        fact="paf is built on pydantic-ai",
        born_on="2026-08-22T00:00:00+00:00", source_ref="run:0",
    ))
    diff1 = out1["diff"]
    assert diff1 is not None
    # First call: names the new id, the type/subject, the line, and shows the entry.
    assert "index_new -> added mem-1 [fact/paf]" in diff1
    assert "line 40" in diff1
    assert "mem-1 [fact/paf] paf is built on pydantic-ai" in diff1

    out2 = asyncio.run(toolset.add_new(
        {}, line_no=41, type="fact", subject="paf",
        fact="paf has a browser_inspect_url tool",
        born_on="2026-08-22T00:00:00+00:00", source_ref="run:0",
    ))
    diff2 = out2["diff"]
    assert diff2 is not None
    # Second call: still shows BOTH staged entries (the live dedup surface).
    assert "index_new -> added mem-2 [fact/paf]" in diff2
    assert "mem-1 [fact/paf] paf is built on pydantic-ai" in diff2
    assert "mem-2 [fact/paf] paf has a browser_inspect_url" in diff2
    assert "staged entries=2" in diff2


def test_index_confirm_diff_names_target(tmp_path) -> None:
    """`index_confirm` diff names the confirmed entry id + source line."""
    import asyncio

    from paf.capabilities.memory._organize import _build_index_toolset

    toolset = _build_index_toolset()(IndexStore(tmp_path))
    asyncio.run(toolset.add_new(
        {}, line_no=40, type="fact", subject="paf",
        fact="paf is built on pydantic-ai",
        born_on="2026-08-22T00:00:00+00:00", source_ref="run:0",
    ))
    out = asyncio.run(toolset.confirm({}, entry_id="mem-1", line_no=41))
    diff = out["diff"]
    assert diff is not None
    assert "index_confirm -> mem-1 now sourced from line 41" in diff


def test_index_new_rejects_near_dup_and_points_at_entry(tmp_path) -> None:
    """Re-adding a rephrased version of an already-staged fact raises ModelRetry
    (Option B) naming the matching entry id and steering to confirm/supersede."""
    import asyncio

    from pydantic_ai import ModelRetry

    from paf.capabilities.memory._organize import _build_index_toolset

    toolset = _build_index_toolset()(IndexStore(tmp_path))
    asyncio.run(toolset.add_new(
        {}, line_no=40, type="fact", subject="paf",
        fact="paf is built on pydantic-ai",
        born_on="2026-08-22T00:00:00+00:00", source_ref="run:0",
    ))
    with pytest.raises(ModelRetry) as ei:
        asyncio.run(toolset.add_new(
            {}, line_no=41, type="fact", subject="paf",
            fact="paf is built on the pydantic-ai stack",
            born_on="2026-08-22T00:00:00+00:00", source_ref="run:0",
        ))
    msg = str(ei.value)
    assert "mem-1" in msg
    assert "index_confirm" in msg and "index_supersede" in msg
    # Nothing was added for line 41 — still just the one entry.
    assert [e.id for e in toolset._get_stage().entries] == ["mem-1"]


def test_index_new_allows_genuinely_distinct_fact(tmp_path) -> None:
    """A fact that is not a near-dup passes the guard (no false positive)."""
    import asyncio

    from paf.capabilities.memory._organize import _build_index_toolset

    toolset = _build_index_toolset()(IndexStore(tmp_path))
    asyncio.run(toolset.add_new(
        {}, line_no=40, type="fact", subject="paf",
        fact="paf is built on pydantic-ai",
        born_on="2026-08-22T00:00:00+00:00", source_ref="run:0",
    ))
    out = asyncio.run(toolset.add_new(
        {}, line_no=41, type="fact", subject="paf",
        fact="paf has a browser_inspect_url tool",
        born_on="2026-08-22T00:00:00+00:00", source_ref="run:0",
    ))
    assert out["status"] == "updated"
    assert [e.id for e in toolset._get_stage().entries] == ["mem-1", "mem-2"]


# ---------------------------------------------------------------------------
# Full-index dedup guard (pass 1) — catches near-dups of EXISTING disk entries,
# not just facts staged this run. Same type+subject scope, max(Jaccard,
# containment) similarity so rich-vs-short repeats are caught too.
# ---------------------------------------------------------------------------


def _commit(toolset, end_line: int) -> None:
    """Finalize + commit a toolset's staged index so a new toolset reloads it."""
    import asyncio

    asyncio.run(toolset.finalize_index({}, end_line))


def test_index_new_rejects_dup_of_existing_disk_entry(tmp_path) -> None:
    """A re-phrase of a fact ALREADY in the on-disk index raises ModelRetry
    naming that entry — the guard now scans the full index, not just staged."""
    import asyncio

    from pydantic_ai import ModelRetry

    from paf.capabilities.memory._organize import _build_index_toolset

    ts1 = _build_index_toolset()(IndexStore(tmp_path))
    asyncio.run(ts1.add_new(
        {}, line_no=40, type="fact", subject="paf",
        fact="paf is built on pydantic-ai",
        born_on="2026-08-22T00:00:00+00:00", source_ref="run:0",
    ))
    _commit(ts1, 40)  # persist mem-1 to disk

    # A brand-new organizer run over the SAME store loads mem-1 from disk.
    ts2 = _build_index_toolset()(IndexStore(tmp_path))
    assert [e.id for e in ts2._get_stage().entries] == ["mem-1"]
    with pytest.raises(ModelRetry) as ei:
        asyncio.run(ts2.add_new(
            {}, line_no=41, type="fact", subject="paf",
            fact="paf is built on the pydantic-ai stack",
            born_on="2026-08-22T00:00:00+00:00", source_ref="run:1",
        ))
    msg = str(ei.value)
    assert "mem-1" in msg
    assert "index_confirm" in msg and "index_supersede" in msg
    assert [e.id for e in ts2._get_stage().entries] == ["mem-1"]


def test_index_new_containment_catches_rich_vs_short_repeat(tmp_path) -> None:
    """A short near-repeat of a RICH existing entry is caught by the containment
    signal even though plain Jaccard over the full facts is below threshold."""
    import asyncio

    from pydantic_ai import ModelRetry

    from paf.capabilities.memory._organize import _build_index_toolset

    ts1 = _build_index_toolset()(IndexStore(tmp_path))
    asyncio.run(ts1.add_new(
        {}, line_no=40, type="fact", subject="paf",
        fact=("paf = Pydantic Agent Framework, a project Landon is working on "
              "for a GitHub portfolio; an opinionated framework built on the "
              "pydantic-ai stack, import name `paf`."),
        born_on="2026-08-22T00:00:00+00:00", source_ref="run:0",
    ))
    _commit(ts1, 40)

    ts2 = _build_index_toolset()(IndexStore(tmp_path))
    with pytest.raises(ModelRetry) as ei:
        asyncio.run(ts2.add_new(
            {}, line_no=41, type="fact", subject="paf",
            fact="User's project paf is built on the pydantic-ai stack.",
            born_on="2026-08-22T00:00:00+00:00", source_ref="run:1",
        ))
    # Rich entry's tokens mostly contain the short repeat's tokens.
    assert "mem-1" in str(ei.value)


def test_index_new_cross_topic_same_words_not_blocked(tmp_path) -> None:
    """Same few tokens but a DIFFERENT type+subject must not block — the scope
    to same type+subject removes cross-topic false positives."""
    import asyncio

    from paf.capabilities.memory._organize import _build_index_toolset

    ts1 = _build_index_toolset()(IndexStore(tmp_path))
    asyncio.run(ts1.add_new(
        {}, line_no=40, type="fact", subject="paf",
        fact="paf is built on pydantic-ai",
        born_on="2026-08-22T00:00:00+00:00", source_ref="run:0",
    ))
    _commit(ts1, 40)

    ts2 = _build_index_toolset()(IndexStore(tmp_path))
    out = asyncio.run(ts2.add_new(
        {}, line_no=41, type="fact", subject="Hope Technologies",
        fact="The Hope site is built with the Elementor page builder",
        born_on="2026-08-22T00:00:00+00:00", source_ref="run:1",
    ))
    assert out["status"] == "updated"
    assert [e.id for e in ts2._get_stage().entries] == ["mem-1", "mem-2"]


# ---------------------------------------------------------------------------
# Consolidation pass (pass 2) — detects existing near-dup clusters, no-ops on clean
# ---------------------------------------------------------------------------


def test_find_consolidation_candidates_finds_dup_clusters(tmp_path) -> None:
    """Near-duplicate pairs (same type+subject) group into clusters; singletons
    and cross-topic entries do not. Builds the index via the low-level
    ``Stage``/``NewDecision`` path (bypassing the ``index_new`` guard, which
    correctly rejects a second near-dup at add time)."""
    from paf.capabilities.memory._organize import (
        IndexStore,
        NewDecision,
        Stage,
        _find_consolidation_candidates,
    )

    store = IndexStore(tmp_path)
    stage = Stage(store)
    for line, subj, fact in [
        (1, "paf", "paf is built on pydantic-ai"),
        (2, "paf", "paf is built on the pydantic-ai stack"),
        (3, "paf", "paf has a browser_inspect_url tool"),
        (4, "Hope Technologies", "The Hope site is built with Elementor"),
        (5, "Hope Technologies", "The Hope site uses Elementor (WordPress)"),
    ]:
        stage.apply(NewDecision(
            line_no=line, type="fact", subject=subj, fact=fact,
            born_on="2026-08-22T00:00:00+00:00", source_ref="run:0",
        ))
    stage.finalize(5)

    clusters = _find_consolidation_candidates(store.load())
    # paf-built pair {mem-1, mem-2} + hope-elementor pair {mem-4, mem-5};
    # browser tool (mem-3) is a singleton, so it appears in no cluster.
    member_sets = sorted(sorted(e.id for e in c) for c in clusters)
    assert member_sets == [
        ["mem-1", "mem-2"],
        ["mem-4", "mem-5"],
    ]


def test_run_consolidation_no_changes_when_clean(tmp_path) -> None:
    """A non-duplicate index yields 'no changes' with zero LLM interaction."""
    import asyncio

    from paf.capabilities.memory._organize import (
        _build_index_toolset,
        run_consolidation,
    )

    ts = _build_index_toolset()(IndexStore(tmp_path))
    for line, subj, fact in [
        (1, "paf", "paf is built on pydantic-ai"),
        (2, "paf", "paf has a browser_inspect_url tool"),
    ]:
        asyncio.run(ts.add_new(
            {}, line_no=line, type="fact", subject=subj,
            fact=fact, born_on="2026-08-22T00:00:00+00:00", source_ref="run:0",
        ))
    _commit(ts, 2)
    assert asyncio.run(run_consolidation(IndexStore(tmp_path))) == "no changes"


