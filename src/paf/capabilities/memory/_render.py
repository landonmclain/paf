"""Deterministic render: journal -> MEMORY.md (the presentation view).

The three-layer memory design: **journal** (append-only fact stream, M2) ->
**[render]** (this module) -> **MEMORY.md** (what the answerer reads). The
render is the "database" step: it takes the journal's write-once records and
produces the organized markdown notebook — ordering by **tier then recency**
(most-recent-first within a tier), and attaching date + source per memory. It
is **pure and deterministic**: no model, no state, no I/O beyond reading the
journal and writing the file. LLM curation may come later; "if it doesn't have
to be [LLM], that is better" — and it doesn't here.

Format (LOCKED with Landon, 2026-08-25): the old type-grouped render (one
``## <type>`` section per memory type) is **thrown out in favor of the
3-tier ladder** — t1 (stable "who they are", supersede-only), t2 (goals /
current work), t3 (ephemeral, recency tail). One ``## t1``/``## t2``/``## t3``
section per tier present, records sorted by ``born_on`` **descending** within
each tier (most-recent-first), each line carrying **at least fact, date
(born_on), and source (source_ref)**:

    ## t1
    - user prefers flat-file storage — 2026-08-17 — seed-a1b2:12

Type grouping is gone — type remains a journal search tag only, not a render
axis. The render is deliberately **unbound** (emits every record) so a
benchmark can compare tier-ordered vs. the flat full dump; bounding/capping is
a future, separate concern.

Determinism: tiers iterate in fixed order (t1 -> t2 -> t3); records within a
tier sort by born_on descending (lexical ISO == chronological, reversed for
most-recent-first); ties keep append order. The same journal always renders
the same bytes.
"""

from __future__ import annotations

from pathlib import Path

from ._journal import JournalStore, MemoryRecord

# Per-record separator between fact / date / source_ref in the md line.
_SEP = " — "

# Fixed render order for the 3-tier ladder; tiers not present are omitted.
_TIER_ORDER = ("t1", "t2", "t3")

# Bounded-presentation caps (per tier) for the LIVE index render — Gap A of
# Global-Durable-Memory-Maintenance. t1 (stable "who they are") is stable
# identity/preference and stays UNBOUNDED; t2 (goals/current) and t3
# (ephemeral) are capped to the most-recent N winners so the injected
# MEMORY.md stays small and current instead of a growing wall that a small
# model misreads as corruption (trace cfe9030e, web-demo_assistant).
_TIER_CAPS: dict[str, int | None] = {"t1": None, "t2": 25, "t3": 15}


def _group_by_tier(records: list[MemoryRecord]) -> dict[str, list[MemoryRecord]]:
    """Group records by `tier`, each group sorted by `born_on` descending.

    Most-recent-first within a tier (newest at top). Ties keep append order
    (stable sort). Iterates in deterministic append order here; final sorted
    emission is in :func:`render_journal_memory`.
    """
    groups: dict[str, list[MemoryRecord]] = {}
    for rec in records:
        groups.setdefault(rec.tier, []).append(rec)
    for recs in groups.values():
        recs.sort(key=lambda r: r.born_on, reverse=True)
    return groups


def render_journal_memory(
    store: JournalStore | list[MemoryRecord],
) -> str:
    """Render the journal into a MEMORY.md string (pure, deterministic).

    ``store`` may be a :class:`JournalStore` (reads all records) or an explicit
    list of :class:`MemoryRecord` (for testing without touching disk). Sections
    (``## t1`` / ``## t2`` / ``## t3``) appear only for tiers that have records,
    in fixed ladder order; within a tier records sort by ``born_on`` descending
    (most-recent-first). Each line: ``- <fact> — <born_on> — <source_ref>``.
    Unbound: every record is emitted.
    """
    records: list[MemoryRecord]
    if isinstance(store, JournalStore):
        records = [r for _, r in store.read_all()]
    else:
        records = list(store)

    groups = _group_by_tier(records)

    sections: list[str] = []
    for tier in _TIER_ORDER:
        recs = groups.get(tier)
        if not recs:
            continue
        lines = [
            f"- {_encode(r.fact)}{_SEP}{_date_only(r.born_on)}{_SEP}{r.source_ref}"
            if r.born_on
            else f"- {_encode(r.fact)}{_SEP}{r.source_ref}"
            for r in recs
        ]
        sections.append(f"## {tier}\n" + "\n".join(lines))

    return "\n\n".join(sections) + ("\n" if sections else "")


def _select_index_winners(
    store: JournalStore, index
) -> list[MemoryRecord]:
    """Return the journal records that are surviving facts per the index.

    ``index`` is a :class:`._index.MemoryIndex` (loaded, not its raw dict). For
    each entry we keep the record at ``entry["current_line"]`` — the single
    live copy of that fact — and drop every line in ``deduped_lines``. The
    journal is the source of the fact text; the index is the verdict on which
    line is the live one.
    """
    records = [r for _, r in store.read_all()]
    winners: list[MemoryRecord] = []
    for entry in index.entries:
        line = int(entry["current_line"])
        # current_line is a 1-based journal line number; records are 0-indexed.
        if 1 <= line <= len(records):
            winners.append(records[line - 1])
    return winners


def render_index_memory(store: JournalStore, index) -> str:
    """Render MEMORY.md **from the derived index** (surviving facts only).

    Groups the index winners by tier (t1/t2/t3 ladder), sorted by ``born_on``
    descending within a tier — the same deterministic form as the raw render,
    but emitting only the deduped survivors instead of every journal record.
    Dedup is meaningless if it doesn't shrink what the answerer reads, and this
    is the render the live write path uses when an index exists.

    **Bounded (Gap A):** each tier is capped to the most-recent N winners
    (:data:`_TIER_CAPS`) — t1 unbounded, t2 capped, t3 capped. Winners are
    already deduped and sorted by recency, so the cap keeps the newest facts
    and lets old ephemeral/goal entries fall out of the injected view (they
    remain reachable in the journal / index, never lost).
    """
    winners = _select_index_winners(store, index)
    groups = _group_by_tier(winners)
    # Bound each already-recency-sorted group to its tier cap (t1 unbounded).
    groups = {
        tier: recs[:_TIER_CAPS[tier]] if _TIER_CAPS.get(tier) else recs
        for tier, recs in groups.items()
    }

    sections: list[str] = []
    for tier in _TIER_ORDER:
        recs = groups.get(tier)
        if not recs:
            continue
        lines = [
            f"- {_encode(r.fact)}{_SEP}{_date_only(r.born_on)}{_SEP}{r.source_ref}"
            if r.born_on
            else f"- {_encode(r.fact)}{_SEP}{r.source_ref}"
            for r in recs
        ]
        sections.append(f"## {tier}\n" + "\n".join(lines))

    return "\n\n".join(sections) + ("\n" if sections else "")


def write_render(store: JournalStore | list[MemoryRecord], md_path: str) -> str:
    """Render and atomically write ``md_path`` (e.g. ``MEMORY.md``).

    **Index-aware:** when ``store`` is a :class:`JournalStore` and a derived
    index (``memory-index.json``) exists beside ``md_path``, the render comes
    from the index — only the surviving (deduped-winner) facts are emitted, so
    ``MEMORY.md`` agrees with ``memory-index.json`` and the assistant reads the
    deduplicated notebook. If no index exists (pure unit-test path, the
    benchmark's fresh directory, or the first render before any fold), it falls
    back to the raw journal render — preserving the benchmark's measured
    contract unchanged.

    Returns the rendered markdown. Uses the memory package's atomic writer so a
    crash mid-write never leaves a torn notebook.
    """
    from ._atomic import write_atomic  # local import keeps top of module import-light

    if isinstance(store, JournalStore):
        index = _load_index_beside(md_path)
        if index is not None:
            content = render_index_memory(store, index)
        else:
            content = render_journal_memory(store)
    else:
        content = render_journal_memory(store)
    write_atomic(md_path, content)
    return content


def _load_index_beside(md_path: str):
    """Load ``memory-index.json`` beside ``md_path``, or ``None`` if absent.

    A corrupt/unreadable index is treated as absent (the raw render wins; the
    next fold regenerates the index). Reading it here means the live write path
    gets index-driven rendering with no signature change at its call sites.
    """
    try:
        from ._index import MemoryIndex

        index_path = str(Path(md_path).with_name("memory-index.json"))
        if Path(index_path).exists():
            return MemoryIndex.load(str(Path(md_path).parent))
    except Exception:
        return None
    return None


def _encode(text: str) -> str:
    """Sanitize a fact for a single md line: replace internal newlines/sep."""
    # Collapse newlines so a multi-line fact never breaks the one-per-line shape.
    return " ".join(text.split())


def _date_only(born_on: str) -> str:
    """Reduce a ``born_on`` ISO timestamp to calendar date for display.

    Presentation-only: the journal/index keep the full ``born_on`` timestamp
    (that is the sort key); MEMORY.md shows just ``YYYY-MM-DD`` because the
    time-of-day is noise in the notebook view (every production record pins
    ``T00:00:00`` anyway — the source turns carry only dates). Ordering is
    unaffected: nothing downstream parses this rendered string.
    """
    if not born_on:
        return born_on
    return born_on.split("T", 1)[0]
