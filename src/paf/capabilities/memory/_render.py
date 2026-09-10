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

import os
from pathlib import Path

from ._journal import JournalStore, MemoryRecord
from ._overrides import OverrideState

# Per-record separator between fact / date / source_ref in the md line.
_SEP = " — "

# Marker appended to a fact that has a PENDING archive suggestion from the
# assistant (awaiting a reviewer verdict). Distinct phrasing — "archive pending
# review" — so the answerer reads it as a *proposed removal*, not a general
# note.
_PENDING_MARKER = " *(archive pending review)*"

# Fixed render order for the 3-tier ladder; tiers not present are omitted.
_TIER_ORDER = ("t1", "t2", "t3")

# Bounded-presentation caps (per tier) for the LIVE index render — Gap A of
# Global-Durable-Memory-Maintenance. Each tier is capped to the most-recent N
# winners so the injected MEMORY.md stays small and current instead of a growing
# wall that a small model misreads as corruption (trace cfe9030e,
# web-demo_assistant): t1 (stable identity/prefs) 75, t2 (goals/current) 25,
# t3 (ephemeral) 15. Old facts fall out of the injected view but stay in the
# journal/index — never lost. (t1 was previously unbounded; capping it is what
# keeps a reviewer's usual accumulation from bloating every request.)
_TIER_CAPS: dict[str, int | None] = {"t1": 75, "t2": 25, "t3": 15}


def _group_by_tier(records: list[MemoryRecord]) -> dict[str, list[MemoryRecord]]:
    """Group records by `tier`, each group sorted (recency + identity-pin).

    Within a tier, sort by ``born_on`` descending (most-recent-first), but
    ``identity``-type records are pinned ABOVE all other types in the same tier
    (so "who the user is" stays at the top of ``## t1`` regardless of age).
    Ties keep append order (stable sort).
    """
    groups: dict[str, list[MemoryRecord]] = {}
    for rec in records:
        groups.setdefault(rec.tier, []).append(rec)
    for recs in groups.values():
        # identity-pin: identity-type records first (top of the tier section),
        # then everything else; recency (born_on desc) within each sub-group.
        identity = [r for r in recs if r.type == "identity"]
        others = [r for r in recs if r.type != "identity"]
        identity.sort(key=lambda r: r.born_on, reverse=True)
        others.sort(key=lambda r: r.born_on, reverse=True)
        recs[:] = identity + others
    return groups


def _apply_overrides(
    pairs: list[tuple[int, MemoryRecord]],
    overrides: dict[int, OverrideState] | None,
) -> tuple[list[tuple[int, MemoryRecord]], set[int]]:
    """Filter / re-tier / mark ``(journal_line, record)`` pairs per the sidecar.

    ``overrides`` is the folded per-target map (``None``/empty -> no-op). Returns
    ``(kept_pairs, pending_lines)`` where ``pending_lines`` arethe journal
    line ids with an assistant archive awaiting review (those lines render with a
    ``*(archive pending review)*`` marker). Applied archives are dropped; applied tier
    changes rebuild the record at the new tier.
    """
    if not overrides:
        return pairs, set()
    kept: list[tuple[int, MemoryRecord]] = []
    pending: set[int] = set()
    for line, rec in pairs:
        st = overrides.get(line)
        if st is None:
            kept.append((line, rec))
            continue
        if st.is_applied_archive:
            continue  # reviewer archived — drop from injection
        if st.is_applied_tier and st.tier and st.tier != rec.tier:
            # reviewer demoted — rebuildthe record at the new tier
            rec = MemoryRecord(
                type=rec.type,
                fact=rec.fact,
                tier=st.tier,
                born_on=rec.born_on,
                source_ref=rec.source_ref,
            )
        if st.is_pending:
            pending.add(line)
        kept.append((line, rec))
    return kept, pending


def render_journal_memory(
    store: JournalStore | list[MemoryRecord],
    overrides: dict[int, OverrideState] | None = None,
) -> str:
    """Render the journal into a MEMORY.md string (pure, deterministic).

    ``store`` may be a :class:`JournalStore` (reads all records) or an explicit
    list of :class:`MemoryRecord` (for testing without touching disk). Sections
    (``## t1`` / ``## t2`` / ``## t3``) appear only for tiers that have records,
    in fixed ladder order; within a tier records sort by ``born_on`` descending
    (most-recent-first). Each line: ``- <fact> — <born_on> — <source_ref>``.

    ``overrides`` (optional, folded sidecar map) is honored deterministically:
    applied archives drop the fact, applied tier-changes move it under a new
    tier, and pending assistant archives render with a ``*(archive pending review)*``
    marker so the subject can see its proposal is awaiting review.

    Unbound: every record is emitted.
    """
    provenance: dict[int, str] = {}
    if isinstance(store, JournalStore):
        pairs = store.read_all()
        provenance = {id(r): f"#{ln}" for ln, r in pairs}
    else:
        pairs = [(id(r), r) for r in store]

    kept, pending = _apply_overrides(pairs, overrides)
    groups = _group_by_tier([r for _, r in kept])

    sections: list[str] = []
    for tier in _TIER_ORDER:
        recs = groups.get(tier)
        if not recs:
            continue
        lines = []
        for r in recs:
            prov = provenance.get(id(r)) or r.source_ref
            marker = _PENDING_MARKER if id(r) in pending else ""
            lines.append(
                f"- {_encode(r.fact)}{_SEP}{_date_only(r.born_on)}{_SEP}{prov}{marker}"
                if r.born_on
                else f"- {_encode(r.fact)}{_SEP}{prov}{marker}"
            )
        sections.append(f"## {tier}\n" + "\n".join(lines))

    return "\n\n".join(sections) + ("\n" if sections else "")


def _select_index_winners(
    store: JournalStore, index
) -> list[tuple[int, MemoryRecord]]:
    """Return the (journal-line-id, record) pairs that are surviving facts.

    ``index`` is a :class:`._index.MemoryIndex` (loaded, not its raw dict). For
    each entry we keep the record at ``entry["current_line"]`` — the single
    live copy of that fact — and drop every line in ``deduped_lines``. The
    journal is the source of the fact text; the index is the verdict on which
    line is the live one. The returned ``line`` is the stable journal id used
    as the rendered provenance (and the address overrides/`find_memory` key on).
    """
    records = [r for _, r in store.read_all()]
    winners: list[tuple[int, MemoryRecord]] = []
    for entry in index.entries:
        line = int(entry["current_line"])
        # current_line is a 1-based journal line number; records are 0-indexed.
        if 1 <= line <= len(records):
            winners.append((line, records[line - 1]))
    return winners


def render_index_memory(
    store: JournalStore,
    index,
    overrides: dict[int, OverrideState] | None = None,
    tier_caps: dict[str, int | None] | None = None,
) -> str:
    """Render MEMORY.md **from the derived index** (surviving facts only).

    Groups the index winners by tier (t1/t2/t3 ladder), sorted by ``born_on``
    descending within a tier — the same deterministic form as the raw render,
    but emitting only the deduped survivors instead of every journal record.
    Dedup is meaningless if it doesn't shrink what the answerer reads, and this
    is the render the live write path uses when an index exists.

    **Bounded (Gap A):** each tier is capped to the most-recent N winners
    (``tier_caps``, default :data:`_TIER_CAPS`) — t1 capped, t2 capped, t3
    capped. Winners are already deduped and sorted by recency, so the cap keeps
    the newest facts and lets old ephemeral/goal entries fall out of the
    injected view (they remain reachable in the journal / index, never lost).

    ``overrides`` (optional, folded sidecar map) is honored deterministically
    before grouping: applied archives drop the fact, applied tier-changes move
    it under a new tier, and pending assistant archives render with a
    ``*(archive pending review)*`` marker.
    """
    caps = tier_caps if tier_caps is not None else _TIER_CAPS
    winners = _select_index_winners(store, index)
    kept, pending = _apply_overrides(winners, overrides)
    groups = _group_by_tier([r for _, r in kept])
    # Bound each already-recency-sorted group to its tier cap.
    groups = {
        tier: recs[: caps[tier]] if caps.get(tier) else recs
        for tier, recs in groups.items()
    }

    # Map a record back to its journal line id (the rendered provenance).
    line_of = {id(r): ln for ln, r in kept}

    sections: list[str] = []
    for tier in _TIER_ORDER:
        recs = groups.get(tier)
        if not recs:
            continue
        lines = []
        for r in recs:
            ln = line_of.get(id(r))
            prov = f"#{ln}" if ln is not None else r.source_ref
            marker = _PENDING_MARKER if ln in pending else ""
            lines.append(
                f"- {_encode(r.fact)}{_SEP}{_date_only(r.born_on)}{_SEP}{prov}{marker}"
                if r.born_on
                else f"- {_encode(r.fact)}{_SEP}{prov}{marker}"
            )
        sections.append(f"## {tier}\n" + "\n".join(lines))

    return "\n\n".join(sections) + ("\n" if sections else "")


def _resolve_tier_caps(agent_name: str | None) -> dict[str, int | None]:
    """Resolve the per-tier injection caps for an agent from its spec.

    The render owns its own caps — Option B: it reads ``memory.tier_caps``
    from the agent's spec directly (via ``paf.specs.resolve_memory_tier_caps``),
    merging any partial overrides over the built-in :data:`_TIER_CAPS`. When
    ``agent_name`` is ``None``, or the spec is absent / declares no
    ``tier_caps``, the built-in caps apply unchanged. Neither the runtime
    Session nor the background queue carries ``tier_caps`` anywhere.
    """
    if not agent_name:
        return dict(_TIER_CAPS)
    from paf.specs import resolve_memory_tier_caps

    overrides = resolve_memory_tier_caps(agent_name)
    if not overrides:
        return dict(_TIER_CAPS)
    caps = dict(_TIER_CAPS)
    caps.update(overrides)
    return caps


def write_render(
    store: JournalStore | list[MemoryRecord],
    md_path: str,
    *,
    agent_name: str | None = None,
) -> str:
    """Render and atomically write ``md_path`` (e.g. ``MEMORY.md``).

    **Index-aware:** when ``store`` is a :class:`JournalStore` and a derived
    index (``memory-index.json``) exists beside ``md_path``, the render comes
    from the index — only the surviving (deduped-winner) facts are emitted, so
    ``MEMORY.md`` agrees with ``memory-index.json`` and the assistant reads the
    deduplicated notebook. If no index exists (pure unit-test path, the
    benchmark's fresh directory, or the first render before any fold), it falls
    back to the raw journal render — preserving the benchmark's measured
    contract unchanged.

    ``agent_name`` (optional) lets the render resolve its own per-tier
    injection caps from the agent's spec (:func:`_resolve_tier_caps`), falling
    back to :data:`_TIER_CAPS`. Absent, it uses the built-in caps. Bounding
    applies only to the index-aware render.

    Returns the rendered markdown. Uses the memory package's atomic writer so a
    crash mid-write never leaves a torn notebook.
    """
    from paf.capabilities._storage._atomic import (
        write_atomic,  # local import keeps top of module import-light
    )

    from ._overrides import (
        load_overrides,  # local import keeps top of module import-light
    )

    overrides = load_overrides(str(Path(md_path).parent))
    tier_caps = _resolve_tier_caps(agent_name)

    if isinstance(store, JournalStore):
        index = _load_index_beside(md_path)
        if index is not None:
            content = render_index_memory(
                store, index, overrides=overrides, tier_caps=tier_caps
            )
        else:
            content = render_journal_memory(store, overrides=overrides)
    else:
        content = render_journal_memory(store, overrides=overrides)
    write_atomic(md_path, content)
    return content


def read_render(directory: str, md_name: str = "MEMORY.md") -> str:
    """Return the current rendered MEMORY.md for ``directory`` (full, unbounded).

    Read path for the **memory reviewer**: unlike the injection seam
    (``JournalViewStore`` / ``_view.read``), which trims to the message-bound
    max_chars, this returns the *complete* rendered notebook so the reviewer
    can curate every surviving fact (incl. tiers the bounded injection drops).

    If ``MEMORY.md`` is missing (no render yet), it is **computed** from the
    journal/index/overrides just as ``write_render`` would — so the reviewer
    never depends on the file having been materialized by a prior pass. Read-only.
    """
    from ._overrides import load_overrides  # local import keeps top import-light

    md_path = os.path.join(directory, md_name)
    if os.path.exists(md_path):
        with open(md_path, encoding="utf-8") as fh:
            return fh.read()
    # No materialized render yet -> compute it (index-aware, honors overrides).
    journal = JournalStore(directory)
    overrides = load_overrides(directory)
    index = _load_index_beside(md_path)
    if index is not None:
        return render_index_memory(journal, index, overrides=overrides)
    return render_journal_memory(journal, overrides=overrides)


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
