"""ARCHIVED index renderer (curated-index view of MEMORY.md).

This is the *curated* counterpart to the live ``paf.capabilities.memory._render``
``render_journal_memory``: it presents the deduped/superseded entries the
memory-organizer maintained in ``memory-index.json`` rather than the raw,
ever-growing journal.

**Status: ARCHIVED 2026-08-26.** The curated index — and the organizer that
maintained it, the consolidation second pass, and this renderer — were
**unhooked from the live compaction path** (2026-08-25) and the index was
removed from the memory pipeline entirely. The live path is now the append-only
journal + deterministic ``write_render`` (tier t1–t3). This module is kept
*whole* in the archive so the idea can be revived later without rewriting it;
the tests in ``tests/`` here exercise it in place.

Kept importable: relative imports were converted to absolute
``paf.capabilities.memory.*`` / ``archive.organizer.*`` so this still loads when
``archive/`` is on ``sys.path`` (or run directly under PYTHONPATH=.:archive).
"""

from __future__ import annotations

from archive.organizer._organize import IndexEntry, IndexStore
from paf.capabilities.memory._atomic import write_atomic
from paf.capabilities.memory._journal import MEMORY_TYPES

# Per-record separator between fact / date / source_ref in the md line.
_SEP = " — "


def _encode(text: str) -> str:
    """Sanitize a fact for a single md line: replace internal newlines/sep."""
    # Collapse newlines so a multi-line fact never breaks the one-per-line shape.
    return " ".join(text.split())


def _group_index_by_type(entries: list[IndexEntry]) -> dict[str, list[IndexEntry]]:
    """Group index entries by ``type``, each group sorted by ``updated_on``.

    ``updated_on`` is the index's recency axis (bump-on-change); lexical ISO
    ordering == chronological. Ties keep index order (stable sort).
    """
    groups: dict[str, list[IndexEntry]] = {}
    for e in entries:
        groups.setdefault(e.type, []).append(e)
    for recs in groups.values():
        recs.sort(key=lambda e: (e.updated_on,))
    return groups


def render_index(
    store: IndexStore | list[IndexEntry],
) -> str:
    """Render the curated index into a MEMORY.md string (pure, deterministic).

    The curated counterpart to :func:`render_journal_memory`: rather than
    emitting every append-only journal record (which repeats facts as it grows),
    it presents the deduped/superseded view the organizer maintains in
    ``memory-index.json``. Same presentation shape — grouped by the 6 semantic
    ``type``s, one ``## <type>`` section per type present, in sorted ``type``
    order, entries sorted by ``updated_on`` ascending within a section, each
    line ``- <fact> — <updated_on> — <source_ref>``.

    ``store`` may be an :class:`IndexStore` (reads all entries) or an explicit
    list of :class:`IndexEntry` (for testing without touching disk).
    """
    entries: list[IndexEntry]
    if isinstance(store, IndexStore):
        entries = store.load()
    else:
        entries = list(store)

    groups = _group_index_by_type(entries)

    sections: list[str] = []
    for type_name in sorted(MEMORY_TYPES):  # type: ignore[name-defined]
        recs = groups.get(type_name)
        if not recs:
            continue
        lines = [
            f"- {_encode(e.fact)}{_SEP}{e.updated_on}{_SEP}{e.source_ref}"
            if e.updated_on
            else f"- {_encode(e.fact)}{_SEP}{e.source_ref}"
            for e in recs
        ]
        sections.append(f"## {type_name}\n" + "\n".join(lines))

    return "\n\n".join(sections) + ("\n" if sections else "")


def write_index_render(
    store: IndexStore | list[IndexEntry], md_path: str
) -> str:
    """Render the curated index and atomically write ``md_path`` (e.g. ``MEMORY.md``).

    Returns the rendered markdown. Same atomic-write guarantee as the live
    ``write_render``.
    """
    content = render_index(store)
    write_atomic(md_path, content)
    return content
