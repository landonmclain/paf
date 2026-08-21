"""Deterministic render: journal -> MEMORY.md (the presentation view).

The three-layer memory design: **journal** (append-only fact stream, M2) ->
**[render]** (this module) -> **MEMORY.md** (what the answerer reads). The
render is the "database" step: it takes the journal's write-once records and
produces the organized markdown notebook — grouping by semantic `type`,
sorting by `born_on` (recency), and attaching date + source per memory. It is
**pure and deterministic**: no model, no state, no I/O beyond reading the
journal and writing the file. LLM curation may come later; "if it doesn't have
to be [LLM], that is better" — and it doesn't here.

Format (LOCKED with Landon, 2026-08-17): the old 4-section canonical
(Facts/Preferences/Pending/Notes) is **thrown out for now**. The render groups
by the journal's 6 semantic `type`s — identity, preference, fact, event, task,
current — one `## <type>` section per type present, records sorted by
`born_on` ascending within each section, and each memory line carries **at
least fact, date (born_on), and source (source_ref)**:

    ## preference
    - user prefers flat-file storage — 2026-08-17T10:30:00+00:00 — seed-a1b2:12

Determinism: types iterate in a fixed sorted order; records within a type sort
by born_on (lexical ISO == chronological), ties keep append order. The same
journal always renders the same bytes.
"""

from __future__ import annotations

from ._journal import MEMORY_TYPES, JournalStore, MemoryRecord

# Per-record separator between fact / born_on / source_ref in the md line.
_SEP = " — "


def _group_by_type(records: list[MemoryRecord]) -> dict[str, list[MemoryRecord]]:
    """Group records by `type`, each group sorted by `born_on` (append-order ties).

    Iterates in deterministic append order here; final sorted emission is in
    :func:`render_journal_memory`.
    """
    groups: dict[str, list[MemoryRecord]] = {}
    for rec in records:
        groups.setdefault(rec.type, []).append(rec)
    for recs in groups.values():
        recs.sort(key=lambda r: (r.born_on,))
    return groups


def render_journal_memory(
    store: JournalStore | list[MemoryRecord],
) -> str:
    """Render the journal into a MEMORY.md string (pure, deterministic).

    ``store`` may be a :class:`JournalStore` (reads all records) or an explicit
    list of :class:`MemoryRecord` (for testing without touching disk). Sections
    appear only for types that have records, in sorted ``type`` order; within a
    section records sort by ``born_on`` ascending. Each line:
    ``- <fact> — <born_on> — <source_ref>``.
    """
    records: list[MemoryRecord]
    if isinstance(store, JournalStore):
        records = [r for _, r in store.read_all()]
    else:
        records = list(store)

    groups = _group_by_type(records)

    sections: list[str] = []
    for type_name in sorted(MEMORY_TYPES):  # type: ignore[name-defined]
        recs = groups.get(type_name)
        if not recs:
            continue
        lines = [
            f"- {_encode(r.fact)}{_SEP}{r.born_on}{_SEP}{r.source_ref}"
            if r.born_on
            else f"- {_encode(r.fact)}{_SEP}{r.source_ref}"
            for r in recs
        ]
        sections.append(f"## {type_name}\n" + "\n".join(lines))

    return "\n\n".join(sections) + ("\n" if sections else "")


def write_render(store: JournalStore | list[MemoryRecord], md_path: str) -> str:
    """Render the journal and atomically write ``md_path`` (e.g. ``MEMORY.md``).

    Returns the rendered markdown. Uses the memory package's atomic writer so a
    crash mid-write never leaves a torn notebook.
    """
    from ._atomic import write_atomic  # local import keeps top of module import-light

    content = render_journal_memory(store)
    write_atomic(md_path, content)
    return content


def _encode(text: str) -> str:
    """Sanitize a fact for a single md line: replace internal newlines/sep."""
    # Collapse newlines so a multi-line fact never breaks the one-per-line shape.
    return " ".join(text.split())
