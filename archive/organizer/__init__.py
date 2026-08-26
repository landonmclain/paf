"""ARCHIVED memory-organizer + curated index (removed from the live memory path).

**Status: ARCHIVED 2026-08-26.** This package preserves the *organizer*, the
*curated index* (``IndexStore`` / ``IndexEntry``), the *consolidation second
pass* (``run_consolidation``), and the *index renderer* (``render_index`` /
``write_index_render``) — the entire "LLM-maintained curated index" idea that
was **unhooked from the live compaction path** on 2026-08-25.

The live memory pipeline no longer uses any of this: it is now the append-only
``memory-journal.jsonl`` + deterministic ``write_render`` (tier t1–t3) with no
index, no organizer, and no second consolidation pass. This archive exists so
the idea can be revived later without rewriting it.

The module is kept **importable** (relative imports converted to absolute
``paf.capabilities.memory.*`` / ``archive.organizer.*``) so it can be loaded for
reference or re-run with ``PYTHONPATH=.:archive``. It is deliberately NOT part
of the installed package (``pyproject.toml`` packages only ``src/paf``).
"""

from archive.organizer._organize import (
    ConfirmDecision,
    IndexEntry,
    IndexStore,
    NewDecision,
    Stage,
    SupersedeDecision,
    render_index_text,
    render_journal_tail,
    run_consolidation,
    run_organizer,
)
from archive.organizer.render_index import (
    render_index,
    write_index_render,
)

__all__ = [
    "ConfirmDecision",
    "IndexEntry",
    "IndexStore",
    "NewDecision",
    "Stage",
    "SupersedeDecision",
    "render_index",
    "render_index_text",
    "render_journal_tail",
    "run_consolidation",
    "run_organizer",
    "write_index_render",
]
