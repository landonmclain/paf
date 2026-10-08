"""Keyword curator — canonical entities + aliases over skills and notes.

Phases 1–2 so far: the on-disk store, the change watermark, the source readers,
the curator model call, and the worker. No capability registration and no
injection yet — see the vault note ``Designs/Keyword-Curator`` for the design and
phase plan. Run by hand with ``python -m paf.capabilities.keyword_curator``.

- `_store`      — the append-only keyword journal, the derived entity map.
- `_watermark`  — per-source change cursors (mtime + size).
- `_sources`    — skill and note readers → bounded digests.
- `_curate`     — the structured-output curator agent (no tools).
- `_background` — the worker pass: discover → curate → append → rebuild → advance.
- `_report`     — review aid: likely duplicate / ambiguous entities.
"""

from __future__ import annotations

from ._background import PassResult, make_batches, run_keyword_pass
from ._curate import CurateBatch, CurateResult, Entity, curate_batch
from ._report import find_issues, format_issues
from ._sources import SourceItem, scan_notes, scan_skills, scan_sources
from ._store import (
    ENTITY_MAP_FILENAME,
    JOURNAL_FILENAME,
    KeywordJournalStore,
    derive_entity_map,
    make_record,
    normalize_term,
    read_entity_map,
    write_entity_map,
)
from ._watermark import WATERMARK_FILENAME, KeywordWatermark

__all__ = [
    "CurateBatch",
    "CurateResult",
    "Entity",
    "PassResult",
    "SourceItem",
    "curate_batch",
    "find_issues",
    "format_issues",
    "make_batches",
    "run_keyword_pass",
    "scan_notes",
    "scan_skills",
    "scan_sources",
    "ENTITY_MAP_FILENAME",
    "JOURNAL_FILENAME",
    "WATERMARK_FILENAME",
    "KeywordJournalStore",
    "KeywordWatermark",
    "derive_entity_map",
    "make_record",
    "normalize_term",
    "read_entity_map",
    "write_entity_map",
]
