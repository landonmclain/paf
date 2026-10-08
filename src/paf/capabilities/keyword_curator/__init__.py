"""Keyword curator — canonical entities + aliases over skills and notes.

Phase 1 (this package so far): the on-disk store and the change-detection
watermark. No model, no capability registration, no injection yet — see the
vault note ``Designs/Keyword-Curator`` for the full design and phase plan.

- `_store`     — the append-only keyword journal, the derived entity map.
- `_watermark` — per-source change cursors (mtime + size).
"""

from __future__ import annotations

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
