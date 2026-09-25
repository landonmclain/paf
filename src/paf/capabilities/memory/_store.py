"""Shared typed memory records, protocols, and validation for the memory capability.

This module holds the load-bearing shared types the memory spine needs
(``_journal``, ``_index``, ``_render``, ``_view``, the owner's
``JournalAppendToolset``):

- ``MemoryFile`` — the single shared record type
- ``MemoryStore`` — the read-side protocol
- ``validate_store_path`` — shared path validation

Only the **read side** lives here. The versioned-CAS write contract
(``write``/``delete``/``get_operation``) and the bounded lexical-search surface
(``lexical_search``, ``SearchableMemoryStore``) were removed: the three-layer
journal->index->render spine made them dead — nothing calls a memory store's
write/delete, and no real code searches one. ``Memory.store`` is read-only: the
read path always supplies ``JournalViewStore`` of the rendered journal.
Atomic writes come from the shared ``_storage._atomic``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_VALID_SEGMENT_RE = re.compile(r"[A-Za-z0-9_.-]{1,200}")


@dataclass(frozen=True)
class MemoryFile:
    """One memory file and its opaque compare-and-set version."""

    content: str
    version: str
    operation_id: str | None
    truncated: bool


def validate_store_path(path: str) -> None:
    r"""Reject path strings that could escape a store's root directory."""
    if not all(
        _VALID_SEGMENT_RE.fullmatch(segment) and ".." not in segment
        for segment in path.split("/")
    ):
        raise ValueError(f"invalid memory path: {path!r}")


@runtime_checkable
class MemoryStore(Protocol):
    """Async read-only view over rendered memory files.

    Only the read surface a capability needs: ``read`` returns a file's
    content. There is no write/delete/search/list — the memory spine renders
    ``MEMORY.md`` once (``_render.write_render``) and recall only reads it back.
    """

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        """Return the file at `path`, or `None` if it does not exist."""
        ...  # pragma: no cover
