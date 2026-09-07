"""The storage seams: :class:`StorageService` and :class:`AppendOnlyStore`.

A capability that persists artifacts talks to one of these seams — never to a
concrete path — so the backend (flat file now, SQLite / Postgres / other later)
can be swapped without touching the capability's analyzers, retrieval, or tool
surface. Modeled on ``agent-foundation`` (and codebase-map's ``StorageService``,
which was already on that shape).

Two distinct persistence shapes, deliberately separate:

- :class:`StorageService` — **whole-artifact** read/write (derived JSON
  artifacts that get fully rewritten: codebase-map's snapshot/map, the
  conversation-titles map). ``read_block`` is the seam-shaped window retrieval
  gives search.
- :class:`AppendOnlyStore` — **append-only timeline** (a write-once log that
  never rewrites history: the memory journal, override sidecars). Separate
  because conflating the two would force an append-only design into
  whole-rewrites.

Both are generic over ``kind`` (artifact/log type, a ``str`` — callers may pass
their own ``StrEnum``) and ``key`` (scope: a scan root, an agent name).

Deliberately **not** part of these seams (see package docstring):

- ``sync_changed`` (source-file freshness / re-index) — a codebase-map
  *retrieval* concern about source content, stays local to codebase-map.
- Semantic search / vector ranking / RAG — a retrieval-capability concern
  layered **on top** of storage, not storage itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from paf.capabilities._storage._block import BlockRef


class StorageService(Protocol):
    """Whole-artifact persistence: read/write by kind+key, enumerate, resolve ranges."""

    def read(self, kind: str, key: str | Path) -> Any | None:
        """Return the artifact for ``kind`` at ``key`` (None if absent/corrupt)."""
        ...

    def write(self, kind: str, key: str | Path, payload: Any) -> None:
        """Persist ``payload`` for ``kind`` at ``key`` (atomic, never torn)."""
        ...

    def list_keys(self) -> list[Path]:
        """Every ``key`` that holds at least one artifact (discovery / backstop)."""
        ...

    def read_block(self, block_ref: BlockRef) -> str | None:
        """Materialise the exact bounded range a :class:`BlockRef` names."""
        ...


class AppendOnlyStore(Protocol):
    """Append-only JSON timeline of records, keyed by ``kind`` + ``key``."""

    def append(self, kind: str, key: str, record: dict[str, Any]) -> dict[str, Any]:
        """Append ``record``; returns it. A crash never leaves a torn line."""
        ...

    def read_all(self, kind: str, key: str) -> list[dict[str, Any]]:
        """All records for ``(kind, key)`` in append order (torn lines skipped)."""
        ...
