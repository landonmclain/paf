"""Retrieval seam — the search-method extension point (BM25 first pass now live).

This is the *how queries are answered* seam (vs `_store.py`'s *where artifacts are
stored* seam). ``RetrievalBackend`` is the designated drop point for a future
semantic/vector backend; ``LexicalRetrieval`` is the default BM25-backed
implementation (Phase 2 quick start, per Landon 2026-08-23): it loads the index
(head) records and runs field-weighted BM25 rank over them, returning the universal
``SearchHit`` shape. The resolved symbol graph + relationships (the key deliverable)
replaces/augments this later.

``LexicalRetrieval`` reads only; it NEVER writes to storage (flat JSON stays the
single source of truth).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ._rank import rank
from ._schema import DistilledEntry, FileInfo, SearchHit
from ._store import ArtifactKind, FlatFileStorage


class RetrievalBackend(Protocol):
    """Search-method seam. A future semantic backend implements these two methods."""

    def search(
        self,
        query: str,
        *,
        mode: str,
        scope: Any = None,
        limit: int = 20,
    ) -> list[SearchHit]: ...

    def distilled_map(self, *, map_tokens: int, focus: Any = None) -> list[DistilledEntry]: ...


def _rehydrate(record: dict) -> FileInfo:
    """Build a ``FileInfo`` from a raw manifest record dict (reverse of asdict)."""
    from ._schema import SymbolInfo

    syms: list[SymbolInfo] = []
    for s in record.get("searchable_symbols") or []:
        span = s.get("span")
        syms.append(
            SymbolInfo(
                name=s.get("name", ""),
                qualified=s.get("qualified"),
                kind=s.get("kind", "OTHER"),
                signature=s.get("signature"),
                doc=s.get("doc"),
                body=s.get("body"),
                span=span,
                visibility=s.get("visibility"),
                children=[],
                parent=s.get("parent"),
            )
        )
    return FileInfo(
        path=record.get("path", ""),
        language=record.get("language"),
        doc=record.get("doc"),
        role=record.get("role"),
        tags=record.get("tags") or [],
        symbols=record.get("symbols") or [],
        imports=record.get("imports") or [],
        references_count=record.get("references_count", 0),
        searchable_symbols=syms,
        unparsed=record.get("unparsed", False),
    )


class LexicalRetrieval:
    """Default: BM25 field-weighted search over the index (head) records (Phase 2 first pass)."""

    def __init__(
        self,
        storage: FlatFileStorage | None = None,
        scan_root: str | Path | None = None,
    ) -> None:
        self.storage = storage or FlatFileStorage()
        self._scan_root_value = Path(scan_root).resolve() if scan_root else None

    def search(
        self,
        query: str,
        *,
        mode: str = "keywords",
        scope: Any = None,
        limit: int = 20,
    ) -> list[SearchHit]:
        """BM25-field-weighted lexical search over the index (head) records.

        ``mode`` is accepted for API compatibility; broad keyword ranking is the
        Phase-2 first-pass behavior (exact/`callers`/graph modes come with the
        graph phase). Reads the index artifact and ranks it.
        """
        records = self._index_records()
        files = [_rehydrate(r) for r in records]
        return rank(query, files, limit=limit)

    def distilled_map(self, *, map_tokens: int, focus: Any = None) -> list[DistilledEntry]:
        """STUB — ranked distilled map (Phase 4)."""
        raise NotImplementedError("distilled_map is not built until Phase 4")

    def _index_records(self) -> list[dict]:
        payload = self.storage.read(ArtifactKind.INDEX, self._scan_root())
        records = (payload or {}).get("files", [])
        return [r for r in records if isinstance(r, dict)]

    def _scan_root(self) -> Path:
        """The scan root we search. v1 is single-repo: explicit scan_root wins,
        else derive from the storage override dir, else cwd (tests pass a storage
        with a fixed index_root, so ``list_scan_roots`` reports the override)."""
        if self._scan_root_value is not None:
            return self._scan_root_value
        roots = self.storage.list_scan_roots()
        if roots:
            return Path(roots[0])
        return Path.cwd()
