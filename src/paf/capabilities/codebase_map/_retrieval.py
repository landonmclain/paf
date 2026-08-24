"""Retrieval seam — the search-method extension point (tokens-backed now live).

This is the *how queries are answered* seam (vs `_store.py`'s *where artifacts are
stored* seam). ``RetrievalBackend`` is the designated drop point for a future
semantic/vector backend; ``LexicalRetrieval`` is the default implementation.

Under the LOCKED read-model (2026-08-23), ``search`` scores over **``tokens.json``**
(the inverted, symbol-scoped search backend — name+sig+doc postings), presents the
**``index`` HEAD** row for each surfaced file (path, language, module docstring,
matched name lists), and dereferences ``manifest`` only on pick — progressive
disclosure. Body text is deliberately not tokenized (grep-on-manifest / ``read_block``
covers in-body). The lean head is the *presentation* layer, not the scoring source.

``LexicalRetrieval`` reads only; it NEVER writes to storage (flat JSON stays the single
source of truth).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ._schema import DistilledEntry, SearchHit
from ._store import ArtifactKind, FlatFileStorage, StorageService
from ._tokens import tokens_rank


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


class LexicalRetrieval:
    """Default: field-weighted search over the ``tokens.json`` search backend.

    Scores tokens (name > signature > doc) → annotates each hit with its ``index``
    HEAD row (module-docstring head + matched name lists) → returns ``SearchHit``s
    whose ``block`` collapses to the best-matched symbol's ``manifest_ref``.
    """

    def __init__(
        self,
        storage: StorageService | None = None,
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
        """Token-scored search over the search backend, presented as index heads.

        ``mode`` is accepted for API compatibility; broad token ranking is the
        Phase-3-adjacent first pass (exact/`callers`/graph modes come with the graph
        phase). Scores ``tokens.json`` (the search backend) and presents each
        surfaced file's lean ``index`` HEAD row (module docstring + matched names).
        """
        tokens = self._tokens_payload()
        return tokens_rank(query, tokens, limit=limit, heads=self._heads_by_path())

    def distilled_map(self, *, map_tokens: int, focus: Any = None) -> list[DistilledEntry]:
        """STUB — ranked distilled map (Phase 4)."""
        raise NotImplementedError("distilled_map is not built until Phase 4")

    # ---- reads --------------------------------------------------------------

    def _tokens_payload(self) -> dict:
        payload = self.storage.read(ArtifactKind.TOKENS, self._scan_root())
        return payload if isinstance(payload, dict) else {}

    def _heads_by_path(self) -> dict[str, dict]:
        """Lean ``index`` HEAD rows keyed by path (the ``present the head`` layer)."""
        index = self.storage.read(ArtifactKind.INDEX, self._scan_root())
        rows = (index or {}).get("files", []) if isinstance(index, dict) else []
        heads: dict[str, dict] = {}
        for r in rows:
            if isinstance(r, dict) and isinstance(r.get("path"), str):
                heads[r["path"]] = r
        return heads

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
