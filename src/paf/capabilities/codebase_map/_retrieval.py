"""Retrieval seam — the search-method extension point (STUB interface, Phase 0 layout).

This is the *how queries are answered* seam (vs `_store.py`'s *where artifacts are
stored* seam). Today it holds ONLY the interface + a lexical-default placeholder — zero
semantic machinery (YAGNI; semantic is "possible, probably never" per Landon 2026-08-22;
we spec the seam, not the implementation).

``RetrievalBackend`` is the designated drop point for a future semantic/vector backend:
it must implement `search` returning the universal `SearchHit` shape and `distilled_map`
returning `DistilledEntry`. The search implementations (`LexicalRetrieval`) are Phase 3;
this file in Phase 0 is the honest interface so the layout is complete.
"""

from __future__ import annotations

from typing import Any, Protocol

from ._schema import DistilledEntry, SearchHit


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
    """STUB — default: keyword / exact / code / callers over the storage seam's artifacts.

    Re-homes Phase 3 search logic (manifest scan, tokens.json inverted index,
    call_graph.json callers). Reads only; NEVER writes to storage (flat JSON stays the
    single source of truth). Not implemented in Phase 0.
    """

    def __init__(self, storage: Any = None) -> None:
        self.storage = storage

    def search(
        self,
        query: str,
        *,
        mode: str = "keywords",
        scope: Any = None,
        limit: int = 20,
    ) -> list[SearchHit]:
        """STUB — lexical search over the storage seam's artifacts (Phase 3)."""
        raise NotImplementedError("LexicalRetrieval.search is not built until Phase 3")

    def distilled_map(self, *, map_tokens: int, focus: Any = None) -> list[DistilledEntry]:
        """STUB — ranked distilled map (Phase 3/4)."""
        raise NotImplementedError("distilled_map is not built until Phase 3")
