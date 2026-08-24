"""CodebaseMap capability — navigation-first codebase browser (5-tool ladder).

Wraps the codebase-map artifacts behind the locked navigation-first read-model
(LOCKED 2026-08-23): the model climbs a ladder instead of brute-forcing keyword
scans.

    get_overview() → lookup_file(path, symbol, detail) → read_block(block_ref)
                                    ↕                        ↑
                            search_codebase(query)  (escape hatch for known words)

- storage: ``StorageService`` — the byte-medium seam (default ``FlatFileStorage``)
- retrieval: ``RetrievalBackend`` — the search-method seam (default
  ``LexicalRetrieval``; a future semantic backend slots in here without touching tools)
- scan_root: required — v1 is single-repo navigation

Search is the *secondary* escape hatch; navigation (overview → lookup → read_block)
is primary. ``get_repo_map`` (distilled map) is Phase 4. ``callers``/``with_graph``
defer to the graph phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability

from ._retrieval import LexicalRetrieval, RetrievalBackend
from ._store import FlatFileStorage, StorageService
from ._toolset import CodebaseMapToolset


@dataclass
class CodebaseMap(AbstractCapability):
    """Navigation-first codebase map: overview → lookup → read_block + search.

    Exposes the five-tool ladder (`get_overview`, `lookup_file`, `read_block`,
    `search_codebase`, `get_repo_map`). All reads go through the storage seam and
    are confined to `scan_root`.
    """

    scan_root: str | Path
    """Project root to index/navigate. v1 is single-repo navigation."""

    storage: StorageService = field(default_factory=FlatFileStorage)
    """Byte-medium seam (flat JSON now; SQLite/object-store later)."""

    retrieval: RetrievalBackend | None = None
    """Search-method seam; defaults to `LexicalRetrieval` over `storage`."""

    max_search_results: int = field(default=25)
    """Default cap for `search_codebase` hits."""

    def get_toolset(self) -> CodebaseMapToolset:
        """Build and return the five-tool codebase-map toolset."""
        retrieval = self.retrieval or LexicalRetrieval(storage=self.storage, scan_root=self.scan_root)
        return CodebaseMapToolset(
            scan_root=self.scan_root,
            storage=self.storage,
            retrieval=retrieval,  # type: ignore[arg-type]  # RetrievalBackend protocol
        )
