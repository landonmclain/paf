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

# Tool-choice steer: for *code-related* questions, reach for the codebase ladder
# before the raw file tools. This exists because `search_codebase` / `lookup_file`
# / `read_block` are scoped to the indexed repo and already skip ignored dirs
# (`data/`, `.agent-codeindex/`), so they return less noise and fewer tokens than
# a raw `search_files` / `read_file` over the whole tree. Steering (not removal):
# the raw FileSystem tools stay for editing, non-code files, and anything outside
# the code index.
_CODEMAP_GUIDANCE = (
    "Code-related questions. When the user is asking about source code -- where "
    "something is defined or referenced, what a symbol does, how pieces fit "
    "together -- start with the codebase tools, not the raw file tools. Work in order:\n"
    "1. Orient first -- `get_overview` to see the tree before deciding where to look.\n"
    "2. Look before you read -- `lookup_file(path)` for a cheap head (symbol list), "
    "then `lookup_file(path, symbol=X, detail=True)` for one symbol's body. Let one "
    "call's result decide the next; don't issue dependent calls together.\n"
    "3. Read the region -- `read_block(path:L1-L2)` only for the raw source of the "
    "exact span you need; prefer the tightest region, not the whole class or file.\n"
    "(Escape hatch) After orienting, you may `search_codebase` with a known symbol or "
    "word to jump straight to a location, then `lookup_file`/`read_block` the result. "
    "Don't search before you've seen the overview.\n"
    "Trust the results -- once the ladder has shown you the exact definition and "
    "region you need, STOP searching. Do not re-run `search_codebase` to re-confirm "
    "wording, quotes, or error strings you can already see in the block you read; do "
    "not issue repeated near-identical queries after you have located the answer. "
    "Search is for finding, not re-verifying -- when you have what you need, answer.\n"
    "These tools are scoped to the indexed repo and already skip build artifacts and "
    "ignored directories like `data/` and `.agent-codeindex/`, so they return far less "
    "noise and fewer tokens than a raw `search_files` over the whole tree (which "
    "sweeps those ignored dirs). Keep the raw FileSystem tools for editing files, "
    "non-code files, and anything outside the code index."
)


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
        retrieval = self.retrieval or LexicalRetrieval(
            storage=self.storage, scan_root=self.scan_root
        )
        return CodebaseMapToolset(
            scan_root=self.scan_root,
            storage=self.storage,
            retrieval=retrieval,  # type: ignore[arg-type]  # RetrievalBackend protocol
        )

    def get_instructions(self) -> list[str] | None:
        """Provide a tool-choice steer: for code questions, use the ladder first.

        This is a static, additive nudge -- it does not touch the memory recall
        ladder (that is owned by the ``Memory`` capability). It steers tool
        selection toward the scoped, token-bounded codebase tools without removing
        the raw FileSystem tools.
        """
        return [_CODEMAP_GUIDANCE]
