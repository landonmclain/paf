"""CodebaseMapToolset (STUB — layout only, matches LOCKED read-model).

Planned (per Codebase-Map-Capability-Spec § `_toolset.py`): a `FunctionToolset`
exposing the **navigation-first** 5-tool surface (LOCKED 2026-08-23):

- `get_overview()` = **orient to the top** — the `overview.json` navigation tree
  (folders → files, collapsed from the ignore list). The STARTING POINT.
- `lookup_file(path, symbol=None, detail=False)` = **the bridge** (head → manifest):
  - `lookup_file(path)`               → the `index` HEAD row (metadata + module-docstring
                                       head + symbol NAME list) — no bodies
  - `lookup_file(path, symbol=X)`     → that symbol's head only (name/kind/span)
  - `lookup_file(path, detail=True)`  → the full `manifest` block for the file
                                       (all symbols' docstring/signature/span)
  - `lookup_file(path, symbol=X, detail=True)` → that symbol's full `manifest` entry
  Guard: never replaces reading the overview — it is a *jump*, not the *path*.
- `read_block(block_ref)` = **the whole-file / region rung** — exact source bytes,
  the only path to raw source (query-the-pointer; heavy source deferred).
- `search_codebase(query, limit=20)` = **the thin escape hatch** — one fused call over
  `tokens.json` (symbol-scoped, field-weighted BM25); **NO mode menu**. Scores tokens
  → PRESENTS the `index` head row (path, language, module docstring, matched names)
  with `manifest_ref` pointers → the model dereferences `manifest`/`read_block` on pick.
- `get_repo_map(map_tokens=None, focus=None)` = **orient map** (aider --map-tokens
  distilled map).

`callers`/`with_graph` defer to the graph phase (YAGNI). A future `semantic` backend is
an alternative `RetrievalBackend`, never a `mode`. All hits collapse to the universal
`SearchHit`/`BlockRef` shape.

None of this is built yet — this is stub layout so the package surface matches the
LOCKED spec (navigation-first, search secondary). Stub exists so the full package layout
is visible up front.
"""

from __future__ import annotations

from typing import Any


class CodebaseMapToolset:
    """STUB — the five-tool codebase navigation toolset (locked read-model).

    Not built yet. Placeholder so the package layout is complete and imports
    resolve. Will become a `FunctionToolset`-derived bundle in a later phase,
    exposing the navigation-first ladder: get_overview → lookup_file → read_block,
    with search_codebase as the thin escape hatch and get_repo_map for orientation.
    """

    def __init__(self, storage: Any = None, retrieval: Any = None, scan_root: str = "") -> None:
        self.storage = storage
        self.retrieval = retrieval
        self.scan_root = scan_root

    def get_overview(self) -> Any:
        """STUB — get_overview (navigation start). Builds on `overview.json`."""

    def lookup_file(self, path: str, symbol: str | None = None, detail: bool = False) -> Any:
        """STUB — lookup_file (head → manifest bridge). Implemented in Phase 3."""

    def read_block(self, block_ref: Any) -> str | None:
        """STUB — read_block (raw source by BlockRef). See `_block.py`."""

    def search_codebase(self, query: str, limit: int = 20) -> list[Any]:
        """STUB — search_codebase (thin escape hatch over tokens.json). Phase 3."""
        raise NotImplementedError("search_codebase is not built until Phase 3")

    def get_repo_map(self, map_tokens: int | None = None, focus: str | None = None) -> Any:
        """STUB — get_repo_map (orient). Implemented in Phase 4."""
        raise NotImplementedError("get_repo_map is not built until Phase 4")
