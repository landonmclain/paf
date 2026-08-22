"""CodebaseMapToolset (STUB — Phase 0 layout only).

Planned (per Codebase-Map-Capability-Spec § `_toolset.py`): a `FunctionToolset` exposing
exactly two tools, chosen for how they *guide* the model:
- `search_codebase(query, mode=None, scope=None, limit=20, with_graph=False)` = **localize**
- `get_repo_map(map_tokens=None, focus=None)` = **orient** (aider --map-tokens distilled map)

`mode` is optional — auto-inferred by default (keyword + exact + source-token fused);
explicit `keywords`/`exact`/`code`/`callers` as escape hatch + extension seam for a
future `semantic`. `with_graph` adds inline caller/callee + blast radius (Phase 2).
All hits return the universal `SearchHit`/`BlockRef` shape.

None of this is built in Phase 0 — Phase 0 is storage + schema + eval shell only.
Stub exists so the full package layout is visible up front.
"""

from __future__ import annotations

from typing import Any


class CodebaseMapToolset:
    """STUB — the two-tool codebase navigation toolset.

    Not built in Phase 0. Placeholder so the package layout is complete and imports
    resolve. Will become a `FunctionToolset`-derived bundle in a later phase.
    """

    def __init__(self, storage: Any = None, retrieval: Any = None, scan_root: str = "") -> None:
        self.storage = storage
        self.retrieval = retrieval
        self.scan_root = scan_root

    def search_codebase(self, query: str, **kwargs: Any) -> list[Any]:
        """STUB — search_codebase (localize). Implemented in Phase 3."""
        raise NotImplementedError("search_codebase is not built until Phase 3")

    def get_repo_map(self, **kwargs: Any) -> list[Any]:
        """STUB — get_repo_map (orient). Implemented in Phase 4."""
        raise NotImplementedError("get_repo_map is not built until Phase 4")
