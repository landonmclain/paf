"""CodebaseMap capability (STUB — Phase 0 layout only).

Planned (per Codebase-Map-Capability-Spec § `_capability.py`): a `@dataclass`
(`AbstractCapability`) with:
- `storage: StorageService` (required — the seam; default `FlatFileStorage()`)
- `retrieval: RetrievalBackend` (default `LexicalRetrieval()` — the search-method seam)
- `scan_root: Path` (required — one project, v1 single-repo nav)
- `map_tokens: int = 1000` (distilled-map budget)
- `max_search_results: int = 25`, `max_search_chars: int = 8000`
- `dependencies: set[str] | None` (tool dependency allowlist)

Implements `from_spec` / `get_serialization_name` / `get_instructions` and exposes the
toolset (search_codebase + get_repo_map). None of that is built in Phase 0 — Phase 0 is
storage + schema + eval shell only. This module exists so the full package layout is
visible up front (per Landon: stub all files in Phase 0). Considered a stub until the
capability is assembled in a later phase.
"""

from __future__ import annotations

from typing import Any

# NOTE: uncomment when assembling the capability (Phase 2+), not Phase 0.
# from pydantic_ai.capabilities import AbstractCapability
# from ._store import FlatFileStorage, StorageService
# from ._retrieval import LexicalRetrieval, RetrievalBackend


class CodebaseMap:  # will become AbstractCapability[...]
    """STUB — the CodebaseMap capability.

    Not yet wired (no toolset, no registry entry, no from_spec). The fields below are
    the planned surface; they will be real dataclass fields with defaults once the
    capability is assembled. Refusing to fake a working capability: leaving this as a
    named placeholder so imports resolve and the layout is complete, but nothing here
    does work yet.
    """

    def __init__(
        self,
        *,
        scan_root: str,
        storage: Any = None,  # StorageService
        retrieval: Any = None,  # RetrievalBackend
        map_tokens: int = 1000,
    ) -> None:
        self.scan_root = scan_root
        self.storage = storage
        self.retrieval = retrieval
        self.map_tokens = map_tokens

    def get_serialization_name(self) -> str | None:
        return "CodebaseMap"

    def get_instructions(self) -> str | None:
        # Planned: None — tools are self-describing (per spec).
        return None

    def __repr__(self) -> str:  # pragma: no cover - stub
        return f"CodebaseMap(scan_root={self.scan_root!r})"  # pragma: no cover
