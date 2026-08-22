"""Call graph + references (STUB — Phase 0 layout only).

Planned (per Codebase-Map-Capability-Spec § `_graph.py` + §5.x): compute caller/callee
edges and inbound-reference counts **on demand** from the per-file symbol index — NOT a
persisted `call_graph.json` that goes stale (the freshness decision, 2026-08-22).
Derivation happens when `callers` / `get_repo_map` / ref-weight ranking asks for it,
feeding `mode="callers"` and `references_count`.

Phase 2 delivers this. Phase 0 is storage + schema + eval shell only; stub for layout.
"""

from __future__ import annotations

from typing import Any

from ._schema import FileInfo


def build_edges(files: list[FileInfo]) -> list[Any]:
    """STUB — derive caller→callee edges from per-file symbols (Phase 2)."""
    raise NotImplementedError("build_edges is not built until Phase 2")


def reference_counts(files: list[FileInfo]) -> dict[str, int]:
    """STUB — inbound-reference counts per symbol, derived on read (Phase 2)."""
    raise NotImplementedError("reference_counts is not built until Phase 2")
