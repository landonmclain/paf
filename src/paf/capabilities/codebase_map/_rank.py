"""Ranking: BM25 wrapper + fusion (STUB — Phase 0 layout only).

Planned (per Codebase-Map-Capability-Spec §3.7.1/§3.7.2): wrap the repo's existing,
dependency-free BM25 core (`paf.capabilities.search._ranking`) — do NOT copy it and do
NOT add a retrieval framework (no Rank-BM25 / whoosh / LlamaIndex — the LlamaIndex
experience was "didn't add anything"). Add field/type weighting (manifest
`searchable_symbols`: name > qualified > file_path > signature > doc > imports > body)
and the fusion score `score = α·bm25 + β·ref_weight + γ·kind_bias` (αβγ tunable).

Phase 3 delivers this. Phase 0 is storage + schema + eval shell only; stub for layout.
"""

from __future__ import annotations

from ._schema import FileInfo, SearchHit


def rank(query: str, files: list[FileInfo], *, limit: int = 20) -> list[SearchHit]:
    """STUB — BM25 field-weighted rank over manifest records (Phase 3)."""
    raise NotImplementedError("rank is not built until Phase 3")
