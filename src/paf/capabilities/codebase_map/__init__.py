"""Codebase-map capability: extract a codebase map, then search it.

Status: **Phase 0 — storage + schema seam + eval shell** (2026-08-22), with Phase 1
extraction (python DONE) shipped and the Phase-2 arrow of the locked read-model
landed (2026-08-23): the ``index`` is the LEAN per-file head, ``manifest`` the DETAILED
layer, and ``tokens`` the symbol-scoped search backend that ``search_codebase`` scores.
Extraction/search phases continue to build on top: Phase 2 graph+refs (build_edges) →
Phase 3 search tuning → Phase 4 map.

Public surface: the storage records (`Span`, `BlockRef`, `SymbolInfo`, `FileInfo`,
`SearchHit`, `DistilledEntry`), the storage seam (`ArtifactKind`, `StorageService`,
`FlatFileStorage`, `ScanScope`, `index_dir_for_scan_root`), the pointer reader
(`read_block`), the scan driver (`scan_and_write`, `walk_all_files`), the reference
counts (`reference_counts`, `apply_references`), the token search backend
(`build_tokens`, `tokens_rank`), and the BM25 first-pass search (`rank`,
`LexicalRetrieval`). Not yet public: the capability/toolset (graph phase).
"""

from __future__ import annotations

from ._block import BlockRef, Span, read_block
from ._graph import apply_references, reference_counts
from ._rank import rank, rank_fused
from ._retrieval import LexicalRetrieval
from ._scan import scan_and_write, walk_all_files
from ._schema import DistilledEntry, FileInfo, SearchHit, SymbolInfo
from ._store import (
    ArtifactKind,
    FlatFileStorage,
    ScanScope,
    StorageService,
    index_dir_for_scan_root,
)
from ._tokens import build_tokens, tokens_rank

__all__ = [
    # records (schema)
    "BlockRef",
    "Span",
    "SymbolInfo",
    "FileInfo",
    "SearchHit",
    "DistilledEntry",
    # storage seam
    "ArtifactKind",
    "StorageService",
    "FlatFileStorage",
    "ScanScope",
    "index_dir_for_scan_root",
    # pointer reader
    "read_block",
    # scan driver
    "walk_all_files",
    "scan_and_write",
    # references (Phase 2)
    "reference_counts",
    "apply_references",
    # token search backend (the search_codebase scorer, locked read-model)
    "build_tokens",
    "tokens_rank",
    # BM25 first-pass search (Phase 2)
    "rank",
    "rank_fused",
    "LexicalRetrieval",
]
