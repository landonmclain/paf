"""Codebase-map capability: extract a codebase map, then search it.

Status: **Phase 0 — storage + schema seam + eval shell** (2026-08-22). The full package
layout is stubbed up front (per Landon); only the storage seam + schema records + eval
shell are implemented so far. Extraction/search phases build on top:
Phase 1 extraction (THE deliverable) → Phase 2 graph+refs → Phase 3 search → Phase 4 map.

Public surface: the storage records (`Span`, `BlockRef`, `SymbolInfo`, `FileInfo`,
`SearchHit`, `DistilledEntry`), the storage seam (`ArtifactKind`, `StorageService`,
`FlatFileStorage`, `ScanScope`, `index_dir_for_scan_root`), the pointer reader
(`read_block`), the scan driver (`scan_and_write`, `walk_all_files`), the reference
counts (`reference_counts`, `apply_references`), and the BM25 first-pass search
(`rank`, `LexicalRetrieval`). Not yet public: the capability/toolset (graph phase).
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
    # BM25 first-pass search (Phase 2)
    "rank",
    "rank_fused",
    "LexicalRetrieval",
]
