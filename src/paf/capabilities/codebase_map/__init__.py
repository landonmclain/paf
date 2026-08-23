"""Codebase-map capability: extract a codebase map, then search it.

Status: **Phase 0 — storage + schema seam + eval shell** (2026-08-22). The full package
layout is stubbed up front (per Landon); only the storage seam + schema records + eval
shell are implemented so far. Extraction/search phases build on top:
Phase 1 extraction (THE deliverable) → Phase 2 graph+refs → Phase 3 search → Phase 4 map.

Public surface: the storage records (`Span`, `BlockRef`, `SymbolInfo`, `FileInfo`,
`SearchHit`, `DistilledEntry`), the storage seam (`ArtifactKind`, `StorageService`,
`FlatFileStorage`, `index_dir_for_scan_root`), the pointer reader (`read_block`),
and the scan driver (`scan_and_write`, `iter_python_files`). Not yet public: the
capability/toolset (Phases 2–4).
"""

from __future__ import annotations

from ._block import BlockRef, Span, read_block
from ._schema import DistilledEntry, FileInfo, SearchHit, SymbolInfo
from ._scan import iter_python_files, scan_and_write
from ._store import (
    ArtifactKind,
    FlatFileStorage,
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
    "index_dir_for_scan_root",
    # pointer reader
    "read_block",
    # scan driver
    "iter_python_files",
    "scan_and_write",
]
