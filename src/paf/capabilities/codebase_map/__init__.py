"""Codebase-map capability: extract a codebase map, then search it.

Status: **Phase 2 lite + the five-tool ladder shipped 2026-08-23**. Extraction
(python DONE), the lean ``index`` head, ``manifest`` detail, and the ``tokens``
search backend are all landed; the navigation-first ``overview`` + the five-tool
read-model (`get_overview`, `lookup_file`, `read_block`, `search_codebase`,
`get_repo_map`) are built. Phases continue: graph+refs (build_edges) → search tuning
→ Phase 4 `get_repo_map`.

Public surface: the storage records (`Span`, `BlockRef`, `SymbolInfo`, `FileInfo`,
`SearchHit`, `DistilledEntry`), the storage seam (`ArtifactKind`, `StorageService`,
`FlatFileStorage`, `ScanScope`, `index_dir_for_scan_root`), the pointer reader
(`read_block`), the scan driver (`scan_and_write`, `walk_all_files`), the outline
(`build_overview`, `load_overview`), the reference counts (`reference_counts`,
`apply_references`), the token search backend (`build_tokens`, `tokens_rank`), the
BM25 first-pass search (`rank`, `rank_fused`, `LexicalRetrieval`), and the five-tool
capability (`CodebaseMap`, `CodebaseMapToolset`).
"""

from __future__ import annotations

from ._block import BlockRef, Span, read_block
from ._capability import CodebaseMap
from ._graph import apply_references, reference_counts
from ._overview import build_overview, load_overview
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
from ._toolset import CodebaseMapToolset

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
    # overview (navigation start)
    "build_overview",
    "load_overview",
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
    # five-tool ladder (navigation-first read-model)
    "CodebaseMapToolset",
    "CodebaseMap",
]
