"""Scan driver — walk a repo and write the codebase-map artifacts (Phase 1→2 bridge).

This is the missing orchestration that ties the two existing pieces together:
``analyze_file`` (extraction, Phase 1) + ``FlatFileStorage`` (storage seam, Phase 0).
It walks a scan root, extracts every supported python file, and persists the results
to the co-located index dir so the artifact files exist **on disk** and are inspectable
(``manifest.json`` = per-file search surface, ``structure.json`` = full symbol tree,
``snapshot.json`` = the freshness fingerprint ``sync_changed`` later keys off).

Python only for now (per Landon: one language at a time; extraction gate passed for
python). ``_SUPPORTED`` in ``_analyze.py`` drives the language set — this driver is
language-agnostic in intent but v1 targets the python tree.

The driver is deliberately *dumb*: it does not rank, search, or build a graph. It
exists so the data lands on disk and can be eyeballed before any search logic is
built on top (spec ordering: extraction completeness is the gate for search tuning).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ._analyze import analyze_file
from ._store import ArtifactKind, FlatFileStorage

_INDEX_SUBDIR = ".agent-codeindex"

# Files/symbols we don't bother indexing. Russet/scratch/ugly internals.
_SKIP_DIRS = {_INDEX_SUBDIR, ".git", "__pycache__", ".venv", "node_modules", "archive", "History"}
_SKIP_FILES = {"__init__.py"}  # docstring-only + __version__ assignments; not indexed (open deferred item)


def iter_python_files(root: str | Path) -> list[Path]:
    """Stable-ordered list of ``*.py`` source files under ``root`` (skipping internals)."""
    root = Path(root)
    out: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(p in _SKIP_DIRS for p in rel_parts):
            continue
        if path.name in _SKIP_FILES:
            continue
        out.append(path)
    return out


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_and_write(scan_root: str | Path, storage: FlatFileStorage | None = None) -> dict[str, Any]:
    """Scan ``scan_root`` for python files, extract, and write the artifacts.

    Returns a small summary dict (file count, symbol count, skipped) for the caller.
    Writes three artifacts via the storage seam:
      - ``snapshot.json``   — per-file {mtime,size,sha256} fingerprint (freshness).
      - ``manifest.json``   — per-file slim ``FileInfo`` (the search surface).
      - ``structure.json``  — per-file full symbol tree with spans/signatures/bodies.
    """
    root = Path(scan_root).resolve()
    storage = storage or FlatFileStorage()

    files = iter_python_files(root)

    manifests: list[dict[str, Any]] = []
    structures: dict[str, Any] = {}
    snap_files: dict[str, Any] = {}
    total_symbols = 0
    skipped: list[str] = []

    for path in files:
        rel = path.relative_to(root).as_posix()
        fi = analyze_file(path, root)
        if fi is None:
            skipped.append(rel)
            continue
        st = path.stat()
        snap_files[rel] = {
            "size": st.st_size,
            "mtime": st.st_mtime,
            "sha256": _hash_file(path),
        }
        total_symbols += len(fi.symbols)
        manifests.append(asdict(fi))
        structures[rel] = {
            "path": rel,
            "language": fi.language,
            "doc": fi.doc,
            "symbols": [asdict(s) for s in fi.searchable_symbols],
        }

    storage.write(ArtifactKind.SNAPSHOT, root, {"files": snap_files})
    storage.write(ArtifactKind.MANIFEST, root, {"files": manifests})
    storage.write(ArtifactKind.STRUCTURE, root, {"files": structures})

    from ._store import index_dir_for_scan_root  # public location helper

    return {
        "scan_root": str(root),
        "python_files_found": len(files),
        "extracted": len(manifests),
        "skipped": skipped,
        "total_symbols": total_symbols,
        "index_dir": str(index_dir_for_scan_root(root)),
    }


def load_manifest(storage: FlatFileStorage, scan_root: str | Path) -> list[Any] | None:
    """Convenience read of the manifest artifact (as written by ``scan_and_write``)."""
    return storage.read(ArtifactKind.MANIFEST, scan_root)
