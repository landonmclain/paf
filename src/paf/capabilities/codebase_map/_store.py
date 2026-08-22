"""Storage seam for the codebase-map capability — the one seam everything hangs off.

``StorageService`` is the *byte-medium* seam: it decides **where artifacts live** (flat
JSON files under ``<scan_root>/.agent-codeindex/codebase_indexes/<sha256(scan_root)>/``
today; SQLite / object-store / MCP later). It is **not** a search seam — *how* queries
are answered is ``_retrieval.py``'s job. Swapping storage does not make search semantic.

``FlatFileStorage`` is the default impl. Requirements locked 2026-08-22:

- **Root = scan_root, co-located** — the index lives under the scanned repo root at
  ``.agent-codeindex/`` (like ``.git/``), NOT in a separate paf ``directory``. Decided.
- **Hash lives ONLY in ``snapshot.json``** — per-file ``{mtime, size, sha256}``; the
  per-file records in structure/manifest/tokens carry no ``content_hash``. Decided.
- **Freshness: on-query lazy** — ``sync_changed`` does a cheap ``stat`` walk, sha256's
  only the files whose mtime/size differ (stat = gate, hash = confirm).
- **Graph/reference counts are derived on read** (§5.x) — ``FlatFileStorage`` never
  persists a global ``call_graph.json`` that can go stale.
- **``read_block`` is "query-the-pointer"** — heavy source is touched only for the exact
  byte span a ``BlockRef`` names.
"""
from __future__ import annotations

import hashlib
import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from ._schema import BlockRef

# --- Inlined atomic-write helpers (verbatim from ``memory._atomic``) ---------
# Kept in-capability so codebase_map has zero coupling to the memory capability
# (the storage seam must be self-contained: swap the medium, not the neighbor).
# A plain ``Path.write_text`` on a long JSON artifact can be interrupted midway,
# leaving a torn file a later read fails on; staging + ``os.replace`` makes the
# write all-or-nothing on the same filesystem.

def _write_atomic(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.staging")
    staging.write_text(content, encoding="utf-8")
    os.replace(staging, path)


def _write_json_atomic(path: str | Path, payload: object) -> None:
    _write_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _read_json(path: str | Path, default: object = None) -> object:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


class ArtifactKind(StrEnum):
    """The canonical artifact names the storage seam reads/writes by."""

    SNAPSHOT = "snapshot"  # snapshot.json    — per-file fingerprint {mtime,size,sha256} (freshness)
    STRUCTURE = "structure"  # structure.json  — per-file symbol/import summary
    MANIFEST = "manifest"  # manifest.json   — per-file slim FileInfo (search surface)
    SEARCH = "search"  # search_index.json — flattened keyword index rows
    TOKENS = "tokens"  # tokens.json     — inverted source-token index
    GRAPH = "graph"  # call_graph.json — caller→callee edges (DERIVED on read, §5.x)
    MAP = "map"  # distilled_map.json — ranked distilled map (optional cache)


# kind -> file basename (kept here so the mapping is one obvious place).
_ARTIFACT_FILENAMES: dict[ArtifactKind, str] = {
    ArtifactKind.SNAPSHOT: "snapshot.json",
    ArtifactKind.STRUCTURE: "structure.json",
    ArtifactKind.MANIFEST: "manifest.json",
    ArtifactKind.SEARCH: "search_index.json",
    ArtifactKind.TOKENS: "tokens.json",
    ArtifactKind.GRAPH: "call_graph.json",
    ArtifactKind.MAP: "distilled_map.json",
}

_INDEX_SUBDIR = ".agent-codeindex"
_CODEDEX_DIR = "codebase_indexes"


class StorageService(Protocol):
    """The byte-medium seam: how artifacts are read/written and pointers resolved.

    Reimplement this (e.g. ``SqliteStorage``) to move where artifacts live behind the
    same capability — analyzers/retrieval/tool surface never change.
    """

    def read(self, kind: ArtifactKind, scan_root: Path) -> Any | None: ...
    def write(self, kind: ArtifactKind, scan_root: Path, payload: Any) -> None: ...
    def list_scan_roots(self) -> list[Path]: ...
    def read_block(self, block_ref: BlockRef) -> str | None: ...
    def sync_changed(self, scan_root: Path) -> set[str]: ...


def _sha256_of_path(root: str | Path) -> str:
    """sha256 of the *directory path* — used only to namespace the index folder.

    This is NOT a content hash: it fingerprints *where* the scan root is, never what's
    inside it, so it never changes on edit. Per-file content hashes live in
    ``snapshot.json`` and are what drive freshness.
    """
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()


def index_dir_for_scan_root(root: str | Path) -> Path:
    """Co-located index dir: ``<scan_root>/.agent-codeindex/codebase_indexes/<sha256(root)>/``."""
    root = Path(root)
    return root / _INDEX_SUBDIR / _CODEDEX_DIR / _sha256_of_path(root)


class FlatFileStorage:
    """Default storage backend: flat JSON under the co-located index dir (atomic writes).

    Reads/writes never raise on a missing/corrupt artifact — ``read`` returns ``None``
    so callers treat "not indexed yet" and "corrupt once" the same way.
    """

    def __init__(self, index_root: str | Path | None = None) -> None:
        # override for tests: point at a fixed temp dir instead of deriving from scan_root
        self._index_root_override: Path | None = Path(index_root) if index_root else None

    # -- location ------------------------------------------------------------
    def _dir_for(self, scan_root: str | Path) -> Path:
        return self._index_root_override or index_dir_for_scan_root(scan_root)

    def _path_for(self, kind: ArtifactKind, scan_root: str | Path) -> Path:
        return self._dir_for(scan_root) / _ARTIFACT_FILENAMES[kind]

    # -- StorageService protocol --------------------------------------------
    def read(self, kind: ArtifactKind, scan_root: str | Path) -> Any | None:
        return _read_json(self._path_for(kind, scan_root))

    def write(self, kind: ArtifactKind, scan_root: str | Path, payload: Any) -> None:
        path = self._path_for(kind, scan_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(path, payload)

    def list_scan_roots(self) -> list[Path]:
        """List scan roots that have at least one artifact in their co-located index dir.

        NOTE: with a purely co-located (non-central) index there is no single registry
        to enumerate from — discovery would require walking every candidate repo. v1 is
        single-repo nav, so this returns the override/empty set and is mainly exercised
        by tests. A cross-repo ``~/projects`` discovery pass is future work.
        """
        if self._index_root_override is not None:
            return [self._index_root_override]
        return []

    def read_block(self, block_ref: BlockRef) -> str | None:
        """Return the exact source bytes for ``block_ref.span`` (bounded read).

        The pointer model: a hit collapses to a ``BlockRef``; the heavy file is touched
        here, and only for the range named. ``read_block`` (in ``_block.py``) handles
        the byte-slice and falls back to a line/whole read when offsets are absent.
        Returns ``None`` if the file doesn't exist.
        """
        from ._block import read_block  # local import to avoid a cycle with _schema

        return read_block(block_ref.file, block_ref.span)

    def sync_changed(self, scan_root: str | Path) -> set[str]:
        """Freshness pass: return repo-relative paths whose *content* changed.

        1. cheap ``stat`` walk over ``scan_root`` → shortlist files whose mtime/size
           differ from ``snapshot.json``;
        2. sha256 ONLY the shortlist to confirm real content change (stat = gate,
           hash = confirm — catches mtime-preserving edits like ``git checkout``);
        3. the confirmed set is what callers re-extract.
        """
        snapshot = self.read(ArtifactKind.SNAPSHOT, scan_root) or {}
        files = snapshot.get("files", {})

        scan_root_p = Path(scan_root)
        confirmed: set[str] = set()

        if not scan_root_p.exists():
            return confirmed

        for path in self._iter_source_files(scan_root_p):
            rel = path.relative_to(scan_root_p).as_posix()
            try:
                st = path.stat()
            except OSError:
                continue
            prev = files.get(rel)
            if prev is not None and prev.get("size") == st.st_size and prev.get("mtime") == st.st_mtime:
                continue  # stat gate: unchanged → skip hashing
            if self._hash_file(path) != (prev.get("sha256") if prev else None):
                confirmed.add(rel)

        return confirmed

    # -- helpers -------------------------------------------------------------
    def _iter_source_files(self, root: Path):
        # YAGNI: v1 walks everything except the index dir + git internals. Expand the
        # ignore set with .gitignore rules in a later phase if needed.
        ignore = {_INDEX_SUBDIR, ".git"}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.parent.name in ignore or any(p in ignore for p in path.relative_to(root).parts):
                continue
            yield path

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
