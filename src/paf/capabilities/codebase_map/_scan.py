"""Scan driver — walk a repo and write the codebase-map artifacts (Phase 1→2 bridge).

This is the missing orchestration that ties the two existing pieces together:
``analyze_file`` (extraction, Phase 1) + ``FlatFileStorage`` (storage seam, Phase 0).
It walks a scan root, extracts every supported file, and persists the results to the
co-located index dir so the artifact files exist **on disk** and are inspectable
(``manifest.json`` = per-file search surface, ``structure.json`` = full symbol tree,
``snapshot.json`` = the freshness fingerprint ``sync_changed`` keys off).

**Scope model (spec §5.y, LOCKED 2026-08-22):** the walker covers EVERYTHING that
isn't ignored — the corpus = files under the scan root minus (gitignore ∪ owned-skip),
via ``ScanScope``. It is deliberately *language-agnostic*: the walk never hardcodes an
extension. Language is an orthogonal extractor axis handled by ``analyze_file`` (per-file
extension → language → extractor). Files that are in scope but yield no parseable
symbols (unknown/oversized/unparseable) are still **recorded** as ``unparsed: true``
records and snapshotted as *seen*, so the corpus is complete and they stay stable rather
than reading as "changed" on every freshness pass.

The driver is deliberately *dumb*: it does not rank, search, or build a graph. It exists
so the data lands on disk and can be eyeballed before any search logic is built on top
(spec ordering: extraction completeness is the gate for search tuning).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ._analyze import analyze_file, detect_language
from ._schema import FileInfo, path_rel_to_scan_root
from ._store import ArtifactKind, FlatFileStorage, ScanScope

# Unparsed files still get a language tag when we can guess one (for later
# per-language extractor wiring); unknown extensions get None, recorded as unparsed.
_INDEX_SUBDIR = ".agent-codeindex"


def walk_all_files(root: str | Path) -> list[Path]:
    """Stable-ordered list of in-scope files under ``root`` (everything not ignored).

    Scope = gitignore ∪ owned-skip via ``ScanScope``; language-agnostic by design.
    """
    return ScanScope(root).iter_files()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _unparsed_record(path: Path, root: Path) -> FileInfo:
    """A minimal record for an in-scope file that yielded no parseable symbols."""
    rel = path_rel_to_scan_root(path, root)
    return FileInfo(
        path=rel,
        language=detect_language(path.name),
        symbols=[],
        imports=[],
        unparsed=True,
    )


def scan_and_write(scan_root: str | Path, storage: FlatFileStorage | None = None) -> dict[str, Any]:
    """Scan ``scan_root`` for in-scope files, extract, and write the artifacts.

    Returns a small summary dict (file count, symbol count, skipped, unparsed) for the
    caller. Writes three artifacts via the storage seam:
      - ``snapshot.json``   — per-file {mtime,size,sha256} fingerprint (freshness).
      - ``manifest.json``   — per-file slim ``FileInfo`` (the search surface).
      - ``structure.json``  — per-file full symbol tree with spans/signatures/bodies.
    Every in-scope file appears in all three — parsed files carry symbols, and files
    with no extractable symbols appear as ``unparsed: true`` records (still snapshotted
    as seen, so they never read as "changed").
    """
    root = Path(scan_root).resolve()
    storage = storage or FlatFileStorage()

    files = walk_all_files(root)

    manifests: list[dict[str, Any]] = []
    structures: dict[str, Any] = {}
    snap_files: dict[str, Any] = {}
    total_symbols = 0
    skipped: list[str] = []
    unparsed: list[str] = []

    for path in files:
        rel = path.relative_to(root).as_posix()
        extracted = analyze_file(path, root)
        if extracted is None:
            extracted = _unparsed_record(path, root)
            unparsed.append(rel)
        st = path.stat()
        snap_files[rel] = {
            "size": st.st_size,
            "mtime": st.st_mtime,
            "sha256": _hash_file(path),
        }
        total_symbols += len(extracted.searchable_symbols)
        manifests.append(asdict(extracted))
        structures[rel] = {
            "path": rel,
            "language": extracted.language,
            "doc": extracted.doc,
            "unparsed": extracted.unparsed,
            "symbols": [asdict(s) for s in extracted.searchable_symbols],
        }

    storage.write(ArtifactKind.SNAPSHOT, root, {"files": snap_files})
    storage.write(ArtifactKind.MANIFEST, root, {"files": manifests})
    storage.write(ArtifactKind.STRUCTURE, root, {"files": structures})

    from ._store import index_dir_for_scan_root  # public location helper

    return {
        "scan_root": str(root),
        "files_found": len(files),
        "extracted": len(manifests),
        "unparsed": unparsed,
        "skipped": skipped,
        "total_symbols": total_symbols,
        "index_dir": str(index_dir_for_scan_root(root)),
    }


def load_manifest(storage: FlatFileStorage, scan_root: str | Path) -> list[Any] | None:
    """Convenience read of the manifest artifact (as written by ``scan_and_write``)."""
    return storage.read(ArtifactKind.MANIFEST, scan_root)
