"""Scan driver — walk a repo and write the codebase-map artifacts (Phase 1→2 bridge).

This is the missing orchestration that ties the two existing pieces together:
``analyze_file`` (extraction, Phase 1) + ``FlatFileStorage`` (storage seam, Phase 0).
It walks a scan root, extracts every supported file, and persists the results to the
co-located index dir so the artifact files exist **on disk** and are inspectable
(``index.json`` = per-file lean HEAD, ``manifest.json`` = full detail,
``tokens.json`` = the symbol-scoped search backend, ``snapshot.json`` = the
freshness fingerprint ``sync_changed`` keys off, ``overview.json`` = the navigation
tree).

**Scope model (spec §5.y, LOCKED 2026-08-22):** the walker covers EVERYTHING that
isn't ignored — the corpus = files under the scan root minus (gitignore ∪ owned-skip),
via ``ScanScope``. It is deliberately *language-agnostic*: the walk never hardcodes an
extension. Language is an orthogonal extractor axis handled by ``analyze_file`` (per-file
extension → language → extractor). Files that are in scope but yield no parseable
symbols (unknown/oversized/unparseable) are still **recorded** as ``unparsed: true``
records and snapshotted as *seen*, so the corpus is complete and they stay stable rather
than reading as "changed" on every freshness pass.

**Artifact split (LOCKED read-model, 2026-08-23):** one extraction is fanned out into
three layers so the model climbs the ladder instead of paying for detail it doesn't use:
  - ``index.json``   — the **lean HEAD**: per-file metadata + module-docstring head +
    symbol NAME LISTS only. Never carries per-symbol body/signature/doc/span.
  - ``manifest.json`` — the **DETAILED layer**: full per-symbol records
    (docstring/signature/body/span) + imports. What ``lookup_file(detail=True)`` reads.
  - ``tokens.json``   — the **search backend**: inverted, symbol-scoped postings
    (name+sig+doc only). What ``search_codebase`` scores.
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
from ._overview import build_overview
from ._schema import FileInfo, path_rel_to_scan_root
from ._store import ArtifactKind, FlatFileStorage, ScanScope, StorageService
from ._tokens import build_tokens

# Unparsed files still get a language tag when we can guess one (for later
# per-language extractor wiring); unknown extensions get None, recorded as unparsed.
_INDEX_SUBDIR = ".agent-codeindex"

# Kind buckets for the lean HEAD's name lists (spec §3.2: classes + functions split).
# Everything else (Constant/Variable/module-level assignments) still appears in the
# combined `symbols` name list; these are just the two name lists the head carries.
_CLASS_KINDS = {"CLASS", "Class"}
_FUNCTION_KINDS = {"FUNCTION", "Function", "METHOD", "Method"}


def walk_all_files(root: str | Path) -> list[Path]:
    """Stable-ordered list of in-scope files under ``root`` (everything not ignored).

    Scope = gitignore ∪ owned-skip via ``ScanScope``; language-agnostic by design.
    """
    return ScanScope(root).iter_files()


def head_record(fi: FileInfo) -> dict[str, Any]:
    """The LEAN ``index.json`` head row for one rich ``FileInfo`` (spec §3.2).

    Metadata + module-docstring head + symbol NAME LISTS only. No body, no signature,
    no per-symbol doc, no span, no ``role``/``tags`` — those live in ``manifest.json``
    (detail) / are tokenized into ``tokens.json`` (search). ``classes`` and
    ``functions`` are name lists bucketed by symbol kind; the flat ``symbols`` list is
    the full name set.
    """
    classes: list[str] = []
    functions: list[str] = []
    for s in fi.searchable_symbols:
        if not s.name:
            continue
        k = s.kind or ""
        if k in _CLASS_KINDS:
            classes.append(s.name)
        elif k in _FUNCTION_KINDS:
            functions.append(s.name)
    return {
        "path": fi.path,
        "language": fi.language,
        "doc": fi.doc,
        "imports": list(fi.imports),
        "classes": classes,
        "functions": functions,
        "symbols": list(fi.symbols),
        "references_count": fi.references_count,
        "unparsed": fi.unparsed,
    }


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


def scan_and_write(scan_root: str | Path, storage: StorageService | None = None) -> dict[str, Any]:
    """Scan ``scan_root`` for in-scope files, extract, and write the artifacts.

    Returns a small summary dict (file count, symbol count, skipped, unparsed) for the
    caller. Writes four artifacts via the storage seam:
      - ``snapshot.json`` — per-file {mtime,size,sha256} fingerprint (freshness).
      - ``index.json``    — per-file lean HEAD (metadata + doc head + symbol NAME lists).
      - ``manifest.json`` — per-file DETAILED symbol tree (docstring/signature/body/span) + imports.
      - ``tokens.json``   — inverted, symbol-scoped search index (name+sig+doc postings).
    Every in-scope file appears in all four — parsed files carry symbols, and files
    with no extractable symbols appear as ``unparsed: true`` records (still snapshotted
    as seen, so they never read as "changed").
    """
    root = Path(scan_root).resolve()
    storage = storage or FlatFileStorage()

    files = walk_all_files(root)

    index_rows: list[dict[str, Any]] = []   # lean HEAD rows
    manifests: dict[str, Any] = {}          # path -> DETAILED record
    snap_files: dict[str, Any] = {}
    rich: list[FileInfo] = []               # in-memory records (tokens + refs source)
    sources: dict[str, str] = {}            # repo-relative path -> raw source (graph edges)
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
        rich.append(extracted)
        sources[rel] = path.read_text(encoding="utf-8", errors="replace")

        # Lean HEAD (name lists only) + DETAILED manifest (full symbol records + imports).
        index_rows.append(head_record(extracted))
        manifests[rel] = {
            "path": rel,
            "language": extracted.language,
            "doc": extracted.doc,
            "imports": list(extracted.imports),
            "unparsed": extracted.unparsed,
            "symbols": [asdict(s) for s in extracted.searchable_symbols],
        }

    # Populate inbound references_count over the **manifest** records (they carry the
    # full symbol text reference-counting walks). Computed at scan time over the full
    # corpus — free + never stale at full-scan scale. The counts then land on both the
    # manifest record and the lean index HEAD row (matched by path).
    from ._graph import apply_references

    manifest_rows = list(manifests.values())
    apply_references(manifest_rows)
    refs_by_path = {m["path"]: m["references_count"] for m in manifest_rows}
    for row in index_rows:
        row["references_count"] = refs_by_path.get(row["path"], 0)

    storage.write(ArtifactKind.SNAPSHOT, root, {"files": snap_files})
    storage.write(ArtifactKind.INDEX, root, {"files": index_rows})
    storage.write(ArtifactKind.MANIFEST, root, {"files": manifests})
    storage.write(ArtifactKind.TOKENS, root, build_tokens(rich))
    storage.write(ArtifactKind.OVERVIEW, root, build_overview(root))

    # Persist the symbol-bearing edge set (reference + Phase-1.2 call refinement) so
    # get_repo_map / distilled_map read it as a first-class artifact instead of re-parsing.
    from ._graph import build_edges

    storage.write(ArtifactKind.GRAPH, root, {"edges": build_edges(rich, sources)})

    from ._store import index_dir_for_scan_root  # public location helper

    return {
        "scan_root": str(root),
        "files_found": len(files),
        "extracted": len(index_rows),
        "unparsed": unparsed,
        "skipped": skipped,
        "total_symbols": total_symbols,
        "index_dir": str(index_dir_for_scan_root(root)),
    }


def load_manifest(storage: FlatFileStorage, scan_root: str | Path) -> list[Any] | None:
    """Convenience read of the manifest artifact (as written by ``scan_and_write``)."""
    return storage.read(ArtifactKind.MANIFEST, scan_root)
