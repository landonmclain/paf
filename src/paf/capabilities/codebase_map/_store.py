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
- **Graph persisted at scan time** — ``_scan.py`` writes the symbol-bearing edge
  set as the GRAPH artifact (``call_graph.json``) when it scans, and stamps
  ``references_count`` into the manifest/index HEAD rows. (This supersedes §5.x's
  original "derive on read": recomputing the graph on every read would require a
  re-parse, so the edges are materialized once at scan and read as a first-class
  artifact by ``get_repo_map`` / ``distilled_map``.)
- **``read_block`` is "query-the-pointer"** — heavy source is touched only for the exact
  byte span a ``BlockRef`` names.
"""
from __future__ import annotations

import hashlib
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from ._schema import BlockRef

# --- Atomic-write helpers: shared canonical import ---------------------------
# These were inlined per-capability (verbatim from ``memory._atomic``); the
# canonical copies now live in ``paf.capabilities._storage._atomic`` and this
# capability conforms to them. Behavior is byte-identical (staging + ``os.replace``);
# the storage seam stays self-contained without coupling to any other capability.
from paf.capabilities._storage._atomic import read_json, write_json_atomic


class ArtifactKind(StrEnum):
    """The canonical artifact names the storage seam reads/writes by."""

    SNAPSHOT = "snapshot"  # snapshot.json    — per-file fingerprint {mtime,size,sha256} (freshness)
    OVERVIEW = "overview"  # overview.json    — navigation TREE (folders → files, collapsed from ignores) — START
    MANIFEST = "manifest"  # manifest.json   — per-file DETAILED symbol tree (docstring/signature/body/span)
    INDEX = "index"  # index.json       — per-file HEAD (metadata + docstring head + symbol names)
    SEARCH = "search"  # search_index.json — flattened keyword index rows (future/legacy slot, not yet written; superseded by TOKENS)
    TOKENS = "tokens"  # tokens.json     — inverted symbol-scoped search index (the search backend)
    GRAPH = "graph"  # call_graph.json — caller→callee edges (persisted at scan time, §5.x)
    MAP = "map"  # repo_map.json — ranked zero-LLM orientation map (written by get_repo_map on demand)


# kind -> file basename (kept here so the mapping is one obvious place).
_ARTIFACT_FILENAMES: dict[ArtifactKind, str] = {
    ArtifactKind.SNAPSHOT: "snapshot.json",
    ArtifactKind.OVERVIEW: "overview.json",
    ArtifactKind.MANIFEST: "manifest.json",
    ArtifactKind.INDEX: "index.json",
    ArtifactKind.SEARCH: "search_index.json",
    ArtifactKind.TOKENS: "tokens.json",
    ArtifactKind.GRAPH: "call_graph.json",
    ArtifactKind.MAP: "repo_map.json",
}

_INDEX_SUBDIR = ".agent-codeindex"
_CODEDEX_DIR = "codebase_indexes"

# --- Scan scope: the single source of truth for "what is part of the codebase". ---
# Per §5.y of the spec (LOCKED 2026-08-22): the corpus = every file under the scan
# root minus (gitignore ∪ owned-skip). The walker never hardcodes a language; language
# is an orthogonal extractor axis. `_scan.py` and `sync_changed` BOTH derive from this.

# Owned skip set — *domain* exclusions that are NOT (and arguably shouldn't be) gitignore
# concerns: older content we still don't want indexed, plus safety nets for common junk
# dirs that may not be gitignored (node_modules) or that git always manages anyway.
# Everything git ignores on its own (`.venv/`, `data/`, `tests/results/`, `__pycache__/`)
# is handled by `git check-ignore`; this is only what git would NOT ignore but we still skip.
_OWNED_SKIP_DIRS = {
    _INDEX_SUBDIR,    # our own derived index — never index the index
    ".git",           # git internals
    "__pycache__",    # bytecode (also usually gitignored; belt + suspenders)
    "node_modules",   # vendored js deps (may not be gitignored)
    "History",        # paf's stale-notes dir — older content, not for the index
    "archive",        # paf's old-code dir — scratch/stale, not for the index
    # data/ is TRACKED in git (benchmark durables: generated/*.jsonl, memory-*/,
    # lme_oracle_journal.json) so git won't ignore it — but it's research/harness
    # data, not source. The codebase map indexes CODE; data/ is out of scope here.
    # (In gitignore it's a "runtime persistence" tree; here it's just noise. Owned-skip
    # keeps the walker fast: pure-Python, no need to fight the gitignore re-includes.)
    "data",
}
_OWNED_SKIP_FILES = {"__init__.py"}  # docstring-only + __version__; not indexed (open deferred item)


class ScanScope:
    """Determines whether a path is part of the indexed codebase.

    In bounds  = path is NOT gitignored AND NOT under an owned-skip dir/filename.
    ``git check-ignore`` reads the real git semantics (nested .gitignore, ! negation,
    global excludes). When git isn't available (non-repo / deployed env), we fall back
    to the owned-skip list alone — which is correct, because if there's no git then
    there's no .gitignore to honor anyway.

    Efficiency: owned-skip is a pure-Python fast path (no subprocess). Git ignore is
    applied in ONE batched ``git check-ignore --stdin`` call over the candidate list
    (never a subprocess per file — that would spawn thousands on a real repo).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._git_ok: bool | None = None  # tri-state: None=unknown, True/False probed

    def _probe_git(self) -> bool:
        if self._git_ok is None:
            try:
                r = subprocess.run(
                    ["git", "-C", str(self.root), "rev-parse", "--is-inside-work-tree"],
                    capture_output=True, text=True, timeout=5,
                )
                self._git_ok = r.returncode == 0
            except (OSError, subprocess.SubprocessError):
                self._git_ok = False
        return self._git_ok

    def _owned_skipped(self, p: Path) -> bool:
        """Pure-Python fast path: is this path under an owned-skip dir or filename?"""
        rel_parts = p.relative_to(self.root).parts
        if any(part in _OWNED_SKIP_DIRS for part in rel_parts):
            return True
        if p.name in _OWNED_SKIP_FILES:
            return True
        return False

    def in_scope(self, path: str | Path) -> bool:
        """Single-path scope check (owned-skip fast path + one git check-ignore call).

        Prefer ``iter_files()`` when walking a whole tree (it batches git check-ignore
        into one subprocess); this is for one-off membership tests.
        """
        p = Path(path)
        if self._owned_skipped(p):
            return False
        if not self._probe_git():
            return True
        try:
            r = subprocess.run(
                ["git", "-C", str(self.root), "check-ignore", "--no-index", "-q", str(p)],
                capture_output=True, timeout=5,
            )
            return r.returncode != 0  # 0 = ignored (out of scope)
        except (OSError, subprocess.SubprocessError):
            return True

    def iter_files(self) -> list[Path]:
        """Stable-ordered, in-scope files under the root (everything not ignored).

        Two stages: (1) walk + owned-skip (pure Python), (2) ONE batched
        ``git check-ignore --stdin`` to drop gitignored paths.
        """
        root = self.root
        candidates: list[Path] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                path.relative_to(root)
            except ValueError:
                continue
            if not self._owned_skipped(path):
                candidates.append(path)

        if not candidates or not self._probe_git():
            return candidates  # no git (or no files) -> owned-skip is the only filter

        # Batch git-ignore check: feed every candidate path on stdin; git prints the
        # matched (ignored) ones to stdout. Use --no-index so committed files are still
        # evaluated against rules regardless of tracking state.
        input_data = "".join(f"{p}\n" for p in candidates)
        try:
            r = subprocess.run(
                ["git", "-C", str(root), "check-ignore", "--no-index", "--stdin"],
                input=input_data, capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return candidates  # git hiccuped -> owned-skip only (safe)
        ignored = {line.rstrip("\n") for line in r.stdout.splitlines() if line}
        # returncode 0 = at least one matched; 1 = none. We just parse stdout either way.
        out = [p for p in candidates if str(p) not in ignored]
        return out


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
        return read_json(self._path_for(kind, scan_root))

    def write(self, kind: ArtifactKind, scan_root: str | Path, payload: Any) -> None:
        path = self._path_for(kind, scan_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, payload)

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

        Quirk of the seam: ``BlockRef.file`` reaching us must already be an **absolute**
        path — the caller (the ``read_block`` tool) resolves the repo-relative pointer
        against ``scan_root`` before crossing the seam, so a fully co-located backend
        doesn't need its own scan_root to locate a file.
        """
        from ._block import read_block  # local import to avoid a cycle with _schema

        return read_block(block_ref.file, block_ref.span)

    def sync_changed(self, scan_root: str | Path) -> set[str]:
        """Freshness pass: return repo-relative paths whose *content* changed.

        Scoped to the indexed corpus (see ``ScanScope`` — gitignore ∪ owned-skip):
        a gitignored file (e.g. ``.venv/``) is **out of scope, not "changed"**, so it is
        never walked and never reported. Within scope:

        1. cheap ``stat`` walk → shortlist files whose mtime/size differ from
           ``snapshot.json``;
        2. sha256 ONLY the shortlist to confirm real content change (stat = gate,
           hash = confirm — catches mtime-preserving edits like ``git checkout``);
        3. the confirmed set (in-scope files whose content actually changed, incl.
           newly-appearing in-scope files) is what callers re-extract.
        """
        snapshot = self.read(ArtifactKind.SNAPSHOT, scan_root) or {}
        files = snapshot.get("files", {})

        confirmed: set[str] = set()
        scope = ScanScope(scan_root)
        for path in scope.iter_files():
            rel = path.relative_to(scope.root).as_posix()
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
    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
