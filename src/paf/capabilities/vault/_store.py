"""Flat-file index store for the vault capability (rides the shared seam).

A small, inspectable, editable index of a vault's notes — the flat-file-first
philosophy locked across paf (memory, codebase_map). The index is a single JSON
file plus a snapshot (per-note fingerprint) for freshness, stored under the paf
repo's ``data/.paf-vault-index/`` (like codebase_map's ``.agent-codeindex/`` but
decoupled from the vault tree):

- ``<vault_index_dir>/<sha256(vault_root)>/index.json``
  — per-note rows: path, frontmatter meta (title/tags/aliases), heading list,
  wikilinks-out, line count.
- ``.../snapshot.json`` — ``{path: {mtime, size, sha256}}`` freshness
  fingerprints.

This storage **delegates to the shared ``FlatFileStorage`` seam**
(``paf.capabilities._storage``) — it does NOT carry its own atomic-write /
JSON helpers. The shared seam owns the mkstemp-safe atomic write that
concurrent single-file editors need, and the same ``StorageService`` backend
(flat file today, SQLite/Postgres later) swaps in with zero vault changes.

The index is keyed by the vault root's normalized absolute path and lives in the
paf repo (NOT co-located inside the vault), so pointing ``vault.root`` at a real
Obsidian vault never drops artifacts into that tree. ``gitignore`` via the
``data`` entry. An explicit ``index_root=`` overrides the location.

Confinement mirrors FileSystem: ``ScanScope`` walks only ``*.md`` under the vault
root, honoring ``allow`` (if non-empty, only matching subtrees) and ``deny``
(matching subtrees always excluded). Symlinks are resolved before authorization.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paf.capabilities._storage._atomic import read_json, write_atomic, write_json_atomic
from paf.config import PROJECT_ROOT

from ._markdown import extract_wikilinks, parse_frontmatter

# Back-compat alias so `_toolset.py` keeps importing `_write_atomic` here; it is
# the SHARED mkstemp-safe atomic write, not a private copy.
_write_atomic = write_atomic

_VERSION = 1

# Repo-owned index root (relative to PROJECT_ROOT), keyed per vault root by hash.
# Lives at the capability level, not in global Settings — see the module docstring.
_DEFAULT_VAULT_INDEX_DIR = "data/.paf-vault-index"


@dataclass
class ScanScope:
    """Determines which ``*.md`` files are part of the vault (allow/deny + root guard)."""

    root: Path
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)

    def _relative_parts(self, path: Path) -> list[str]:
        try:
            return list(path.relative_to(self.root).parts)
        except ValueError:
            return []

    def _matches_any(self, rel_posix: str, patterns: list[str]) -> bool:
        import fnmatch

        if not patterns:
            return False
        return any(
            fnmatch.fnmatch(rel_posix, pat)
            or fnmatch.fnmatch(Path(rel_posix).name, pat)
            for pat in patterns
        )

    def _allowed(self, path: Path) -> bool:
        rel = path.relative_to(self.root).as_posix()
        return self._allowed_rel(rel)

    def _allowed_rel(self, rel_posix: str) -> bool:
        """Scope check on a vault-relative POSIX path (used for prospective writes)."""
        if self.deny and self._matches_any(rel_posix, self.deny):
            return False
        if self.allow:
            return self._matches_any(rel_posix, self.allow)
        return True

    def iter_notes(self) -> list[Path]:
        """Stable-ordered, in-scope ``*.md`` files under the root."""

        def _walk(d: Path) -> list[Path]:
            out: list[Path] = []
            for p in sorted(d.iterdir()):
                if p.is_symlink():
                    # resolve before authorizing; dead/escaping links skipped
                    try:
                        resolved = p.resolve()
                        resolved.relative_to(self.root)
                    except (OSError, ValueError):
                        continue
                    if p.is_dir():
                        out.extend(_walk(p))
                    elif p.suffix == ".md":
                        out.append(p)
                elif p.is_dir():
                    out.extend(_walk(p))
                elif p.suffix == ".md":
                    out.append(p)
            return out

        notes = [p for p in _walk(self.root) if self._allowed(p)]
        notes.sort(key=lambda p: p.relative_to(self.root).as_posix())
        return notes


def index_dir_for_root(root: str | Path) -> Path:
    """Index dir for a vault root: repo-owned, keyed by the root's hash.

    Lives under ``<data/vault_index_dir>/<sha256(root)>/`` in the paf repo —
    NOT co-located inside the vault tree — so pointing a vault at a real
    Obsidian vault or any per-project/temp tree never drops artifacts into it.
    ``FlatFileVaultStorage(index_root=...)`` overrides this default.
    """
    h = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    return Path(PROJECT_ROOT) / Path(_DEFAULT_VAULT_INDEX_DIR) / h


class FlatFileVaultStorage:
    """Default vault index storage: flat JSON via the SHARED atomic seam.

    ``kind`` = ``index`` / ``snapshot``; artifacts live at
    ``index_dir_for_root(root)`` (repo-owned, hash-keyed by vault root). Reads/
    writes use the shared ``_storage._atomic`` helpers (mkstemp-safe atomic
    write) — no private copies. Reads/writes never raise on a missing/corrupt
    artifact: ``read_*`` returns ``None`` so callers treat "not indexed yet"
    and "corrupt once" the same.
    """

    def __init__(self, index_root: str | Path | None = None) -> None:
        self._override: Path | None = Path(index_root).resolve() if index_root else None
        self._base: Path | None = self._override

    def _dir(self, root: str | Path) -> Path:
        """The directory holding this vault's index/snapshot artifacts.

        An explicit ``index_root`` is used verbatim (flat ``index.json`` /
        ``snapshot.json`` at its root, matching the original behavior);
        otherwise the repo-owned, hash-keyed ``index_dir_for_root(root)``.
        """
        if self._base is not None:
            return self._base
        return index_dir_for_root(root)

    def read_index(self, root: str | Path) -> dict | None:
        return read_json(self._dir(root) / "index.json")

    def write_index(self, root: str | Path, payload: dict) -> None:
        write_json_atomic(self._dir(root) / "index.json", payload)

    def read_snapshot(self, root: str | Path) -> dict | None:
        return read_json(self._dir(root) / "snapshot.json")

    def write_snapshot(self, root: str | Path, payload: dict) -> None:
        write_json_atomic(self._dir(root) / "snapshot.json", payload)

    def sync_changed(self, root: str | Path, scope: ScanScope) -> set[str]:
        """Which in-scope notes changed since the last snapshot (stat gate, hash confirm).

        Returns the set of vault-relative paths whose content changed. The
        snapshot fingerprints are stored via the shared atomic seam; this
        freshness check is a vault *retrieval* concern (reading source content),
        so it stays local to the vault — it is not generic storage.
        """
        snapshot = self.read_snapshot(root) or {}
        files = snapshot.get("files", {}) if isinstance(snapshot, dict) else {}
        confirmed: set[str] = set()
        for path in scope.iter_notes():
            rel = path.relative_to(Path(root).resolve()).as_posix()
            try:
                st = path.stat()
            except OSError:
                continue
            prev = files.get(rel) if isinstance(files, dict) else None
            if (
                prev is not None
                and prev.get("size") == st.st_size
                and prev.get("mtime") == st.st_mtime
            ):
                continue
            if self._hash_file(path) != (prev.get("sha256") if prev else None):
                confirmed.add(rel)
        return confirmed

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


# --- Index building ----------------------------------------------------------


def build_index(
    root: str | Path, scope: ScanScope, storage: FlatFileVaultStorage
) -> dict:
    """(Re)build the full index for a vault: per-note rows + fresh snapshot.

    Pure derivation from the in-scope walk. Returns the ``index.json`` payload;
    also writes the paired snapshot so subsequent ``sync_changed`` has a baseline.
    """
    import re

    notes: list[dict] = []
    snapshot_files: dict[str, dict] = {}

    for path in scope.iter_notes():
        rel = path.relative_to(Path(root).resolve()).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_frontmatter(text)
        headings = re.findall(r"^#{1,6}\s+(.+)$", body, re.MULTILINE)
        try:
            st = path.stat()
        except OSError:
            continue
        snapshot_files[rel] = {
            "mtime": st.st_mtime,
            "size": st.st_size,
            "sha256": storage._hash_file(path),
        }
        notes.append(
            {
                "path": rel,
                "title": meta.get("title") or Path(rel).stem,
                "tags": _as_list(meta.get("tags")),
                "aliases": _as_list(meta.get("aliases")),
                "headings": headings[:50],
                "links": extract_wikilinks(body),
                "line_count": len(text.splitlines()),
            }
        )

    payload = {
        "version": _VERSION,
        "root": str(Path(root).resolve()),
        "note_count": len(notes),
        "notes": notes,
    }
    storage.write_index(root, payload)
    storage.write_snapshot(root, {"files": snapshot_files})
    return payload


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return []


def load_index_notes(index: dict | None) -> list[dict]:
    """The per-note rows from an index payload (always a list, possibly empty)."""
    if not isinstance(index, dict):
        return []
    rows = index.get("notes")
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
