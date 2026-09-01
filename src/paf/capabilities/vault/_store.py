"""Flat-file index store for the vault capability.

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

The index is keyed by the vault root's normalized absolute path and lives in the
paf repo (NOT co-located inside the vault), so pointing ``vault.root`` at a real
Obsidian vault — or a per-project/temp vault anywhere — never drops artifacts
into that tree. ``gitignore`` via the ``data`` entry. An explicit
``FlatFileVaultStorage(index_root=...)`` overrides this default location.

Everything is deterministic, hand-editable, and diffable. No opaque DB, no
embeddings, no LLM in the read path.

Confinement mirrors FileSystem: ``ScanScope`` walks only ``*.md`` under the vault
root, honoring ``allow`` (if non-empty, only matching subtrees) and ``deny``
(matching subtrees always excluded). Symlinks are resolved before authorization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from paf.config import PROJECT_ROOT, settings

from ._markdown import extract_wikilinks, parse_frontmatter

_VERSION = 1


def _write_atomic(path: str | Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (unique temp + os.replace).

    Thread-safe with respect to concurrent writers: each caller uses a
    process-unique temp name (``tempfile.mkstemp``) so two threads writing the
    SAME target can never steal each other's staging file mid-replace. A fixed
    ``.name.staging`` name raced — one thread's ``os.replace`` could consume the
    staging file another thread was about to move, raising ``FileNotFoundError``.
    The temp is removed on any failure so no ``*.tmp`` debris is left behind.
    """
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _write_json_atomic(path: str | Path, payload: object) -> None:
    _write_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _read_json(path: Path, default: object = None) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


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
        return any(fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(Path(rel_posix).name, pat) for pat in patterns)

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


class VaultStorage(Protocol):
    """The byte-medium seam: where the index/snapshot live and how notes read.

    Mirrors codebase_map's ``StorageService`` so a future SQLite/object-store
    backend can swap in behind the same capability surface.
    """

    def read_index(self, root: Path) -> dict | None: ...
    def write_index(self, root: Path, payload: dict) -> None: ...
    def read_snapshot(self, root: Path) -> dict | None: ...
    def write_snapshot(self, root: Path, payload: dict) -> None: ...
    def sync_changed(self, root: Path, scope: ScanScope) -> set[str]: ...


def _sha256_of_path(root: str | Path) -> str:
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]


def index_dir_for_root(root: str | Path) -> Path:
    """Index dir for a vault root: repo-owned, keyed by the root's hash.

    Lives under ``<data/vault_index_dir>/<sha256(root)>/`` in the paf repo —
    NOT co-located inside the vault tree — so pointing a vault at a real
    Obsidian vault or any per-project/temp tree never drops artifacts into it.
    ``FlatFileVaultStorage(index_root=...)`` overrides this default.
    """
    return Path(PROJECT_ROOT) / Path(settings.vault_index_dir) / _sha256_of_path(root)


class FlatFileVaultStorage:
    """Default storage: flat JSON under the repo-owned index dir (see class module).

    Reads/writes never raise on a missing/corrupt artifact — ``read_*`` returns
    ``None`` so callers treat "not indexed yet" and "corrupt once" the same.
    """

    def __init__(self, index_root: str | Path | None = None) -> None:
        self._override: Path | None = Path(index_root).resolve() if index_root else None

    def _dir(self, root: Path) -> Path:
        return self._override or index_dir_for_root(root)

    def read_index(self, root: Path) -> dict | None:
        data = _read_json(self._dir(root) / "index.json")
        return data if isinstance(data, dict) else None

    def write_index(self, root: Path, payload: dict) -> None:
        _write_json_atomic(self._dir(root) / "index.json", payload)

    def read_snapshot(self, root: Path) -> dict | None:
        data = _read_json(self._dir(root) / "snapshot.json")
        return data if isinstance(data, dict) else None

    def write_snapshot(self, root: Path, payload: dict) -> None:
        _write_json_atomic(self._dir(root) / "snapshot.json", payload)

    def sync_changed(self, root: Path, scope: ScanScope) -> set[str]:
        snapshot = self.read_snapshot(root) or {}
        files = snapshot.get("files", {}) if isinstance(snapshot, dict) else {}
        confirmed: set[str] = set()
        for path in scope.iter_notes():
            rel = path.relative_to(root).as_posix()
            try:
                st = path.stat()
            except OSError:
                continue
            prev = files.get(rel) if isinstance(files, dict) else None
            if prev is not None and prev.get("size") == st.st_size and prev.get("mtime") == st.st_mtime:
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

_HEADING_RE_PATTERN: Any = None  # lazy import guard (re used inline in build)


def build_index(root: Path, scope: ScanScope, storage: FlatFileVaultStorage) -> dict:
    """(Re)build the full index for a vault: per-note rows + fresh snapshot.

    Pure derivation from the in-scope walk. Returns the ``index.json`` payload;
    also writes the paired snapshot so subsequent ``sync_changed`` has a baseline.
    """
    import re

    notes: list[dict] = []
    snapshot_files: dict[str, dict] = {}

    for path in scope.iter_notes():
        rel = path.relative_to(root).as_posix()
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
        "root": str(root),
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
