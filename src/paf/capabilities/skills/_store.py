"""Flat-file index store for the skills capability (rides the shared seam).

A small, inspectable, editable index of a skill library — the flat-file-first
philosophy locked across paf (memory, codebase_map, vault). The index is a
single JSON file plus a snapshot (per-skill fingerprint) for freshness, stored
under the paf repo's ``data/.skills-index/`` (mirroring the vault's
``data/.paf-vault-index/``, decoupled from the skill library tree):

- ``<skills_index_dir>/<sha256(library_root)>/index.json``
  — per-skill rows: the known label set (skill_id, name, category, description,
  tags, hermes_tags, keywords, hint) plus a resources manifest. No BM25
  postings or embeddings — the labels are the searchable vocabulary.
- ``.../snapshot.json`` — ``{path: {mtime, size, sha256}}`` freshness
  fingerprints, used only for lazy staleness detection (stat gate, hash
  confirm).

This storage **delegates to the shared ``FlatFileStorage`` seam**
(``paf.capabilities._storage``) — it does NOT carry its own atomic-write / JSON
helpers. The shared seam owns the mkstemp-safe atomic write and the same
``StorageService`` backend (flat file today, SQLite/Postgres later) swaps in
with zero skills changes.

The index is keyed by the library root's normalized absolute path and lives in
the paf repo (NOT co-located inside the skill library), so pointing ``skills``
at a real skill tree never drops artifacts into it. ``gitignore`` via the
``data`` entry. An explicit ``index_root=`` overrides the location.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from paf.capabilities._storage._atomic import read_json, write_json_atomic
from paf.config import PROJECT_ROOT

from ._corpus import SkillDoc, parse_frontmatter

_VERSION = 1

# Repo-owned index root (relative to PROJECT_ROOT), keyed per library root by
# hash. Lives at the capability level, not in global Settings — see the module
# docstring.
_DEFAULT_SKILLS_INDEX_DIR = "data/.skills-index"


def index_dir_for_root(root: str | Path) -> Path:
    """Index dir for a skill library root: repo-owned, keyed by the root's hash.

    Lives under ``<data/skills_index_dir>/<sha256(root)>/`` in the paf repo —
    NOT co-located inside the skill library — so pointing skills at a real
    skill tree never drops artifacts into it.
    ``FlatFileSkillsStorage(index_root=...)`` overrides this default.
    """
    h = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    return Path(PROJECT_ROOT) / Path(_DEFAULT_SKILLS_INDEX_DIR) / h


class FlatFileSkillsStorage:
    """Default skills index storage: flat JSON via the SHARED atomic seam.

    ``kind`` = ``index`` / ``snapshot``; artifacts live at
    ``index_dir_for_root(root)`` (repo-owned, hash-keyed by library root).
    Reads/writes use the shared ``_storage._atomic`` helpers (mkstemp-safe
    atomic write) — no private copies. Reads/writes never raise on a
    missing/corrupt artifact: ``read_*`` returns ``None`` so callers treat
    "not indexed yet" and "corrupt once" the same.
    """

    def __init__(self, index_root: str | Path | None = None) -> None:
        self._override: Path | None = Path(index_root).resolve() if index_root else None
        self._base: Path | None = self._override

    def _dir(self, root: str | Path) -> Path:
        """The directory holding this library's index/snapshot artifacts.

        An explicit ``index_root`` is used verbatim (flat ``index.json`` /
        ``snapshot.json`` at its root); otherwise the repo-owned, hash-keyed
        ``index_dir_for_root(root)``.
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

    def _hash_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


# --- Index building ----------------------------------------------------------


def build_index(
    docs: list[SkillDoc], storage: FlatFileSkillsStorage, root: str | Path
) -> dict:
    """(Re)build the full skills index for a library root: per-skill rows + fresh snapshot.

    Pure derivation from the corpus docs that live under ``root`` (skills from
    other roots in a multi-root corpus are skipped — the index is keyed per
    library root). Returns the ``index.json`` payload; also writes the paired
    snapshot so subsequent ``sync_changed`` has a baseline.
    """
    root = Path(root).resolve()
    skills: list[dict] = []
    snapshot_files: dict[str, dict] = {}

    for doc in docs:
        try:
            rel = doc.skill_file.resolve().relative_to(root).as_posix()
        except ValueError:
            continue  # skill outside this library root — not indexed here
        try:
            st = doc.skill_file.stat()
        except OSError:
            continue
        snapshot_files[rel] = {
            "mtime": st.st_mtime,
            "size": st.st_size,
            "sha256": storage._hash_file(doc.skill_file),
        }
        fm = parse_frontmatter(doc.skill_file)
        skills.append(
            {
                "skill_id": doc.skill_id,
                "name": doc.name,
                "category": doc.category,
                "description": fm.get("description", ""),
                "tags": _as_list(fm.get("tags")),
                "hermes_tags": _as_list(fm.get("metadata.hermes.tags")),
                "keywords": _as_list(fm.get("metadata.paf.keywords")),
                "hint": doc.hint,
                "resources": [r.relative_path for r in doc.resources],
            }
        )

    payload = {
        "version": _VERSION,
        "root": str(root),
        "skill_count": len(skills),
        "skills": skills,
    }
    storage.write_index(root, payload)
    storage.write_snapshot(root, {"files": snapshot_files})
    return payload


def sync_changed(root: str | Path, storage: FlatFileSkillsStorage) -> set[str]:
    """Which in-scope SKILL.md files changed since the last snapshot.

    Returns the set of library-root-relative paths whose content changed. The
    snapshot fingerprints are stored via the shared atomic seam; this freshness
    check is a skills *retrieval* concern (reading source content), so it stays
    local to the skills store — it is not generic storage.
    """
    root = Path(root).resolve()
    snapshot = storage.read_snapshot(root) or {}
    files = snapshot.get("files", {}) if isinstance(snapshot, dict) else {}
    confirmed: set[str] = set()
    for skill_file in root.rglob("SKILL.md"):
        try:
            rel = skill_file.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        try:
            st = skill_file.stat()
        except OSError:
            continue
        prev = files.get(rel) if isinstance(files, dict) else None
        if (
            prev is not None
            and prev.get("size") == st.st_size
            and prev.get("mtime") == st.st_mtime
        ):
            continue
        if storage._hash_file(skill_file) != (prev.get("sha256") if prev else None):
            confirmed.add(rel)
    return confirmed


def _as_list(value: Any) -> list[str]:
    """Coerce a frontmatter value to a list of strings.

    ``parse_frontmatter`` flattens inline/block lists to space-joined strings,
    so a string is split on whitespace; a real list is passed through.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [v for v in value.split() if v]
    return []
