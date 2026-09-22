"""Default :class:`StorageService` backend: flat JSON files under a root dir.

Layout is ``<root>/<key>/<kind>.json`` — the artifact ``kind`` becomes the
filename, and ``key`` scopes it into its own subdirectory. Reads/writes never
raise on a missing/corrupt artifact: ``read`` returns ``None`` so callers treat
"not present yet" and "corrupt once" the same way. Writes are atomic
(:func:`.atomic.write_json_atomic`).

This is the flat default of the seam. A future backend (SQLite / Postgres)
reimplements :class:`StorageService` and is dropped in with zero caller changes.
"""

from __future__ import annotations

import os
from pathlib import Path

from paf.capabilities._storage._atomic import read_json, write_json_atomic
from paf.capabilities._storage._block import BlockRef


def _safe_key_segment(key: str | Path) -> str:
    """Coerce ``key`` to a safe, stable single path segment.

    A bare, already-safe name (``demo_assistant``) is kept verbatim — so a
    capability keyed by agent name gets a clean ``<root>/<agent>/`` dir that
    lines up with the rest of the system (e.g. the memory dir). A deep/absolute
    key (a scan root like ``/home/alice/projects/foo``) is hashed so it nests
    cleanly without deep paths. Pure function of the string so tests can
    predict locations.
    """
    import hashlib
    import re

    s = os.fspath(key)
    if not s or s in {".", os.sep}:
        return "_"
    # Bare single-segment, safe names pass through untouched.
    if re.fullmatch(r"[A-Za-z0-9_.-]+", s):
        return s
    # Nested safe path (e.g. "<agent>/conversations/<cid>") passes through so
    # conversation-scoped stores nest cleanly under the agent dir.
    if re.fullmatch(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)+", s):
        return s
    safe = s.strip("/\\").replace(":", "_").replace(os.sep, "_") or "_"
    return f"{safe[:48]}-{hashlib.sha256(s.encode('utf-8')).hexdigest()[:8]}"


class FlatFileStorage:
    """Default :class:`StorageService`: flat JSON under a root dir (atomic writes)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def in_temp(cls) -> FlatFileStorage:
        """A throwaway instance rooted in a fresh temp dir (tests / sandboxes)."""
        import tempfile

        return cls(tempfile.mkdtemp(prefix="paf-store-"))

    # -- location -----------------------------------------------------------
    def _path_for(self, kind: str, key: str | Path) -> Path:
        return self.root / _safe_key_segment(key) / f"{kind}.json"

    # -- StorageService protocol --------------------------------------------
    def read(self, kind: str, key: str | Path):
        return read_json(self._path_for(kind, key))

    def write(self, kind: str, key: str | Path, payload) -> None:
        write_json_atomic(self._path_for(kind, key), payload)

    def list_keys(self) -> list[Path]:
        """Every non-hidden subdirectory of ``root`` (one per distinct key)."""
        out = []
        if not self.root.is_dir():
            return out
        for child in self.root.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                out.append(child)
        return sorted(out)

    def read_block(self, block_ref: BlockRef) -> str | None:
        """Materialise the bounded source range a ``BlockRef`` names.

        The pointer may reference a file stored under this root (via
        ``<root>/<key>/<kind>.json``) or an arbitrary absolute path (source
        files, which a search index points at). Absent spans fall back to the
        whole file capped by ``read_block``'s bound.
        """
        from paf.capabilities._storage._block import read_block as _read_block

        return _read_block(block_ref.file, block_ref.span)
