"""Flat-file/JSON storage backend — the inspectable default.

State lives as human-readable JSON files (gitignored at runtime) so both the
agent and a human can read/edit it. This is the "vault-style" inspectable
artifact layer: state is legible and bidirectional, not a black box.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paf.storage.protocol import Storage


class FlatFileStorage:
    """JSON-file-backed storage under a single directory.

    Each key maps to ``<dir>/<key>.json``. Simple, inspectable, diffable.
    """

    def __init__(self, root: str | Path = "data/state") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Guard against path traversal via key.
        safe = Path(key).name
        return self.root / f"{safe}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def set(self, key: str, value: Any) -> None:
        p = self._path(key)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=2, sort_keys=True)

    def search(self, query: str, limit: int = 50) -> list[str]:
        """Filename substring search over keys."""
        q = query.lower()
        hits = [p.name[: -len(".json")] for p in self.root.glob("*.json") if q in p.name.lower()]
        return hits[:limit]

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()
