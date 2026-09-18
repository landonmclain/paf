"""Default :class:`AppendOnlyStore` backend: append-only JSONL under a root dir.

Files are ``<root>/<key>/<kind>.jsonl`` (key scoped into its own subdirectory,
kind becomes the filename) — the same layout as :class:`FlatFileStorage`, just
with an append-only-on-JSONL contract instead of whole-artifact JSON.

Append semantics mirror the journal's ``_append_block_atomic`` (generalized to
the shared package): read current file, append new JSON records, write the
whole thing back atomically (staging + ``os.replace``). A reader therefore
sees either the prior complete log or the new complete log — never a torn
line — which is why a crash can't corrupt an append-only log. Deliberately NOT
a bare ``open(..., 'a')`` append, which could leave a partial trailing line.

A future backend (SQLite) backs ``append`` with an INSERT and ``read_all`` with
a SELECT — same seam, swappable medium.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from paf.capabilities._storage._atomic import write_atomic
from paf.capabilities._storage._flat_file import _safe_key_segment


def _append_jsonl_atomic(path: str | Path, records: list[dict]) -> None:
    """Append ``records`` as JSON lines to ``path`` atomically (see module doc)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
    parts = existing
    for rec in records:
        parts += json.dumps(rec, ensure_ascii=False) + "\n"
    write_atomic(path, parts)


class FlatJsonlStore:
    """Default :class:`AppendOnlyStore`: append-only JSONL under a root dir."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # -- location -----------------------------------------------------------
    def _path_for(self, kind: str, key: str) -> Path:
        return self.root / _safe_key_segment(key) / f"{kind}.jsonl"

    # -- AppendOnlyStore protocol -------------------------------------------
    def append(self, kind: str, key: str, record: dict) -> dict:
        with self._lock:
            _append_jsonl_atomic(self._path_for(kind, key), [record])
        return record

    def read_all(self, kind: str, key: str) -> list[dict]:
        path = self._path_for(kind, key)
        if not path.exists():
            return []
        with self._lock:
            out = []
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        # A torn/corrupt line is skipped — the log's current
                        # state is always derivable from the good records.
                        continue
            return out
