"""Atomic file writes: staging file + ``os.replace``.

A plain ``Path.write_text`` on a long JSON artifact can be interrupted midway,
leaving a truncated/half-written file that a later read fails on. Writing to a
sibling ``.staging`` temp then ``os.replace``-ing it into place makes a write
all-or-nothing: observers either see the old complete artifact or the new one,
never a torn write.

New consumers should import FROM HERE (``paf.capabilities._storage._atomic``).
"""

from __future__ import annotations

import json
from pathlib import Path


def write_atomic(path: str | Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (unique temp + os.replace).

    The temp file lives in the same directory (so ``os.replace`` is an atomic
    rename on the same filesystem) with a **process-unique** name
    (``tempfile.mkstemp``), so concurrent writers to the SAME target can never
    steal each other's staging file mid-replace (a fixed ``.<name>.staging``
    name raced — one thread's ``os.replace`` could consume the staging file
    another thread was about to move, raising ``FileNotFoundError``). The
    temp is removed on any failure so no ``*.tmp`` debris is left behind.
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


def write_json_atomic(path: str | Path, payload: object) -> None:
    """Write ``payload`` to ``path`` as pretty-printed JSON, atomically."""
    write_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False))


def read_json(path: str | Path, default: object = None) -> object:
    """Read a JSON file, returning ``default`` (not raising) if absent/corrupt."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default
