"""Atomic file writes: staging file + ``os.replace`` (inlined from ``agent-foundation``).

These three helpers were the memory package's only use of the ``agent-foundation``
dependency (``agent_foundation.storage.atomic``). They are small and dependency-free,
so they are copied here verbatim to keep ``paf.capabilities.memory`` self-contained
and free of the external ``agent-foundation`` package. turn_index (which consumed
the ``StorageService`` seam directly) has been archived; this file is the port of
the foundation's atomic helper, serving memory.

Why: a plain ``Path.write_text`` on a long JSON artifact can be interrupted midway,
leaving a truncated/half-written file that a later read fails on. Writing to a
sibling ``.staging`` temp then ``os.replace``-ing it into place makes a write
all-or-nothing: observers either see the old complete artifact or the new one,
never a torn write.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def write_atomic(path: str | Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (staging + rename).

    The staging file lives in the same directory (so ``os.replace`` is an
    atomic rename on the same filesystem) with a ``.<name>.staging`` name that
    any reader skips or cleans up lazily.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.staging")
    staging.write_text(content, encoding="utf-8")
    os.replace(staging, path)


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
