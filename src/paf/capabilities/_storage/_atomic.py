"""Atomic file writes: staging file + ``os.replace``.

A plain ``Path.write_text`` on a long JSON artifact can be interrupted midway,
leaving a truncated/half-written file that a later read fails on. Writing to a
sibling ``.staging`` temp then ``os.replace``-ing it into place makes a write
all-or-nothing: observers either see the old complete artifact or the new one,
never a torn write.

**Canonical shared home.** These helpers used to be inlined per-capability; the
established copies were:

    - ``paf/capabilities/memory/_atomic.py`` — near-verbatim copy (memory keeps
      its own to stay dependency-free / self-contained).
    - ``paf/capabilities/vault/_store.py``,
      ``paf/capabilities/codebase_map/_store.py`` — their own private
      ``_write_atomic/_write_json_atomic/_read_json`` copies.
    - ``archive/turn_index/_store.py`` — the one consumer that imported the
      shared ``agent_foundation.storage`` seam directly (since archived).

New consumers should import FROM HERE (``paf.capabilities._storage._atomic``).
Migrating the existing capability copies onto this module is a deferred
clean-up; the conversation-title store is the first consumer of the shared
module. NOTE: vault's copy uses a **process-unique ``tempfile.mkstemp``** temp
name (thread-safe for concurrent single-file editors) — the shared helper uses
the fixed ``.<name>.staging`` name; on unification the shared module should
adopt vault's mkstemp semantics.
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
