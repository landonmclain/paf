"""overview.json — the navigation tree (the STARTING POINT).

The model's first read: a folder/file tree with counts, collapsed from the
combined ignore list. No roles, no per-file summaries. Filename
is the key — the model reads the tree top-down, then drills via ``lookup_file``.

We build it from the SAME walker as the scan (``ScanScope.iter_files``) so the
corpus the overview advertises is exactly the corpus the index covers — no
divergence between "what the tree shows" and "what lookup/search can find."
Directories are derived from the walked files (a dir with no in-scope files is
not emitted, because it has no indexed children).

Each tree entry is ``{path, type: dir|file, depth, file_count? | line_count?}``
(byte-lean): dirs carry ``file_count``; files carry ``line_count``
(the count of source lines, cheaply read byte-by-byte — no parse needed).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

from ._store import ArtifactKind, ScanScope, StorageService

_VERSION = 1


def _line_count(path: Path) -> int:
    """Count source lines cheaply (no parse) — ``text.splitlines()`` length."""
    try:
        return len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return 0


def build_overview(scan_root: str | Path) -> dict[str, Any]:
    """Build the ``overview.json`` payload for ``scan_root`` (in-memory).

    Pure derivation from the in-scope walk — returns the dict the storage seam
    writes. Does NOT touch storage. Callers (``scan_and_write`` / a tool that
    lazily builds a missing overview) write it with ``storage.write(OVERVIEW,
    scan_root, payload)``.
    """
    root = Path(scan_root).resolve()
    scope = ScanScope(root)
    files = scope.iter_files()

    by_dir: dict[str, int] = {}
    entries: list[dict[str, Any]] = []

    for path in files:
        rel = path.relative_to(root).as_posix()
        # count the file under every ancestor dir
        parts = rel.split("/")
        for i in range(1, len(parts)):  # skip the bare filename (i starts at 1)
            d = "/".join(parts[:i])
            by_dir[d] = by_dir.get(d, 0) + 1
        entries.append(
            {
                "path": rel,
                "type": "file",
                "depth": len(parts),
                "line_count": _line_count(path),
            }
        )

    # dir entries — one per ancestor dir (in any, sorted top-down below)
    for d, count in by_dir.items():
        entries.append(
            {"path": d, "type": "dir", "depth": d.count("/") + 1, "file_count": count}
        )

    # stable deterministic order: shallow depth first, then path
    entries.sort(key=lambda e: (e["depth"], e["path"]))

    return {
        "scan_root": str(root),
        "generated": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "version": _VERSION,
        "tree": entries,
    }


def load_overview(
    storage: StorageService, scan_root: str | Path
) -> dict[str, Any] | None:
    """Read the overview artifact (None when not indexed yet)."""
    payload = storage.read(ArtifactKind.OVERVIEW, Path(scan_root).resolve())
    return payload if isinstance(payload, dict) else None
