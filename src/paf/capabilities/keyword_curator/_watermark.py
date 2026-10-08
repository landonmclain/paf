"""Change-detection cursors for the keyword curator.

Skills and notes have no natural sequence, so each is tracked by a state
snapshot — ``mtime`` + ``size`` — and re-processed only when the snapshot
differs. (Memory, a later phase, needs no cursor: it reconciles against the
refs already in the journal.)

``keyword-watermark.json`` sits beside the journal::

    {"updated_at": "...", "files": {"skill:foo": {"mtime": 1.0, "size": 42}}}

The worker calls :meth:`KeywordWatermark.advance` and :meth:`save` only after a
pass completes without error, so a failed pass is simply retried: the journal is
lossless and appends are idempotent, so reprocessing costs a model call, never
correctness. A missing or corrupt file loads as fresh, which re-queues every
source.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from paf.capabilities._storage._atomic import write_atomic

WATERMARK_FILENAME = "keyword-watermark.json"

Snapshot = tuple[float, int]
"""A source file's ``(mtime, size)``."""


@dataclass
class KeywordWatermark:
    """Per-source ``(mtime, size)`` snapshots of what has been keyworded."""

    files: dict[str, Snapshot] = field(default_factory=dict)
    updated_at: str | None = None

    @classmethod
    def load(cls, directory: str | os.PathLike) -> KeywordWatermark:
        """Read the watermark from ``directory`` (fresh if absent/corrupt)."""
        path = Path(directory) / WATERMARK_FILENAME
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        files: dict[str, Snapshot] = {}
        raw = data.get("files")
        for ref, snap in (raw if isinstance(raw, dict) else {}).items():
            if not isinstance(snap, dict):
                continue
            mtime, size = snap.get("mtime"), snap.get("size")
            if (
                isinstance(mtime, int | float)
                and not isinstance(mtime, bool)
                and isinstance(size, int)
                and not isinstance(size, bool)
            ):
                files[str(ref)] = (float(mtime), size)
        updated = data.get("updated_at")
        return cls(
            files=files, updated_at=updated if isinstance(updated, str) else None
        )

    def save(self, directory: str | os.PathLike) -> None:
        """Persist atomically (staging + ``os.replace``)."""
        payload = {
            "updated_at": self.updated_at,
            "files": {
                ref: {"mtime": mtime, "size": size}
                for ref, (mtime, size) in sorted(self.files.items())
            },
        }
        write_atomic(
            Path(directory) / WATERMARK_FILENAME,
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        )

    def changed(self, current: Mapping[str, Snapshot]) -> list[str]:
        """Refs in ``current`` that are new or whose snapshot differs (sorted)."""
        return sorted(
            ref for ref, snap in current.items() if self.files.get(ref) != tuple(snap)
        )

    def vanished(self, current: Mapping[str, Snapshot]) -> list[str]:
        """Tracked refs that are no longer in ``current`` (sorted)."""
        return sorted(ref for ref in self.files if ref not in current)

    def advance(
        self,
        done: Mapping[str, Snapshot],
        *,
        remove: Iterable[str] = (),
    ) -> None:
        """Record ``done`` snapshots as keyworded and forget ``remove`` refs.

        Call only after a pass completed without error. Does not save.
        """
        for ref, snap in done.items():
            self.files[ref] = (float(snap[0]), int(snap[1]))
        for ref in remove:
            self.files.pop(ref, None)
        self.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
