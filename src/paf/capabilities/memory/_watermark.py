"""Persisted-consumption watermark for background memory consolidation.

The trigger that decides *when* a background memory-consolidation job should
fire, decoupled from eviction. Eviction (``TurnBoundSlidingWindow``) is
**trim-only** and never calls memory; here the owner runs *because enough
durable conversation has accumulated since the last time it consumed*, tracked
by a flat-file watermark.

This module holds the pure, testable pieces — the watermark record and the
``should_fire`` trigger predicate — with **no model and no persistence calls**.
The queue consumer that uses them lives in :mod:`. _background`.

Ref model (LOCKED, see ``_journal.py``): every message carries a
``{run_id}:{turn_id}`` address, where ``turn_id`` is the ordinal of the run's
user-prompt anchor, stable for the life of the run (unlike a
``message_index`` which rots as messages append). The watermark is therefore a
single ``{run_id, turn_id}`` pair: *"the last user-turn the owner has consumed."*

Design notes from ``Designs/Background-Memory-Watermark-Spec.md``:
- Flat-file-first — ``memory-watermark.json`` sits beside the journal /
  ``MEMORY.md``, diffable and swappable (no opaque DB).
- The turn-count metric runs **ahead of** the window's ``keep_messages`` (e.g.
  keep 20 → fire every 10) so the owner always consumes *before* eviction, and
  eviction stays only a safety net.
- ``turn_id`` is only monotonic *within one run*; cross-run enumeration needs
  the persisted ``started_at`` ordering (documented in the spec). The trigger
  predicate here is deliberately run-scoped; cross-run anchoring is handled by
  the queue consumer's window selection, not this predicate.

Nothing here imports ``pydantic_ai`` or any LLM. Everything is a pure func /
dataclass so the trigger is unit-tested without paying for a model call.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ._atomic import write_atomic

_WATERMARK_FILENAME = "memory-watermark.json"

#: Sentinel for "no turn consumed yet" (a fresh session / first run).
_EMPTY = ""


@dataclass
class MemoryWatermark:
    """The last user-turn the owner has consumed, as a persistent flat file.

    ``run_id`` / ``turn_id`` mirror the ``{run_id}:{turn_id}`` ref model of
    ``_journal.py``. An empty ``turn_id`` (or a missing file) means "nothing
    consumed yet" — the first trigger sees the whole backlog as new.

    Stored next to the journal at ``<memory_dir>/memory-watermark.json`` so the
    durable "what's seen" marker survives restart and stays diffable.
    """

    run_id: str = ""
    turn_id: int | None = None

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike,
        *,
        run_id: str = "",
        filename: str = _WATERMARK_FILENAME,
    ) -> MemoryWatermark:
        """Read the watermark for ``directory`` (or a fresh empty one).

        ``filename`` defaults to the consolidation watermark file; a caller may
        pass a different name (e.g. a separate review watermark) so the owner
        and reviewer cadences track independently.
        """
        path = Path(directory) / filename
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return cls(run_id=data.get("run_id", run_id), turn_id=data.get("turn_id"))
            except (json.JSONDecodeError, OSError, TypeError):
                # Corrupt/absent -> treat as fresh; append-only journal is the
                # lossless source of truth, a missing watermark only defers a
                # beat, never loses a fact.
                pass
        return cls(run_id=run_id)

    def save(
        self, directory: str | os.PathLike, filename: str = _WATERMARK_FILENAME
    ) -> None:
        """Persist the watermark atomically (staging + ``os.replace``)."""
        path = Path(directory) / filename
        payload = json.dumps(
            {"run_id": self.run_id, "turn_id": self.turn_id}, ensure_ascii=False
        )
        write_atomic(os.fspath(path), payload + "\n")

    def is_empty(self) -> bool:
        """True when nothing has been consumed yet."""
        return self.turn_id is None

    def __repr__(self) -> str:
        return f"MemoryWatermark(run_id={self.run_id!r}, turn_id={self.turn_id!r})"


def should_fire(
    watermark: MemoryWatermark,
    *,
    current_turn_id: int,
    every: int,
) -> bool:
    """The metric trigger: has enough new material accumulated to run a job?

    ``True`` when at least ``every`` user-turns have accumulated past the
    watermark since the last consolidation — *or* when nothing has been
    consumed yet (fresh session, whole backlog is new).

    ``every`` is the trigger cadence; the spec mandates ``every < keep`` (window
    ``keep_messages``) so the owner always consumes ahead of eviction.
    """
    if every <= 0:
        raise ValueError(f"every must be > 0, got {every!r}")
    if watermark.turn_id is None:
        return True
    return (current_turn_id - watermark.turn_id) >= every
