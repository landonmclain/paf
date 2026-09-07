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
from datetime import datetime, timedelta
from pathlib import Path

from paf.capabilities._storage._atomic import write_atomic

_WATERMARK_FILENAME = "memory-watermark.json"

#: Sentinel for "no turn consumed yet" (a fresh session / first run).
_EMPTY = ""


@dataclass
class MemoryWatermark:
    """The position the owner has consumed, as a persistent flat file.

    ``consumed_ts`` is the **time anchor** (monotonic across runs — the fix from
    ``Designs/Background-Memory-Watermark-Spec.md``): the watermark's ordering
    key is the persisted ``started_at`` of the last foldable material, not a
    per-run ``turn_id`` ordinal (which is only monotonic within one run and rots
    as sessions fork/continue). ``run_id`` / ``turn_id`` mirror the
    ``{run_id}:{turn_id}`` ref model of ``_journal.py`` and are kept as the
    precise ``source_ref`` address, no longer as the ordering key.

    A ``None`` ``consumed_ts`` (or a missing file) means "nothing consumed yet" —
    the first trigger sees the whole backlog as new and the ``should_fire``
    predicate falls back to a fresh-session fire.

    Stored next to the journal at ``<memory_dir>/memory-watermark.json`` so the
    durable "what's seen" marker survives restart and stays diffable.
    """

    run_id: str = ""
    turn_id: int | None = None
    consumed_ts: datetime | None = None

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

        A legacy file (only ``{run_id, turn_id}``, no ``consumed_ts``) loads
        fine with ``consumed_ts`` left ``None``; a corrupt/unparseable
        ``consumed_ts`` is likewise treated as fresh — the append-only journal
        is the lossless source of truth, so a missing watermark only defers a
        beat, never loses a fact.
        """
        path = Path(directory) / filename
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                ts_raw = data.get("consumed_ts")
                consumed_ts = None
                if isinstance(ts_raw, str):
                    consumed_ts = datetime.fromisoformat(ts_raw)
                return cls(
                    run_id=data.get("run_id", run_id),
                    turn_id=data.get("turn_id"),
                    consumed_ts=consumed_ts,
                )
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
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
            {
                "run_id": self.run_id,
                "turn_id": self.turn_id,
                "consumed_ts": self.consumed_ts.isoformat() if self.consumed_ts else None,
            },
            ensure_ascii=False,
        )
        write_atomic(os.fspath(path), payload + "\n")

    def is_empty(self) -> bool:
        """True when nothing has been consumed yet."""
        return self.consumed_ts is None

    def __repr__(self) -> str:
        return f"MemoryWatermark(run_id={self.run_id!r}, turn_id={self.turn_id!r})"


def should_fire(
    watermark: MemoryWatermark,
    *,
    tip_ts: datetime,
    every: datetime.timedelta,
) -> bool:
    """The metric trigger: have we accumulated enough new material to run a job?

    ``True`` when at least ``every`` (a :class:`datetime.timedelta`) has elapsed
    since the watermark's ``consumed_ts`` — a **time** metric that is monotonic
    across runs (``started_at`` is native + index-sorted on the store, unlike a
    per-run ``turn_id`` ordinal which is only monotonic within one run and resets
    when a session forks/continues). Or ``True`` when nothing has been consumed
    yet (fresh / legacy watermark with no ``consumed_ts``) — the whole backlog is
    new.

    ``every`` is the cadence; the spec mandates it run ahead of eviction
    (``every < keep``) so the owner always consumes before the window trims.
    """
    if every is None or every <= timedelta(0):
        raise ValueError(f"every must be a positive timedelta, got {every!r}")
    if watermark.consumed_ts is None:
        # Fresh watermark (or a legacy one with only {run_id, turn_id}): the
        # time anchor is unknown, so treat the whole backlog as new. The
        # append-only journal is lossless, so an extra first run costs at most
        # one owner pass.
        return True
    return (tip_ts - watermark.consumed_ts) >= every
