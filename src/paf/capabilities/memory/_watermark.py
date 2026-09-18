"""Persisted-consumption watermark for background memory consolidation.

The trigger that decides *when* a background memory-consolidation job should
fire, decoupled from eviction. Eviction (``TurnBoundSlidingWindow``) is
**trim-only** and never calls memory; here the owner runs *because enough
durable conversation has accumulated since the last time it consumed*, tracked
by a flat-file watermark.

This module holds the pure, testable pieces — the watermark record and the
``should_fire`` trigger predicate — with **no model and no persistence calls**.
The queue consumer that uses them lives in :mod:`. _background`.

Ref model (see ``_journal.py``): a turn's durable address is anchored on the
run's persisted ``started_at`` (a native UTC timestamp, index-sorted on the
store), which is **monotonic across runs** — unlike a ``turn_id`` ordinal,
which resets when a session forks/continues and therefore can't anchor
cross-run cadence. The watermark is a single ``consumed_ts`` timestamp: the
last time the owner folded durable turns.

Design notes from ``Designs/Background-Memory-Watermark-Spec.md``:
- Flat-file-first — ``memory-watermark.json`` sits beside the journal /
  ``MEMORY.md``, diffable and swappable (no opaque DB).
- The cadence is a **time** metric: ``should_fire`` fires once at least
  ``consolidate_every`` **minutes** have elapsed since ``consumed_ts`` (and
  immediately on a fresh/legacy watermark with no anchor). This is decoupled
  from context eviction — eviction is trim-only and never calls memory — so
  eviction is just a safety net, not a trigger or dependency.
- The trigger predicate here is deliberately time-anchored; the queue
  consumer's window selection handles which turns were new since the anchor.

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

        A totally absent file is a **fresh** watermark (nothing consumed). It
        is returned with ``run_id=""`` and ``turn_id=None`` — NOT re-anchored
        at the ``run_id`` argument — because ``resolve_unseen_span`` treats a
        watermark whose run is found in ``list_runs`` as *mid-history* (all
        runs strictly older are consumed). Pre-anchoring a fresh watermark at
        the tip run made a brand-new conversation's whole backlog look
        already-consumed, so a turn-count-gated consumer (e.g. the thread
        summarizer, which won't fire until >=10 *unsettled* turns) deadlocked:
        only in-tip turns counted, never reached the threshold, the watermark
        was never persisted, and it re-bootstrapped at the new tip every turn.
        An empty fresh watermark resolves the whole backlog as new (the
        documented "fresh => everything new" contract). ``run_id`` is still
        honored for the file-exists (legacy / round-trip) paths below.
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
        return cls()

    def save(
        self, directory: str | os.PathLike, filename: str = _WATERMARK_FILENAME
    ) -> None:
        """Persist the watermark atomically (staging + ``os.replace``)."""
        path = Path(directory) / filename
        payload = json.dumps(
            {
                "run_id": self.run_id,
                "turn_id": self.turn_id,
                "consumed_ts": self.consumed_ts.isoformat()
                if self.consumed_ts
                else None,
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
    every: timedelta,
) -> bool:
    """The metric trigger: have we accumulated enough new material to run a job?

    ``True`` when at least ``every`` (a :class:`datetime.timedelta`) has elapsed
    since the watermark's ``consumed_ts`` — a **time** metric that is monotonic
    across runs (``started_at`` is native + index-sorted on the store, unlike a
    per-run ``turn_id`` ordinal which is only monotonic within one run and resets
    when a session forks/continues). Or ``True`` when nothing has been consumed
    yet (fresh / legacy watermark with no ``consumed_ts``) — the whole backlog is
    new.

    ``every`` is the cadence (a positive :class:`datetime.timedelta`, minutes in
    practice). Pure time-based and decoupled from context eviction — eviction
    is trim-only and never calls memory, so nothing binds this to the window.
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
