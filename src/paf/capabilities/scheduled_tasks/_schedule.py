"""Schedule parsing for scheduled tasks (Phase 2).

The only schedule form is a cron expression, parsed with ``croniter``. The
scheduler computes the next fire time in the user's zone (via ``paf.clock``)
so cron semantics match the user's wall clock, not UTC.
"""

from __future__ import annotations

from datetime import datetime

from croniter import croniter

from paf.clock import now_local


def next_fire(schedule: str, after: datetime | None = None) -> datetime:
    """The next fire time for a cron ``schedule`` strictly after ``after``.

    ``after`` defaults to now in the user's zone (``paf.clock.now_local``) and
    must be timezone-aware; the returned datetime is timezone-aware in the
    user's zone. Raises ``ValueError`` for an unparseable cron expression.
    """
    base = after if after is not None else now_local()
    try:
        it = croniter(schedule, base)
    except (ValueError, KeyError) as exc:
        raise ValueError(f"invalid cron schedule {schedule!r}: {exc}") from exc
    return it.get_next(datetime)
