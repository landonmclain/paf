"""One place for "what time is it, in the user's zone".

Rule: **store UTC, show local.** Persisted timestamps (journal ``born_on``,
run records, watermarks, process rows) stay UTC so they sort and compare
lexically. Anything the *model* reads — the per-message ``[date: …]`` stamp,
rendered thread-summary times, process status times — goes through here so it
is in the user's wall-clock zone, with the UTC offset kept in ISO strings so
they stay unambiguous and machine-parseable.

The zone is ``settings.timezone`` (``TIMEZONE`` in ``.env``, an IANA name such
as ``America/Chicago``); when unset it is the machine's zone (``/etc/localtime``),
and UTC as the last resort.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, tzinfo
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from paf.config import settings

_ZONEINFO_MARKER = "zoneinfo/"


def _system_zone_name() -> str | None:
    """IANA name of the machine's zone, from the ``/etc/localtime`` symlink."""
    try:
        target = os.path.realpath("/etc/localtime")
    except OSError:
        return None
    if _ZONEINFO_MARKER in target:
        return target.split(_ZONEINFO_MARKER, 1)[1]
    return None


@lru_cache(maxsize=1)
def local_zone() -> tzinfo:
    """The user's zone: ``settings.timezone``, else the system zone, else UTC."""
    for name in (settings.timezone, _system_zone_name()):
        if name:
            try:
                return ZoneInfo(name)
            except (ZoneInfoNotFoundError, ValueError):
                continue
    return UTC


def zone_name() -> str:
    """Display name of the user's zone (``America/Chicago`` / ``UTC``)."""
    return str(local_zone())


def now_local() -> datetime:
    """Current time in the user's zone, second precision."""
    return datetime.now(local_zone()).replace(microsecond=0)


def to_local(value: str | datetime | None) -> datetime | None:
    """Convert an ISO string / datetime to the user's zone (naive = UTC).

    Returns ``None`` for empty or unparseable input.
    """
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(local_zone())


def local_iso(value: str | datetime | None) -> str:
    """ISO string in the user's zone (with offset); input echoed if unparseable."""
    dt = to_local(value)
    if dt is None:
        return "" if value is None else str(value)
    return dt.isoformat()
