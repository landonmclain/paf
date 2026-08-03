"""Example raw toolset — a plain, typed tool usable by any agent."""

from __future__ import annotations


def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


def current_datetime() -> str:
    """Return the current UTC datetime as an ISO-8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
