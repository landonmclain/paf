"""Storage Protocol — the seam every stateful subsystem hangs off.

The same narrow shape (``get``/``set``/``search`` + bulk) backs flat-file,
SQLite, Postgres, or object-store. Backends swap via config, never a rewrite.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    """Minimal key/value store with search — the core storage abstraction."""

    def get(self, key: str) -> Any | None:
        """Return the value for ``key`` or None if absent."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Persist ``value`` under ``key``."""
        ...

    def search(self, query: str, limit: int = 50) -> Sequence[str]:
        """Return keys/ids matching ``query`` (implementation-defined search)."""
        ...

    def delete(self, key: str) -> None:
        """Remove ``key`` if present."""
        ...
