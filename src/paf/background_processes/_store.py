"""Dumb append-only event store for background processes.

A lossless log of every background-process action. It does not interpret
records: no status enum, no start/end, no derived state. It just appends
whatever event it is handed and reads them back in order. Status gating lives
in the background queue, not here.

The store is a thin wrapper over the shared :class:`AppendOnlyStore` seam
(``FlatJsonlStore`` default backend), scoped by ``key=<agent>`` so each agent's
events live in ``data/processes/<agent>/runs.jsonl``. The seam is swappable
(SQLite later) without touching this module's callers.
"""

from __future__ import annotations

import hashlib
from typing import Any

from paf.capabilities._storage import AppendOnlyStore, FlatJsonlStore

DEFAULT_KIND = "runs"


def _safe_cid(cid: str) -> str:
    """Coerce a conversation id to a safe, stable single path segment.

    Hashed (matching the thread summarizer's per-conversation folders) so the
    journal path is safe against arbitrary cid characters and consistent with
    the codebase's per-conversation folder convention.
    """
    return hashlib.sha256(cid.encode("utf-8")).hexdigest()[:24]


class RunEventStore:
    """Dumb append-only event store for one agent, over the :class:`AppendOnlyStore` seam."""

    def __init__(
        self, store: AppendOnlyStore, *, agent: str, kind: str = DEFAULT_KIND
    ) -> None:
        self._store = store
        self._agent = agent
        self._kind = kind

    @classmethod
    def open(
        cls,
        root: str,
        agent: str,
        *,
        kind: str = DEFAULT_KIND,
        conversation_id: str | None = None,
    ) -> RunEventStore:
        """Open a store rooted at ``root`` (``data/processes``) for ``agent``.

        With ``conversation_id`` the store is conversation-scoped and roots at
        ``<root>/<agent>/conversations/<cid>/runs.jsonl``; otherwise it is the
        per-agent journal ``<root>/<agent>/runs.jsonl``.
        """
        if conversation_id is None:
            return cls(FlatJsonlStore(root), agent=agent, kind=kind)
        key = f"{agent}/conversations/{_safe_cid(conversation_id)}"
        return cls(FlatJsonlStore(root), agent=key, kind=kind)

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append one event record. No interpretation — just a record."""
        return self._store.append(self._kind, self._agent, event)

    def read_all(self) -> list[dict[str, Any]]:
        """All events in append order (file line order is the source of truth)."""
        return self._store.read_all(self._kind, self._agent)

    def recent(self, process_id: str, conversation_id: str) -> dict[str, Any] | None:
        """The newest run record for one process + conversation (or None).

        Lines are append order = chronological order (each publish is stamped
        with a microsecond-stripped ``now``, so two publishes in the same
        second collide on timestamp; file line order disambiguates the winner,
        same rule the coordinator's journal mandates). Returned as the raw
        record dict so callers can derive status/detail/progress from it.
        """
        for record in reversed(self.read_all()):
            if record.get("process_id") != process_id:
                continue
            if record.get("conversation_id") != conversation_id:
                continue
            return record
        return None
