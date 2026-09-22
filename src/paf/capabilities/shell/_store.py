"""Dumb append-only event store for shell background commands.

A lossless log of every shell background-command action, kept APART from the
coordinator's ``RunEventStore`` so shell's richer detail (``pid``, ``command``,
``cwd``, ``exit_code``) never bends the coordinator's thin 7-field journal
shape. It mirrors :class:`RunEventStore` exactly: it does not interpret
records (no status enum, no derived state); it just appends whatever event it
is handed and reads them back in order.

The join key is ``run_id``: the SAME ``run_id`` that shell also writes as a
*basic* record into the coordinator's ``RunEventStore`` (``runs.jsonl``, the one
the UI watches). Given a row in the feed, look up that ``run_id`` here for the
rich/live-attach detail (``pid`` especially — "the only way status check works").

The store is a thin wrapper over the shared :class:`AppendOnlyStore` seam
(``FlatJsonlStore`` default backend), scoped by ``key=<agent>`` so records live
in ``data/processes/<agent>/shell_cmds.jsonl`` — the same ``processes_root`` as
``RunEventStore``. The seam is swappable (SQLite later) without touching this
module's callers.
"""

from __future__ import annotations

from typing import Any

from paf.capabilities._storage import AppendOnlyStore, FlatJsonlStore

DEFAULT_KIND = "shell_cmds"


class ShellCommandStore:
    """Dumb append-only event store for one agent's shell commands."""

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
    ) -> ShellCommandStore:
        """Open a store rooted at ``root`` (``data/processes``) for ``agent``.

        With ``conversation_id`` the store is conversation-scoped and roots at
        ``<root>/<agent>/conversations/<cid>/shell_cmds.jsonl`` (colocated with
        the basic ``runs.jsonl`` for the same conversation); otherwise it is the
        per-agent journal ``<root>/<agent>/shell_cmds.jsonl``.
        """
        if conversation_id is None:
            return cls(FlatJsonlStore(root), agent=agent, kind=kind)
        # Lazy import: background_processes._store transitively re-enters
        # paf.capabilities, so a top-level import here would be circular.
        from paf.background_processes._store import _safe_cid

        key = f"{agent}/conversations/{_safe_cid(conversation_id)}"
        return cls(FlatJsonlStore(root), agent=key, kind=kind)

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append one rich shell record. No interpretation — just a record."""
        return self._store.append(self._kind, self._agent, event)

    def read_all(self) -> list[dict[str, Any]]:
        """All shell records in append (file line) order."""
        return self._store.read_all(self._kind, self._agent)

    def all_for(self, run_id: str) -> list[dict[str, Any]]:
        """Every rich line for one shell process ``run_id``, in append order.

        A process appends multiple lines over its life (``running`` -> a
        terminal ``done``/``error``), so this is the full journal for the run;
        the LAST element is its current state. Used by ``check``/``stop`` to
        resolve status (and to detect that the supervisor died before appending
        the terminal line).
        """
        return [r for r in self.read_all() if r.get("run_id") == run_id]

    def recent(self, run_id: str) -> dict[str, Any] | None:
        """The newest rich line for one shell process ``run_id`` (or None).

        Lines are append order = chronological order. ``run_id`` is the join key
        shared with the basic ``runs.jsonl`` record. Returned as the raw dict so
        callers derive status/pid/exit_code/live-attach detail from it.
        """
        lines = self.all_for(run_id)
        return lines[-1] if lines else None
