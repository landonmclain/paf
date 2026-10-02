"""Process event emitter — the shared live-event bus for background processes.

Specializes the generic :class:`GenericEventEmitter` (``paf.events``) for
background processes, mirroring how ``PlanEventEmitter`` specializes it for
planning. Any process — the coordinator's registered tasks, a shell command, a
future curator agent — emits its lifecycle transitions here, and the process
feed subscribes via ``on``. Keeping this generic (not coordinator-specific)
means every process shares ONE dispatch mechanism instead of each carrying a
bespoke copy.

The emitter is LIVE-ONLY: it carries transitions while a client is connected.
The durable store (``RunEventStore`` / per-process stores) is the source of
truth for reconnect history — the emitter never needs to be durable.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from paf.events import GenericEventEmitter, Listener


class ProcessEventType(str, Enum):  # noqa: UP042 - str-Enum hybrid keeps status JSON-serializable
    """The lifecycle statuses a background process can emit.

    ``turn_started`` is a transport signal (not a process lifecycle state): it
    tells a conversation-lane process feed that a detached notify turn started,
    so the UI can re-attach and stream it. It lives on the same enum so it
    routes through the shared per-agent emitter's ``on_any`` dispatch.
    """

    queued = "queued"
    running = "running"
    done = "done"
    error = "error"
    skipped = "skipped"
    turn_started = "turn_started"


class ProcessEvent(BaseModel):
    """A single background-process lifecycle transition.

    Attributes:
        id: Stable row key (``{namespace}.{scope}.{seq}``) the UI coalesces
            transitions against.
        label: Human label, e.g. "Consolidating memory".
        status: The new lifecycle status.
        progress: 0..1 when known.
        detail: Branch-specific free text.
        lane: Which lane the process belongs to ("agent" or "conversation"),
            so the feed routes it to the right panel.
        scope: The conversation id for a conversation-lane process, or None
            for an agent-lane (per-agent) process.
        lines: Optional itemized result, one readable line per item (e.g. each
            fact the memory reviewer archived), for a UI to expand under the row.
    """

    id: str
    label: str
    status: ProcessEventType
    progress: float | None = None
    detail: str = ""
    lane: Literal["agent", "conversation"] = "agent"
    scope: str | None = None
    lines: list[str] = Field(default_factory=list)


EventCallback = Listener[ProcessEvent]
"""A sync or async listener invoked with each emitted `ProcessEvent`."""


class ProcessEventEmitter(GenericEventEmitter[ProcessEventType, ProcessEvent]):
    """Register listeners and dispatch `ProcessEvent`s to them.

    Specializes the generic ``GenericEventEmitter`` with process event types
    and payloads. The key is derived from the event's own ``status``.
    """

    async def emit_process(self, event: ProcessEvent) -> None:
        """Invoke every listener for ``event.status``, awaiting async ones."""
        await super().emit(event, event.status)

    def on_any(self, callback: EventCallback) -> EventCallback:
        """Register ``callback`` for every process status (all transitions).

        Mirrors the coordinator's ``subscribe_tasks``: one callback receives
        every queued/running/done/error/skipped transition. Returns the
        callback unchanged (usable as a decorator).
        """
        for t in ProcessEventType:
            self.on(t, callback)
        return callback

    def off_any(self, callback: EventCallback) -> bool:
        """Remove ``callback`` from every process status; True if it was registered.

        Mirrors the coordinator's ``unsubscribe_tasks`` so a disconnecting feed
        can release a callback registered via :meth:`on_any`.
        """
        removed = False
        for t in ProcessEventType:
            removed = self.off(t, callback) or removed
        return removed


# Per-agent emitter singleton (one live bus per agent_name, never per-session),
# mirroring ``background._agent_coordinators``. Both the coordinator and any
# process look up the same emitter by agent name, so they share one fan-out —
# no ctx plumbing needed.
_agent_emitters: dict[str, ProcessEventEmitter] = {}


def get_agent_emitter(agent_name: str) -> ProcessEventEmitter:
    """Return the shared per-agent process emitter, creating it on first use."""
    emitter = _agent_emitters.get(agent_name)
    if emitter is None:
        emitter = ProcessEventEmitter()
        _agent_emitters[agent_name] = emitter
    return emitter


def clear_agent_emitters() -> None:
    """Drop all emitters (test isolation only)."""
    _agent_emitters.clear()
