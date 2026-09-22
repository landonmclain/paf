"""paf-owned process event system, specialized over the generic EventEmitter.

The pub/sub dispatch lives in the shared ``GenericEventEmitter``
(``src/paf/events.py``); this module is a thin domain specialization: the
process-specific event types + payload, with ``ProcessEventEmitter`` as a
``GenericEventEmitter[ProcessEventType, ProcessEvent]`` so applications (e.g. the
paf web Processes panel) react to background-process transitions.

The public surface is ``on``/``off``/``emit`` plus the
``on_queued``/``on_running``/``on_done``/``on_error`` convenience decorators.
``emit`` accepts a single ``ProcessEvent`` and derives the key from
``event.event_type`` (the generic only needs the explicit-key variant for
non-payload-keyed events, which processes do not use).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from paf.events import GenericEventEmitter, Listener


class ProcessEventType(str, Enum):  # noqa: UP042 - str-Enum hybrid keeps event_type JSON-serializable
    """The kinds of transition a background process can emit.

    Attributes:
        queued: The process was enqueued but not yet started.
        running: The process is actively running.
        done: The process finished successfully.
        error: The process failed or was terminated abnormally.
    """

    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


class ProcessEvent(BaseModel):
    """A single background-process transition delivered to registered listeners.

    Attributes:
        event_type: What happened.
        id: The process run id (the ``{namespace}.{scope}.{seq}`` key).
        label: A short human label for the process.
        status: The process's current status (matches ``event_type`` for these
            transitions).
        detail: Optional detail (e.g. exit code / output tail).
        lane: Which lane the process belongs to (``agent`` or ``conversation``).
        scope: The conversation id for a conversation-lane process, else None.
        updated_at: ISO timestamp of the transition.
    """

    event_type: ProcessEventType
    id: str
    label: str
    status: str
    detail: str = ""
    lane: str = "conversation"
    scope: str | None = None
    updated_at: str = ""


EventCallback = Listener[ProcessEvent]
"""A sync or async listener invoked with each emitted `ProcessEvent`."""


class ProcessEventEmitter(GenericEventEmitter[ProcessEventType, ProcessEvent]):
    """Register listeners and dispatch `ProcessEvent`s to them.

    Specializes the generic ``GenericEventEmitter`` with process event types and
    payloads. Example:

    ```python
    from paf.capabilities.shell import ProcessEventEmitter

    emitter = ProcessEventEmitter()

    @emitter.on_running
    async def announce(event):
        print('running:', event.id)
    ```
    """

    async def emit_process(self, event: ProcessEvent) -> None:
        """Invoke every listener for ``event.event_type``, awaiting async ones.

        The key is derived from the event itself (unlike the generic's
        explicit-key ``emit(event, event_type)``); the caller calls
        ``emitter.emit_process(ProcessEvent(...))``.
        """
        await super().emit(event, event.event_type)

    def on_queued(self, callback: EventCallback) -> EventCallback:
        """Register a listener for `queued` events."""
        return self.on(ProcessEventType.queued, callback)

    def on_running(self, callback: EventCallback) -> EventCallback:
        """Register a listener for `running` events."""
        return self.on(ProcessEventType.running, callback)

    def on_done(self, callback: EventCallback) -> EventCallback:
        """Register a listener for `done` events."""
        return self.on(ProcessEventType.done, callback)

    def on_error(self, callback: EventCallback) -> EventCallback:
        """Register a listener for `error` events."""
        return self.on(ProcessEventType.error, callback)
