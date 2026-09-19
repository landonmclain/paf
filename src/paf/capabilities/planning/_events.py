"""paf-owned plan event system, specialized over the generic EventEmitter.

The pub/sub dispatch lives in the shared ``GenericEventEmitter``
(``src/paf/events.py``); this module is a thin domain specialization: the
plan-specific event types + payload, with ``PlanEventEmitter`` as a
``GenericEventEmitter[PlanEventType, PlanEvent]`` so applications (e.g. the paf
web sidebar) react to plan mutations.

The public surface is ``on``/``off``/``emit`` plus the
``on_created``/``on_completed``/... convenience decorators. ``emit`` accepts a
single ``PlanEvent`` and derives the key from ``event.event_type`` (the generic
only needs the explicit-key variant for non-payload-keyed events, which planning
does not use).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from paf.capabilities.planning._types import PlanItem
from paf.events import GenericEventEmitter, Listener


class PlanEventType(str, Enum):  # noqa: UP042 - str-Enum hybrid keeps event_type JSON-serializable
    """The kinds of change a store can emit.

    Attributes:
        created: A new step was added.
        updated: A step's fields changed (any field).
        status_changed: A step's status changed.
        deleted: A step was removed.
        completed: A step transitioned to `completed`.
    """

    created = "created"
    updated = "updated"
    status_changed = "status_changed"
    deleted = "deleted"
    completed = "completed"


class PlanEvent(BaseModel):
    """A single plan change delivered to registered listeners.

    Attributes:
        event_type: What happened.
        item: The affected step (post-change for updates).
        previous_state: The step before the change, when the emitter captured it.
    """

    event_type: PlanEventType
    item: PlanItem
    previous_state: PlanItem | None = None


EventCallback = Listener[PlanEvent]
"""A sync or async listener invoked with each emitted `PlanEvent`."""


class PlanEventEmitter(GenericEventEmitter[PlanEventType, PlanEvent]):
    """Register listeners and dispatch `PlanEvent`s to them.

    Specializes the generic ``GenericEventEmitter`` with plan event types and
    payloads. Example:

    ```python
    from paf.capabilities.planning import PlanEventEmitter

    emitter = PlanEventEmitter()

    @emitter.on_completed
    async def announce(event):
        print('done:', event.item.content)
    ```
    """

    async def emit_plan(self, event: PlanEvent) -> None:
        """Invoke every listener for ``event.event_type``, awaiting async ones.

        The key is derived from the event itself (unlike the generic's
        explicit-key ``emit(event, event_type)``); the store calls
        ``emitter.emit_plan(PlanEvent(...))``.
        """
        await super().emit(event, event.event_type)

    def on_created(self, callback: EventCallback) -> EventCallback:
        """Register a listener for `created` events."""
        return self.on(PlanEventType.created, callback)

    def on_updated(self, callback: EventCallback) -> EventCallback:
        """Register a listener for `updated` events."""
        return self.on(PlanEventType.updated, callback)

    def on_status_changed(self, callback: EventCallback) -> EventCallback:
        """Register a listener for `status_changed` events."""
        return self.on(PlanEventType.status_changed, callback)

    def on_completed(self, callback: EventCallback) -> EventCallback:
        """Register a listener for `completed` events."""
        return self.on(PlanEventType.completed, callback)

    def on_deleted(self, callback: EventCallback) -> EventCallback:
        """Register a listener for `deleted` events."""
        return self.on(PlanEventType.deleted, callback)
