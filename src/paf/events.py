"""Generic pub/sub event primitive shared by paf capabilities.

The one observation seam for long-running or externally-observable activity.
Any store or tool that mutates something the outside world should watch — plan
steps, docker-sandboxed shell runs, test runs, subagent progress — can:

- declare an event "type" (a str/``Enum`` discriminant),
- emit an event at each mutation, and
- let a UI/sidebar subscribe via ``on(key, cb)``.

Keeping this generic (not plan-specific) means planning, shell, tests, and
subagents all share ONE dispatch mechanism instead of each carrying a bespoke
copy. This module is the extraction of the pub/sub primitive that originally
lived (plan-shaped) in ``capabilities/planning/_events.py``.

Design notes:

- **Listener-exception isolation** — ``emit`` never lets one listener's
  exception break the emitter or the other listeners. A failing subscriber is
  logged and skipped (the same guarantee ``background.py``'s ``_publish_task``
  gives the task bus), so a UI bug can never stall a store write or a turn.
- **Sync + async listeners** — a listener may be a plain callable or a
  coroutine function; ``emit`` awaits async ones in registration order.
- **Typed but not over-typed** — it stays generic over payload ``Evt`` and
  discriminant ``Key``; callers specialize (e.g. ``PlanEventEmitter``) so
  static analysis still sees concrete event payloads.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from enum import Enum

log = logging.getLogger(__name__)

type Listener[EvtT] = Callable[[EvtT], None | Awaitable[None]]
"""A sync or async listener invoked with each emitted event."""


class GenericEventEmitter[KeyT: str | Enum, EvtT]:
    """Register listeners and dispatch typed events to them.

    ``KeyT`` is the event-type discriminant (a ``str`` or ``Enum``); ``EvtT``
    is the payload carried to listeners. Emitting is synchronous-ordered:
    listeners registered for an event fire in registration order, with async
    ones awaited. One listener raising never prevents the others from running
    (the exception is logged and swallowed).
    """

    def __init__(self) -> None:
        self._listeners: dict[KeyT, list[Listener[EvtT]]] = {}

    def on(self, event_type: KeyT, callback: Listener[EvtT]) -> Listener[EvtT]:
        """Register ``callback`` for ``event_type`` and return it (usable as a decorator)."""
        self._listeners.setdefault(event_type, []).append(callback)
        return callback

    def off(self, event_type: KeyT, callback: Listener[EvtT]) -> bool:
        """Remove ``callback`` from ``event_type``; return whether it was registered."""
        bucket = self._listeners.get(event_type)
        if bucket is None:
            return False
        try:
            bucket.remove(callback)
        except ValueError:
            return False
        return True

    async def emit(self, event: EvtT, event_type: KeyT) -> None:
        """Invoke every listener registered for ``event_type``, awaiting async ones.

        A listener that raises is logged and skipped — it must never prevent
        the remaining listeners from receiving the event, nor raise out of the
        emit (a store write or turn must not fail because a UI subscriber
        broke).
        """
        for callback in self._listeners.get(event_type, ()):
            try:
                result = callback(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - a subscriber must never break the emitter
                log.exception(
                    "event emitter listener failed type=%r; skipped",
                    getattr(event_type, "value", event_type),
                )
