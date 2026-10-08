"""Process-wide registry of plan event emitters, keyed by scoped conversation id.

The live plan-store producer (`FilePlanStore` resolved in ``build.py``) and the
SSE observer (`ui/sse.py``) must subscribe to the SAME ``PlanEventEmitter`` for
a given conversation, or plan mutations would be invisible to the browser.

Keying is by **scoped conversation id** (``{profile}:{cid}``, e.g.
``demo:paf-ui-e1``), which is already unambiguous: the profile prefix means two
different agents/profiles can never share a scoped id, so a module-global dict
is safe. Both sides derive the same scoped id from the same session/context
(``session.scoped_id`` / ``ctx.conversation_id``), so they always hit the same
emitter.

Emitters are created lazily and kept for the process lifetime; the SSE layer
subscribes/unsubscribes listeners per open stream. An idle emitter with no
listeners is a harmless no-op (``emit`` to an empty bucket is free), so we do
not need refcount teardown.
"""

from __future__ import annotations

import threading

from paf.capabilities.planning._events import PlanEventEmitter

_lock = threading.Lock()
_emitters: dict[str, PlanEventEmitter] = {}


def get_plan_emitter(scoped_conversation_id: str) -> PlanEventEmitter:
    """Return the process-wide ``PlanEventEmitter`` for a scoped conversation id.

    Creates it on first use. Safe to call from the runtime worker and the SSE
    event loop concurrently (module-level lock).
    """
    with _lock:
        emitter = _emitters.get(scoped_conversation_id)
        if emitter is None:
            emitter = PlanEventEmitter()
            _emitters[scoped_conversation_id] = emitter
        return emitter


def clear_plan_emitters() -> None:
    """Drop all emitters (test isolation only)."""
    with _lock:
        _emitters.clear()
