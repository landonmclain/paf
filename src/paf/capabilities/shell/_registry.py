"""Process-wide registry of process event emitters, keyed by agent name.

The live background-process producers (the shell ``ShellBackgroundQueue`` and the
coordinator's task bus) and the SSE observer (``ui/sse.py``) must subscribe to the
SAME ``ProcessEventEmitter`` for a given agent, or process transitions would be
invisible to the browser.

Keying is by **agent name** (e.g. ``assistant``), which is already unambiguous:
the coordinator's agent lane and the shell queue both derive the same agent name
from the same session/agent, so they always hit the same emitter. A process event
carries its own ``lane`` + ``scope`` so one per-agent emitter can serve both the
agent lane and the conversation lane.

Emitters are created lazily and kept for the process lifetime; the SSE layer
subscribes/unsubscribes listeners per open stream. An idle emitter with no
listeners is a harmless no-op (``emit`` to an empty bucket is free), so we do
not need refcount teardown.
"""

from __future__ import annotations

import threading

from paf.capabilities.shell._events import ProcessEventEmitter

_lock = threading.Lock()
_emitters: dict[str, ProcessEventEmitter] = {}


def get_process_emitter(agent: str) -> ProcessEventEmitter:
    """Return the process-wide ``ProcessEventEmitter`` for an agent name.

    Creates it on first use. Safe to call from the runtime worker and the SSE
    event loop concurrently (module-level lock).
    """
    with _lock:
        emitter = _emitters.get(agent)
        if emitter is None:
            emitter = ProcessEventEmitter()
            _emitters[agent] = emitter
        return emitter


def clear_process_emitters() -> None:
    """Drop all emitters (test isolation only)."""
    with _lock:
        _emitters.clear()
