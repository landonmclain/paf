"""Unit tests for the process event emitter (``src/paf/background_processes/_events.py``).

The generic dispatch/isolation guarantees are covered in ``test_events.py``;
these tests exercise the background-process specialization:

- ``emit_process`` dispatches to listeners registered via ``on_any``,
- ``on_any`` receives every status transition,
- ``off_any`` releases a callback registered via ``on_any``,
- the per-agent singleton returns one shared emitter per agent name,
- the event payload carries lane/scope for feed routing.

All strings in THIS test file are literal — no f-strings.
"""

from __future__ import annotations

import asyncio

from paf.background_processes._events import (
    ProcessEvent,
    ProcessEventEmitter,
    ProcessEventType,
    get_agent_emitter,
)


def test_emit_process_dispatches_to_on_any_listener() -> None:
    """A listener registered via ``on_any`` receives an emitted transition."""

    async def run() -> None:
        emitter = ProcessEventEmitter()
        seen: list[ProcessEvent] = []
        emitter.on_any(seen.append)
        event = ProcessEvent(
            id="consolidation.demo.3",
            label="Consolidating memory",
            status=ProcessEventType.running,
            lane="agent",
        )
        await emitter.emit_process(event)
        assert seen == [event]

    asyncio.run(run())


def test_on_any_receives_every_status() -> None:
    """``on_any`` fires for queued/running/done/error/skipped alike, plus the
    turn_started re-attach signal (a transport event on the same bus)."""

    async def run() -> None:
        emitter = ProcessEventEmitter()
        statuses: list[str] = []
        emitter.on_any(lambda e: statuses.append(e.status.value))
        for status in ProcessEventType:
            await emitter.emit_process(ProcessEvent(id="p.1", label="P", status=status))
        assert statuses == [
            "queued",
            "running",
            "done",
            "error",
            "skipped",
            "turn_started",
        ]

    asyncio.run(run())


def test_off_any_releases_on_any_callback() -> None:
    """After ``off_any``, the callback no longer receives transitions."""

    async def run() -> None:
        emitter = ProcessEventEmitter()
        seen: list[ProcessEvent] = []
        cb = seen.append
        emitter.on_any(cb)
        assert emitter.off_any(cb) is True
        await emitter.emit_process(
            ProcessEvent(id="p.1", label="P", status=ProcessEventType.done)
        )
        assert seen == []

    asyncio.run(run())


def test_off_any_of_unregistered_callback_returns_false() -> None:
    """Removing a callback never registered reports False, no error."""

    def run() -> None:
        emitter = ProcessEventEmitter()
        assert emitter.off_any(lambda e: None) is False

    run()


def test_event_carries_lane_and_scope_for_routing() -> None:
    """The payload keeps lane/scope so the feed can route to the right panel."""

    async def run() -> None:
        emitter = ProcessEventEmitter()
        got: list[tuple[str, str | None]] = []
        emitter.on_any(lambda e: got.append((e.lane, e.scope)))
        await emitter.emit_process(
            ProcessEvent(
                id="thread_summarize.demo:paf-abc.2",
                label="Summarizing thread",
                status=ProcessEventType.queued,
                lane="conversation",
                scope="paf-abc",
            )
        )
        assert got == [("conversation", "paf-abc")]

    asyncio.run(run())


def test_get_agent_emitter_returns_shared_singleton_per_agent() -> None:
    """The per-agent emitter is one shared instance per agent name."""

    a1 = get_agent_emitter("demo_assistant")
    a2 = get_agent_emitter("demo_assistant")
    b = get_agent_emitter("assistant")
    assert a1 is a2
    assert a1 is not b
