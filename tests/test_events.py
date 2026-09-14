"""Unit tests for the generic EventEmitter primitive (``src/paf/events.py``).

This is the shared observation seam extracted from the plan emitter. The
planning-specific behavior (``PlanEventEmitter`` emission from the store) is
covered in ``test_planning_capability.py``; these tests exercise the generic
layer itself:

- dispatch to multiple listeners in registration order,
- async listener awaiting (and mixed sync/async ordering),
- the listener-exception isolation guarantee (one failing listener never
  breaks the emitter or the other listeners, and never raises out of emit),
- ``off`` / decorator-as-``on``,
- type discrimination (emit to one key never reaches another key's listeners),
- a non-plan specialization proving the primitive is genuinely generic.

All strings in THIS test file are literal — no f-strings.
"""

from __future__ import annotations

import asyncio

import pytest

from paf.events import GenericEventEmitter

# --- basic dispatch -----------------------------------------------------------


def test_dispatches_to_all_listeners_in_registration_order() -> None:
    """Listeners for an event fire in the order they were registered."""

    async def run() -> None:
        emitter = GenericEventEmitter[str, int]()
        order: list[str] = []
        emitter.on("tick", lambda n: order.append("first"))
        emitter.on("tick", lambda n: order.append("second"))
        emitter.on("tick", lambda n: order.append("third"))
        await emitter.emit(42, "tick")
        assert order == ["first", "second", "third"]

    asyncio.run(run())


def test_payload_passed_through_to_listener() -> None:
    """The emitted payload reaches the listener unchanged."""

    async def run() -> None:
        emitter = GenericEventEmitter[str, dict]()
        seen: list[dict] = []
        emitter.on("evt", seen.append)
        await emitter.emit({"a": 1}, "evt")
        assert seen == [{"a": 1}]

    asyncio.run(run())


def test_emit_to_one_type_does_not_reach_other_types() -> None:
    """Type discrimination: listeners registered under another key stay silent."""

    async def run() -> None:
        emitter = GenericEventEmitter[str, str]()
        got: list[str] = []
        emitter.on("a", got.append)
        emitter.on("b", lambda s: got.append("WRONG"))
        await emitter.emit("payload", "a")
        assert got == ["payload"]

    asyncio.run(run())


# --- async listeners ----------------------------------------------------------


def test_async_listener_is_awaited() -> None:
    """A coroutine-function listener is awaited so its side effect completes."""

    async def run() -> None:
        emitter = GenericEventEmitter[str, int]()
        done: list[int] = []

        async def listener(n: int) -> None:
            await asyncio.sleep(0)
            done.append(n)

        emitter.on("tick", listener)
        await emitter.emit(7, "tick")
        assert done == [7]

    asyncio.run(run())


def test_sync_and_async_listeners_run_in_registration_order() -> None:
    """Mixed sync/async listeners all run, in registration order."""

    async def run() -> None:
        emitter = GenericEventEmitter[str, int]()
        order: list[str] = []

        async def second(n: int) -> None:
            await asyncio.sleep(0)
            order.append("async-second")

        emitter.on("x", lambda n: order.append("sync-first"))
        emitter.on("x", second)
        emitter.on("x", lambda n: order.append("sync-third"))
        await emitter.emit(1, "x")
        assert order == ["sync-first", "async-second", "sync-third"]

    asyncio.run(run())


# --- listener-exception isolation ---------------------------------------------


def test_failing_listener_does_not_stop_later_listeners() -> None:
    """The core isolation guarantee: one raising listener must not block others."""

    async def run() -> None:
        emitter = GenericEventEmitter[str, int]()
        reached: list[str] = []

        def boom(n: int) -> None:
            raise RuntimeError("subscriber bug")

        emitter.on("tick", boom)
        emitter.on("tick", lambda n: reached.append("after"))
        # The failing listener is first, but the later one must still run and the
        # emit must not raise out to the caller.
        await emitter.emit(1, "tick")
        assert reached == ["after"]

    asyncio.run(run())


def test_failing_async_listener_does_not_stop_later_listeners() -> None:
    """The isolation guarantee also holds when the failing listener is async."""

    async def run() -> None:
        emitter = GenericEventEmitter[str, int]()
        reached: list[str] = []

        async def boom(n: int) -> None:
            raise ValueError("async subscriber bug")

        emitter.on("tick", boom)
        emitter.on("tick", lambda n: reached.append("survivor"))
        await emitter.emit(1, "tick")
        assert reached == ["survivor"]

    asyncio.run(run())


def test_failing_listener_does_not_raise_out_of_emit() -> None:
    """A subscriber exception is swallowed: emit returns normally."""

    async def run() -> None:
        emitter = GenericEventEmitter[str, int]()

        def boom(n: int) -> None:
            raise RuntimeError("boom")

        emitter.on("tick", boom)
        await emitter.emit(1, "tick")  # must not raise

    asyncio.run(run())


# --- off / decorator ----------------------------------------------------------


def test_off_removes_a_registered_listener() -> None:
    """After ``off``, the listener no longer receives events."""

    async def run() -> None:
        emitter = GenericEventEmitter[str, int]()
        seen: list[int] = []
        listener = lambda n: seen.append(n)  # noqa: E731 - intentional alias
        emitter.on("tick", listener)
        assert emitter.off("tick", listener) is True
        await emitter.emit(1, "tick")
        assert seen == []

    asyncio.run(run())


def test_off_of_unregistered_listener_returns_false() -> None:
    """Removing a listener that was never registered reports False, no error."""

    def run() -> None:
        emitter = GenericEventEmitter[str, int]()
        assert emitter.off("tick", lambda n: None) is False
        # Emitting with no listeners for a key is a silent no-op.
        asyncio.run(emitter.emit(1, "tick"))

    run()


def test_on_returns_the_registered_callback() -> None:
    """``on`` returns the callback it registered (assignment / decorator-factory use)."""

    async def run() -> None:
        emitter = GenericEventEmitter[str, str]()
        got: list[str] = []

        def listener(s: str) -> None:
            got.append(s)

        returned = emitter.on("tick", listener)
        assert returned is listener
        await emitter.emit("via-return", "tick")
        assert got == ["via-return"]

    asyncio.run(run())


# --- genericity across a non-plan specialization -------------------------------


def test_generic_across_a_simple_string_discriminant() -> None:
    """A str-discriminant specialization (e.g. a shell/test process emitter) works.

    This demonstrates the primitive is not plan-shaped: any process/tool that
    wants shared visibility can subtype ``GenericEventEmitter`` with its own
    key enum / payload and get the same dispatch + isolation guarantees.
    """

    async def run() -> None:
        # Stand-in for a docker-shell process emitter: keys are phases.
        phases: dict[str, list[int]] = {"start": [], "done": []}
        emitter = GenericEventEmitter[str, int]()

        def on_start(cid: int) -> None:
            phases["start"].append(cid)

        def on_done(cid: int) -> None:
            phases["done"].append(cid)

        emitter.on("start", on_start)
        emitter.on("done", on_done)
        await emitter.emit(100, "start")
        await emitter.emit(100, "done")
        assert phases == {"start": [100], "done": [100]}

    asyncio.run(run())


def test_emit_with_no_listeners_for_key_is_noop() -> None:
    """Emitting a key nobody has subscribed to is a silent no-op."""

    async def run() -> None:
        emitter = GenericEventEmitter[str, int]()
        emitter.on("other", lambda n: pytest.fail("must not fire"))
        await emitter.emit(9, "unsubscribed")  # no-op, no exception

    asyncio.run(run())
