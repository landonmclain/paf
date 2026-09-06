"""Tests for the SSE transport teardown path (ui/sse.py ``_turn_frames``).

The focus is the teardown contract that fixed "tracing sometimes just stops":
when the turn's SSE generator is torn down mid-suspension (Stop / client
disconnect) the code MUST cancel the session *cooperatively* via
``session.cancel()`` (CancellationToken) rather than injecting a foreign-task
``asyncio.CancelledError`` into the ``session.stream()`` generator. Cancelling
the task directly is what made OTel's ``context.detach()`` raise
``ValueError: <Token> was created in a different Context`` (the span never ended
--> trace lost) and produced ``aclose(): asynchronous generator is already
running``. We assert the teardown settles cleanly and triggers the cooperative
cancel path.
"""

from __future__ import annotations

import asyncio

from paf.runtime import StreamEvent
from ui import sse


def _event() -> StreamEvent:
    return StreamEvent(kind="text", text="partial")


def test_turn_frames_teardown_cancels_cooperatively_and_settles() -> None:
    """Closing the SSE generator mid-turn must use session.cancel(), not a
    foreign Task.cancel(), and must settle without leaking CancelledError."""

    class _FakeTurn:
        """async iterator that yields one event then blocks mid-suspension."""

        def __init__(self, released: asyncio.Event) -> None:
            self._released = released
            self._sent = False
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._sent:
                self._sent = True
                return _event()
            # Suspend open (equivalent to stream() yielding a chunk mid-turn).
            await self._released.wait()
            raise StopAsyncIteration

    class _FakeSession:
        def __init__(self) -> None:
            self.cancel_called = False
            self.task_subscribed = None
            self._released = asyncio.Event()

        def stream(self, prompt, *, conversation_id=None):
            del prompt, conversation_id
            return _FakeTurn(self._released)

        def subscribe_tasks(self, cb) -> None:
            self.task_subscribed = cb

        async def cancel(self) -> None:
            self.cancel_called = True
            self._released.set()  # let the suspended stream() complete/close

    s = _FakeSession()
    frames = sse._turn_frames(s, "hi", "conv-1")  # type: ignore[attr-defined]

    async def _drive():
        # Pull one frame -> the turn is now suspended mid-turn (Stop case).
        first = await anext(frames)
        assert "partial" in first
        # Now tear the SSE connection down (client disconnect / Stop).
        await frames.aclose()

    asyncio.run(_drive())

    assert s.cancel_called is True, "teardown must cancel the session cooperatively"
    # The stream task must have settled (no leaked CancelledError from aclose).


def test_turn_frames_drains_queued_task_frames_on_turn_end() -> None:
    """Task frames still queued when the turn ends must be yielded, not dropped.

    Regression for the \"process row stuck running/silent\" gap: a background
    process (e.g. memory consolidation) that transitions running->done as the
    turn finishes must surface its final state before the SSE stream closes.
    """

    class _FakeTurn:
        def __init__(self) -> None:
            self._sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._sent:
                self._sent = True
                return _event()
            raise StopAsyncIteration

    class _FakeTask:
        def __init__(self) -> None:
            self.kind = "task:consolidating memory"
            self.status = "DONE"
            self.detail = "5 turns -> 1 summary"
            self.progress = 1.0
            self.metadata = {"id": "mem-1", "progress": 1.0}

    frame = "NO_TASK_FRAME"

    class _FakeSession:
        def __init__(self) -> None:
            self.task_subscribed = None

        def stream(self, prompt, *, conversation_id=None):
            del prompt, conversation_id
            return _FakeTurn()

        def subscribe_tasks(self, cb) -> None:
            self.task_subscribed = cb

        async def cancel(self) -> None:
            pass

    s = _FakeSession()
    frames = sse._turn_frames(s, "hi", "conv-1")  # type: ignore[attr-defined]

    async def _drive():
        nonlocal frame
        # Grab the text frame first.
        got = await anext(frames)
        assert "partial" in got
        # Fire a DONE task frame via the subscribed callback (queues it).
        s.task_subscribed(_FakeTask())
        # Then drive to the end; the queued task frame must be drained+yielded.
        frame = await anext(frames)
        try:
            await anext(frames)
        except StopAsyncIteration:
            pass

    asyncio.run(_drive())

    assert frame != "NO_TASK_FRAME", "queued task frame was dropped at turn end"
    assert "consolidating memory" in frame and "DONE" in frame


def test_turn_frames_teardown_survives_no_active_turn() -> None:
    """Closing a generator whose turn already finished must be a clean no-op."""

    class _FakeSession:
        def stream(self, prompt, *, conversation_id=None):
            del prompt, conversation_id

            async def _g():
                yield _event()

            return _g()

        def subscribe_tasks(self, cb) -> None:
            del cb

        async def cancel(self) -> None:
            pass

    frames = sse._turn_frames(_FakeSession(), "hi", "conv-1")  # type: ignore[attr-defined]

    async def _drive():
        await anext(frames)
        await frames.aclose()  # turn task already done -> must not raise

    asyncio.run(_drive())
