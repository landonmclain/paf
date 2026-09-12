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

from paf.background import PROCESSING, BackgroundCoordinator, BackgroundTask
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
            self.background = self  # bare coordinator stub: subscribe routes here

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
            self.background = self  # bare coordinator stub

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
        def __init__(self) -> None:
            self.background = self  # bare coordinator stub

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


def test_background_coordinator_task_bytes_identical_via_session_slot() -> None:
    """Regression (BackgroundCoordinator extraction): a real coordinator reached
    through the ``session.background`` slot must replay active tasks to
    ``subscribe_tasks`` subscribers and produce byte-identical ``task:`` SSE
    frames as before the refactor (when subscribe_tasks/task_history lived ON the
    Session as forwarders).

    The pre-refactor contract had two halves we pin here on the coordinator:
    1. ``subscribe_tasks`` REPLAYS already-active (queued/running) tasks
       SYNCHRONOUSLY on subscribe, so a late-joining SSE tab sees \"running\".
    2. sse._task_frame maps the replayed BackgroundEvent to the same wire frame
       sse._frame would emit from the event directly — byte-identical.
    """

    class _StubStore:
        def list_runs(self, conversation_id=None):
            return []

    class _Profile:
        capability_names = ("Memory",)  # memory-only; no ThreadSummarizer

    coord = BackgroundCoordinator(
        _StubStore(),
        agent_name="assistant",
        # A memory-capable profile so the coordinator is non-bare (mirrors a
        # real session the sse layer would attach). We don't need for_session
        # here — the point is the coordinator's OWN subscribe/task_history
        # surface, which sse binds to via session.background.
        profile=_Profile(),
    )

    # Simulate a background job already in flight (registered + promoted to
    # PROCESSING), exactly as the consolidation trigger would leave it when an
    # SSE tab joins mid-fold.
    task = BackgroundTask(id="consolidation.conv-1.3", label="Consolidating memory (conv-1)")
    task.status = PROCESSING
    coord._publish_task(task)

    # Half 1: replay on subscribe; capture the replayed event(s).
    replayed: list = []

    def _on_task(ev) -> None:
        replayed.append(ev)

    coord.subscribe_tasks(_on_task)

    assert replayed, "active task must be replayed synchronously on subscribe"
    ev = replayed[0]
    assert ev.status == PROCESSING

    # Half 2: the SSE wire bytes from this coordinator flow are byte-identical
    # to building the frame directly from the same event via sse's own
    # serializer (the path _turn_frames uses).
    name, data = sse._task_frame(ev)
    assert name == "task"
    assert data["id"] == "consolidation.conv-1.3"
    via_coordinator = sse._frame(name, data)

    # Rebuild the exact frame from the task's own as_event() to prove the byte
    # contract holds all the way from the coordinator's registry -> replay ->
    # serialize -> frame (no shape drift in the move).
    ev2 = task.as_event()
    name2, data2 = sse._task_frame(ev2)
    direct = sse._frame(name2, data2)
    assert via_coordinator == direct, "coordinator replay frame differs from direct serialization"

    # Half 3: task_history scopes to the conversation and returns the row
    # (the /api/conversations/{id}/tasks endpoint reads this through session.background).
    rows = coord.task_history("conv-1")
    assert [r.id for r in rows] == ["consolidation.conv-1.3"]
    assert coord.task_history("conv-other") == []
