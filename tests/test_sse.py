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
from paf.background_processes._events import (
    ProcessEvent,
    ProcessEventType,
    get_agent_emitter,
)
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


def test_turn_frames_does_not_forward_task_frames() -> None:
    """Background/maintenance task frames must NOT ride the turn's SSE stream.

    Phase 1 of the Unified-Background-Process-Feed-Spec decouples the two: a
    background ``task`` frame landing mid-turn used to be merged into ``/api/chat``
    and could tear a thinking/text block (the ``"The u" / "ser asks…"`` split by
    ``task:Summarizing thread``). ``_turn_frames`` now only drives the turn and
    never subscribes to the background coordinator — task frames flow on the
    dedicated ``/api/processes/stream`` feed instead (see
    ``test_processes_stream_*``).
    """

    subscribed = {"v": False}

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

    class _FakeSession:
        def __init__(self) -> None:
            self.background = self  # coordinator stub

        def stream(self, prompt, *, conversation_id=None):
            del prompt, conversation_id
            return _FakeTurn()

        def subscribe_tasks(self, cb) -> None:
            subscribed["v"] = True  # must never be called by _turn_frames

        async def cancel(self) -> None:
            pass

    frames = sse._turn_frames(_FakeSession(), "hi", "conv-1")  # type: ignore[attr-defined]

    async def _drive():
        got = await anext(frames)
        assert "partial" in got
        try:
            await anext(frames)
        except StopAsyncIteration:
            pass

    asyncio.run(_drive())

    assert subscribed["v"] is False, (
        "_turn_frames must not subscribe to background tasks"
    )


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
    # SSE tab joins mid-fold. Seed the registry directly rather than via
    # _publish_task, which now also appends to the run-event journal (a side
    # effect this SSE-surface test does not care about).
    task = BackgroundTask(
        id="consolidation.conv-1.3", label="Consolidating memory (conv-1)"
    )
    task.status = PROCESSING
    task.updated_at = "2026-09-17T00:00:00+00:00"
    coord._background_tasks[task.id] = task

    # Half 1: replay on subscribe; capture the replayed event(s).
    replayed: list = []

    def _on_task(ev) -> None:
        replayed.append(ev)

    coord.subscribe_tasks(_on_task)

    assert replayed, "active task must be replayed synchronously on subscribe"
    ev = replayed[0]
    assert ev.status == ProcessEventType.running

    # Half 2: the replayed event is a ProcessEvent; _task_frame maps it to the
    # wire frame the UI consumes (id/label/status/progress/detail).
    name, data = sse._task_frame(ev)
    assert name == "task"
    assert data["id"] == "consolidation.conv-1.3"
    assert data["label"] == "Consolidating memory (conv-1)"
    assert data["status"] == "running"

    # Half 3: task_history scopes to the conversation and returns the row
    # (the /api/conversations/{id}/tasks endpoint reads this through session.background).
    rows = coord.task_history("conv-1")
    assert [r.id for r in rows] == ["consolidation.conv-1.3"]
    assert coord.task_history("conv-other") == []


def test_process_feed_restores_history_then_streams_live_and_unsubscribes() -> None:
    """The dedicated process feed restores current rows, streams live
    transitions, stays open past the turn, and unsubscribes on teardown."""

    class _Coordinator:
        def __init__(self, history: list) -> None:
            self._history = history
            self.agent_name = "feed-test-agent"
            self.lane = "conversation"

        def task_history(self, conversation_id):
            assert conversation_id == "conv-1"
            return self._history

    class _Session:
        def __init__(self, coord) -> None:
            self.background = coord

        def _scoped_cid(self, cid):
            # bare session (no profile) => identity; mirrors Session._scoped_cid
            return cid

    class _Hist:
        def __init__(self, tid, status) -> None:
            self.id = tid
            self.label = f"task:Consolidating memory ({tid})"
            self.status = status
            self.progress = 0.5
            self.detail = "partial"
            self.updated_at = "2026-09-13T00:00:00"

    coord = _Coordinator([_Hist("consolidation.conv-1.1", "DONE")])
    emitter = get_agent_emitter("feed-test-agent")

    class _Req:
        async def is_disconnected(self) -> bool:
            # Stay connected for the whole test — the disconnect watcher must
            # not fire and cut this feed short.
            return False

    feed = sse._process_feed(coord, "conv-1", _Req())  # type: ignore[attr-defined]

    async def _drive():
        out = []

        # Pump frames from the feed as they arrive; drive via the shared emitter.
        async def _pump():
            async for f in feed:
                out.append(f)
                if len(out) >= 3:
                    return

        pump_task = asyncio.create_task(_pump())
        # Let the restore history frame arrive.
        await asyncio.sleep(0.01)

        # Fire live events on the per-agent emitter (simulate queued->running).
        def _ev(status):
            return ProcessEvent(
                id="consolidation.conv-1.2",
                label="Consolidating memory (conv-1)",
                status=status,
                progress=1.0,
                detail="5 turns -> 1 summary",
                lane="conversation",
                scope="conv-1",
            )

        await emitter.emit_process(_ev("running"))
        await emitter.emit_process(_ev("done"))

        await asyncio.wait_for(pump_task, timeout=2)
        return out

    out = asyncio.run(_drive())

    # Frame 1 = restored history row.
    assert "consolidation.conv-1.1" in out[0]
    # Frames 2-3 = live queued/running transition for a NEW row id.
    joined = "\n".join(out)
    assert "consolidation.conv-1.2" in joined
    assert "Consolidating memory" in joined
    # Teardown (generator exhausted after 3 frames) unsubscribes from the emitter.
    assert all(len(v) == 0 for v in emitter._listeners.values()), (
        "feed teardown must unsubscribe the subscriber"
    )


def test_process_feed_bare_session_streams_nothing_and_stays_alive() -> None:
    """A session with no background coordinator yields nothing (and doesn't
    crash), and the feed teardown is a clean no-op."""

    class _Session:
        background = None

    class _Req:
        async def is_disconnected(self) -> bool:
            return False

    feed = sse._process_feed(None, "conv-1", _Req())  # type: ignore[attr-defined]

    async def _drive():
        await feed.aclose()  # type: ignore[attr-defined] - must not raise on a bare session

    asyncio.run(_drive())


def test_process_feed_terminates_on_client_disconnect() -> None:
    """The never-ending processes feed must end when the client disconnects.

    Without disconnect handling the generator pins its SSE socket open forever,
    so uvicorn's graceful shutdown waits on it until systemd's TimeoutStopSec
    SIGKILLs the process (~90s restart stalls). is_disconnected() flips True a
    few polls in; the feed must terminate (not hang) within a bounded time.
    """

    class _Session:
        background = None

        def _scoped_cid(self, conversation_id: str) -> str:
            return conversation_id

    class _Req:
        def __init__(self) -> None:
            self.n = 0

        async def is_disconnected(self) -> bool:
            # First two polls say connected, then the client goes away.
            self.n += 1
            return self.n >= 3

    feed = sse._process_feed(None, "conv-1", _Req())  # type: ignore[attr-defined]

    async def _drive():
        out = []

        # Must settle on its own once the disconnect watcher fires — no aclose,
        # no manual cancel. Bounded wait proves it doesn't hang forever.
        async def _collect():
            async for f in feed:  # type: ignore[attr-defined]
                out.append(f)

        await asyncio.wait_for(_collect(), timeout=5)

    asyncio.run(_drive())
