"""Phase 1.4 spike: the durable-turn transport invariants, end to end.

Drives the REAL ``Session`` + ``TurnHandle`` + ``ui.sse._turn_frames`` machinery
(the production transport) with a scripted slow turn — the ONLY stub is the
model content source (``Session.stream``), which is inherently a stub here since
the spike must control timing deterministically and run offline (no API key).

Verifies the three spike outcomes from Durable-Server-Side-Turns-Reattach-Spec:
  (a) kill the client socket  -> the Session-owned turn KEEPS RUNNING
      (disconnect != cancel, the core invariant we broke in Phases 1.2/1.3)
  (b) a re-subscribe (a second /api/chat) catches the LIVE TAIL mid-turn
  (c) explicit cancel (POST /api/cancel -> Session.cancel) still stops the turn
"""

import asyncio

from paf.runtime import CancellationToken, Session, StreamEvent
from ui import sse


def _slow_turn_stream(*, n: int = 20, delay: float = 0.25):
    """A scripted slow turn: ``n`` text frames with a delay, honoring the token.

    Slow enough that the turn is still running across the disconnect/re-attach
    window, so the tail catch and cancel are observable.
    """

    async def _stream(prompt, *, conversation_id=None, _token=None):
        del prompt, conversation_id
        for i in range(n):
            if _token is not None and _token.cancelled:
                break
            yield StreamEvent(kind="text", text=f"frame-{i}")
            await asyncio.sleep(delay)

    return _stream


class _Request:
    """A Starlette Request stand-in with a toggleable disconnected flag."""

    def __init__(self) -> None:
        self._disconnected = False

    async def is_disconnected(self) -> bool:
        return self._disconnected


async def _drain(agen) -> None:
    async for _fr in agen:
        pass


def _bare_session() -> Session:
    """A REAL Session object, minimal (durable-turn fields only), no agent/store.

    ``start_turn``/``TurnHandle``/``subscribe``/``cancel`` all operate on these
    fields, so the durable-turn machinery is exercised for real.
    """
    sess = object.__new__(Session)
    sess._turn = None
    sess._token = CancellationToken()
    sess.conversation_id = "conv-1"
    sess.profile = None
    sess._scoped_cid = None  # type: ignore[method-assign]  # fall back to plain cid
    sess.stream = _slow_turn_stream()  # type: ignore[method-assign]
    return sess


def test_spike_detach_keeps_turn_then_reattach_catches_tail_then_cancel() -> None:

    async def scenario():
        sess = _bare_session()

        # Client A attaches (the first /api/chat). Read two frames, then "lose" it.
        req_a = _Request()
        frames_a = sse._turn_frames(sess, "hello", "conv-1", req_a)  # type: ignore[attr-defined]
        f0 = await anext(frames_a)
        f1 = await anext(frames_a)
        # Frames are delivered from the subscriber's attach point onward (a frame
        # emitted before subscribe is dropped for history replay), so only assert
        # they are distinct live text frames.
        assert "frame-" in f0 and "frame-" in f1 and f0 != f1

        # Kill client A's socket: mark it disconnected. Its SSE stream must end,
        # but the Session-owned turn must keep running server-side.
        req_a._disconnected = True
        await asyncio.wait_for(_drain(frames_a), timeout=5)

        handle = sess._turn
        assert handle is not None
        assert not handle.done, "(a) turn continues after the socket dies"
        assert handle.has_subscribers is False, "client A fully detached (no cancel)"

        # Client B re-attaches (a second /api/chat) mid-turn -> catches the LIVE TAIL.
        req_b = _Request()
        frames_b = sse._turn_frames(sess, "ignored", "conv-1", req_b)  # type: ignore[attr-defined]
        tail = [await anext(frames_b)]
        assert "frame-" in tail[0], "(b) re-subscribe catches the live tail"
        assert handle is sess._turn, "start_turn re-attached, not a second turn"
        # Let B observe several live frames before we cancel, to prove it is
        # really catching the in-flight stream (not an empty re-attach).
        try:
            while len(tail) < 3:
                tail.append(await asyncio.wait_for(anext(frames_b), timeout=1.0))
        except (TimeoutError, StopAsyncIteration):
            pass
        assert len(tail) >= 2, "re-attaching client saw multiple live tail frames"

        # Explicit cancel (POST /api/cancel) still stops the turn.
        await sess.cancel()
        async for fr in frames_b:
            tail.append(fr)
        assert handle.done, "(c) explicit cancel settles the turn"
        assert all("frame-" in fr for fr in tail), "every tail frame is a live frame"
        return tail

    tail = asyncio.run(scenario())
    assert any("frame-" in fr for fr in tail)
