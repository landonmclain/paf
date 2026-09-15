"""Tests for the plan-live-sidebar SSE wiring (ui/sse.py plan frames + /plan).

Plan-Live-Sidebar-Spec: the authoritative ``FilePlanStore`` JSON is the single
source of truth; the Plans sidebar reads it from the server live. These tests
pin:

1. The conversation-scoped plan-emitter registry (``_registry``) — the bridge
   that lets the agent's store writer and the SSE observer share ONE emitter.
2. ``_turn_frames`` publishing a full-snapshot ``plan`` SSE frame on a plan
   mutation, mid-turn (live tick), then unsubscribing on teardown.
3. The REST ``/api/conversations/{id}/plan`` snapshot endpoint shape.
4. ``_serialize_plan_items`` producing the exact UI item shape.

All collection strings are literal — no f-strings.
"""

from __future__ import annotations

import asyncio
from unittest import mock

from paf.capabilities.planning import PlanEvent, PlanEventType, PlanItem
from paf.capabilities.planning._registry import clear_plan_emitters, get_plan_emitter
from ui import sse


def test_registry_returns_same_emitter_per_scoped_cid() -> None:
    """The registry hands back ONE emitter per scoped cid (writer/observer share it)."""
    clear_plan_emitters()
    a1 = get_plan_emitter("demo:abc123")
    a2 = get_plan_emitter("demo:abc123")
    b = get_plan_emitter("demo:xyz789")
    assert a1 is a2, "same scoped cid must resolve to the same emitter"
    assert a1 is not b, "different scoped cids must get different emitters"


def _fake_turn(block_on: asyncio.Event) -> object:
    """An async iterator that yields one event, then blocks until ``block_on``."""
    sent = {"v": False}

    class _Turn:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not sent["v"]:
                sent["v"] = True
                return sse.StreamEvent(kind="text", text="partial")
            await block_on.wait()  # suspend open (like stream() awaiting content)
            raise StopAsyncIteration

    return _Turn()


def _fake_session(block_on: asyncio.Event) -> object:
    class _Session:
        cancelled = {"v": False}

        def _scoped_cid(self, cid):
            return cid  # bare session -> identity (mirrors Session._scoped_cid)

        def stream(self, prompt, *, conversation_id=None):
            del prompt, conversation_id
            return _fake_turn(block_on)

        async def cancel(self) -> None:
            self.cancelled["v"] = True
            block_on.set()  # let the suspended stream settle

    return _Session()


def test_turn_frames_emits_plan_snapshot_on_mutation_and_unsubscribes() -> None:
    """A plan mutation mid-turn pushes a full-snapshot ``plan`` frame live."""
    clear_plan_emitters()
    block_on = asyncio.Event()
    sess = _fake_session(block_on)
    frames = sse._turn_frames(sess, "hi", "conv-1")  # type: ignore[attr-defined]

    plan_frame = {"items": [{"id": "c5b38e0a", "content": "step one", "status": "in_progress"}]}

    async def _drive():
        # 1. First frame = turn text.
        first = await anext(frames)
        assert "partial" in first
        # 2. Fire a plan mutation on the shared emitter (the agent's store does
        #    this in production). The SSE listener must push a snapshot frame.
        with mock.patch.object(sse, "_plan_snapshot", new=mock.AsyncMock(return_value=plan_frame["items"])):
            emitter = get_plan_emitter("conv-1")
            await emitter.emit(
                PlanEvent(
                    event_type=PlanEventType.status_changed,
                    item=PlanItem(id="c5b38e0a", content="step one", status="in_progress"),
                )
            )
            second = await asyncio.wait_for(anext(frames), timeout=2)
        assert "event: plan" in second
        assert "c5b38e0a" in second
        assert "in_progress" in second
        # 3. Tear down; the emitter subscription must be removed.
        await frames.aclose()
        # A subsequent same emitter must no longer call back into this stream.
        assert get_plan_emitter("conv-1")._listeners[PlanEventType.status_changed] == []

    asyncio.run(_drive())


def test_turn_frames_no_plan_frame_without_mutation() -> None:
    """No plan event -> no ``plan`` frame rides the turn stream."""
    clear_plan_emitters()
    block_on = asyncio.Event()
    sess = _fake_session(block_on)
    frames = sse._turn_frames(sess, "hi", "conv-1")  # type: ignore[attr-defined]

    async def _drive():
        first = await anext(frames)
        assert "partial" in first
        block_on.set()  # let the turn end
        with mock.patch.object(sse, "_plan_snapshot", new=mock.AsyncMock(return_value=[])):
            got = []
            try:
                while True:
                    got.append(await anext(frames))
            except StopAsyncIteration:
                pass
        assert all("event: plan" not in f for f in got), "no plan event -> no plan frame"

    asyncio.run(_drive())


def test_serialize_plan_items_shape() -> None:
    """_serialize_plan_items produces the exact UI item shape with real ids."""
    items = [
        PlanItem(id="aaaa1111", content="c1", status="completed"),
        PlanItem(id="bbbb2222", content="c2", status="pending", depends_on=["aaaa1111"]),
    ]
    out = sse._serialize_plan_items(items)
    assert out == [
        {
            "id": "aaaa1111",
            "content": "c1",
            "status": "completed",
            "active_form": "",
            "parent_id": None,
            "depends_on": [],
        },
        {
            "id": "bbbb2222",
            "content": "c2",
            "status": "pending",
            "active_form": "",
            "parent_id": None,
            "depends_on": ["aaaa1111"],
        },
    ]


def test_rest_endpoint_returns_authoritative_snapshot() -> None:
    """GET /api/conversations/{id}/plan returns the server-truth items shape."""
    payload = {
        "conversation_id": "conv-1",
        "agent": "demo_assistant",
        "items": [{"id": "c5b38e0a", "content": "step one", "status": "completed"}],
        "item_count": 1,
    }

    class _Sess:
        def _scoped_cid(self, cid):
            return f"demo:{cid}"

    async def _drive():
        with mock.patch.object(sse, "_get_session", new=mock.AsyncMock(return_value=_Sess())), mock.patch.object(
            sse, "_plan_snapshot", new=mock.AsyncMock(return_value=payload["items"])
        ):
            resp = await sse.conversation_plan("conv-1")  # type: ignore[attr-defined]
        assert resp == payload
        assert resp["item_count"] == 1

    asyncio.run(_drive())
