"""Tests for agent-chosen push: the ``Push`` capability / ``send_push`` tool, the
``AgentPusher`` guardrails, the profile gate, and the Session hook that marks a
turn ``pushed`` (so the settle push doesn't double-notify).

Network-free: delivery goes through an injected fake sender.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from paf import push, runtime
from paf.capabilities import PROFILES
from paf.capabilities._storage import FlatFileStorage
from paf.capabilities.build import build_runtime_capabilities
from paf.capabilities.profiles import AgentProfile
from paf.capabilities.push import Push, PushToolset
from paf.capabilities.registry import CAPABILITY_TYPES
from paf.push import (
    AgentPusher,
    PushRequest,
    PushResult,
    PushService,
    PushStore,
    VapidConfig,
)

CONFIG = VapidConfig("priv", "pub", "mailto:me@example.com")
P256DH = push.b64url_encode(b"\x04" + b"a" * 64)
AUTH = push.b64url_encode(b"b" * 16)


def _sub(n: int = 1) -> dict[str, Any]:
    return {
        "endpoint": f"https://fcm.googleapis.com/fcm/send/dev-{n}",
        "keys": {"p256dh": P256DH, "auth": AUTH},
    }


class _Sender:
    def __init__(self, fail_status: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_status = fail_status

    async def __call__(self, **kw: Any) -> None:
        self.calls.append(kw)
        if self.fail_status:
            err = RuntimeError("push failed")
            err.status_code = self.fail_status  # type: ignore[attr-defined]
            raise err


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _pusher(tmp_path, *, devices: int = 1, config=CONFIG, sender=None, **kw):
    sender = sender or _Sender()
    svc = PushService(PushStore(FlatFileStorage(tmp_path)), config, send=sender)
    for n in range(devices):
        svc.store.add(_sub(n))
    clock = _Clock()
    return AgentPusher(lambda: svc, clock=clock, **kw), sender, clock


def _req(**kw: Any) -> PushRequest:
    base: dict[str, Any] = dict(
        agent="assistant",
        conversation_id="c-1",
        title="Price alert",
        body="BTC hit 100k",
    )
    base.update(kw)
    return PushRequest(**base)


# --- AgentPusher guardrails ---------------------------------------------------
def test_sent_payload_has_fixed_deeplink_and_urgency(tmp_path) -> None:
    p, sender, _ = _pusher(tmp_path)
    res = asyncio.run(p(_req(urgent=True)))
    assert res == PushResult("sent", 1)
    (call,) = sender.calls
    data = json.loads(call["data"])
    assert data == {
        "title": "Price alert",
        "body": "BTC hit 100k",
        "agent": "assistant",
        "url": "/?agent=assistant&c=c-1",
        "cid": "c-1",
        "tag": "c-1",
    }
    assert call["headers"] == {"Urgency": "high"}


def test_no_conversation_falls_back_to_root_link(tmp_path) -> None:
    p, sender, _ = _pusher(tmp_path)
    asyncio.run(p(_req(conversation_id=None)))
    data = json.loads(sender.calls[0]["data"])
    assert data["url"] == "/" and "cid" not in data and "tag" not in data


def test_text_is_capped_and_cleaned_never_a_link(tmp_path) -> None:
    p, sender, _ = _pusher(tmp_path)
    asyncio.run(p(_req(title="**" + "T" * 200, body="b " * 500)))
    data = json.loads(sender.calls[0]["data"])
    assert len(data["title"]) <= 60 and len(data["body"]) <= 200
    assert "*" not in data["title"]
    # The agent has no way to set the URL: it is built from the conversation id.
    assert data["url"].startswith("/?agent=")


def test_unconfigured_and_no_devices_do_not_burn_budget(tmp_path) -> None:
    p, _, _ = _pusher(tmp_path, config=None)
    assert asyncio.run(p(_req())).status == "not_configured"
    p2, sender2, _ = _pusher(tmp_path / "x", devices=0, max_per_hour=1)
    for _ in range(3):
        assert asyncio.run(p2(_req())).status == "no_devices"
    assert sender2.calls == []


def test_failed_send_does_not_burn_budget_or_dedupe(tmp_path) -> None:
    p, sender, _ = _pusher(tmp_path, sender=_Sender(fail_status=500), max_per_hour=1)
    assert asyncio.run(p(_req())).status == "failed"
    assert asyncio.run(p(_req())).status == "failed"  # not "duplicate"/"rate_limited"
    assert len(sender.calls) == 2


def test_duplicate_suppressed_then_allowed_after_window(tmp_path) -> None:
    p, _, clock = _pusher(tmp_path, dedupe_seconds=600)
    assert asyncio.run(p(_req())).status == "sent"
    clock.t += 599
    assert asyncio.run(p(_req())).status == "duplicate"
    assert asyncio.run(p(_req(body="different body"))).status == "sent"
    clock.t += 700
    assert asyncio.run(p(_req())).status == "sent"


def test_rate_limit_per_agent_rolling_hour(tmp_path) -> None:
    p, _, clock = _pusher(tmp_path, max_per_hour=3, dedupe_seconds=0)
    for i in range(3):
        assert asyncio.run(p(_req(body=f"n{i}"))).status == "sent"
    assert asyncio.run(p(_req(body="n3"))).status == "rate_limited"
    # Another agent has its own budget.
    assert asyncio.run(p(_req(agent="demo_assistant", body="n3"))).status == "sent"
    clock.t += 3601  # the window rolls
    assert asyncio.run(p(_req(body="n4"))).status == "sent"


# --- the tool -----------------------------------------------------------------
@pytest.mark.parametrize(
    ("status", "needle"),
    [
        ("sent", "sent to 2 device"),
        ("rate_limited", "limit"),
        ("duplicate", "identical"),
        ("no_devices", "no device"),
        ("not_configured", "aren't configured"),
        ("failed", "didn't accept"),
    ],
)
def test_tool_reports_every_status_to_the_model(status: str, needle: str) -> None:
    async def notifier(title: str, body: str, urgent: bool) -> PushResult:
        return PushResult(status, 2)  # type: ignore[arg-type]

    out = asyncio.run(PushToolset(notifier).send_push("t", "b"))
    assert needle in out


def test_capability_has_no_toolset_when_unwired() -> None:
    assert Push(notifier=None).get_toolset() is None
    assert Push.get_serialization_name() == "Push"
    assert "Push" in CAPABILITY_TYPES


def test_agent_run_calls_send_push_through_the_capability() -> None:
    """A real pydantic-ai run: the model calls send_push and the notifier sees it."""
    seen: list[tuple[str, str, bool]] = []

    async def notifier(title: str, body: str, urgent: bool) -> PushResult:
        seen.append((title, body, urgent))
        return PushResult("sent", 1)

    agent = Agent(
        TestModel(call_tools=["send_push"]),
        capabilities=[Push(notifier=notifier)],
    )
    result = agent.run_sync("go")
    assert (
        len(seen) == 1 and isinstance(seen[0][0], str) and isinstance(seen[0][2], bool)
    )
    assert "Notification sent to 1 device" in result.output


# --- the profile is the gate ------------------------------------------------------
def test_profile_gate_controls_whether_the_tool_exists() -> None:
    async def notifier(t: str, b: str, u: bool) -> PushResult:
        return PushResult("sent", 1)

    with_push = AgentProfile(name="t1", capability_names=("Push",))
    without = AgentProfile(name="t2", capability_names=("StepPersistence",))
    # ApprovalBroker is plumbing on every agent (see test_build_agent); the
    # profile decides everything else.
    from paf.approvals import ApprovalBroker

    caps = [
        c
        for c in build_runtime_capabilities(with_push, push_notifier=notifier)
        if not isinstance(c, ApprovalBroker)
    ]
    assert [type(c) for c in caps] == [Push] and caps[0].get_toolset() is not None
    assert not any(
        isinstance(c, Push)
        for c in build_runtime_capabilities(without, push_notifier=notifier)
    )
    # Wired nowhere (CLI): the capability builds but offers no tool.
    assert build_runtime_capabilities(with_push)[0].get_toolset() is None


def test_which_shipped_profiles_can_push() -> None:
    can = {n for n, p in PROFILES.items() if "Push" in p.capability_names}
    assert can == {"assistant", "demo"}  # secure + benchmark profiles stay silent


# --- Session hook: marks the turn so the settle push doesn't double-notify ----------
def _session(notifier):
    s = object.__new__(runtime.Session)
    s._turn = None
    s._token = None
    s.agent_name = "assistant"
    s.conversation_id = "c-session"
    s._push_notifier = notifier
    s._on_turn_settled = None
    return s


def test_notify_user_marks_turn_pushed_only_when_sent() -> None:
    reqs: list[PushRequest] = []

    def notifier_returning(status: str):
        async def n(req: PushRequest) -> PushResult:
            reqs.append(req)
            return PushResult(status, 1)  # type: ignore[arg-type]

        return n

    async def go(status: str) -> bool:
        s = _session(notifier_returning(status))

        async def stream(prompt, *, conversation_id=None, _token=None):  # noqa: ARG001
            await s._notify_user("T", "B", False)
            yield runtime.StreamEvent(kind="text", text="done")

        s.stream = stream  # type: ignore[method-assign]
        h = s.start_turn("p", conversation_id="c-turn")
        async for _ in h.subscribe():
            pass
        return h.pushed

    assert asyncio.run(go("sent")) is True
    assert asyncio.run(go("rate_limited")) is False  # the auto settle push stays on
    # The running turn's conversation wins over the session default.
    assert reqs[0].conversation_id == "c-turn" and reqs[0].agent == "assistant"


def test_notify_user_falls_back_to_session_conversation_without_a_turn() -> None:
    reqs: list[PushRequest] = []

    async def n(req: PushRequest) -> PushResult:
        reqs.append(req)
        return PushResult("sent", 1)

    asyncio.run(_session(n)._notify_user("T", "B", True))
    assert reqs[0].conversation_id == "c-session" and reqs[0].urgent is True


def test_turn_where_agent_pushed_gets_no_settle_push() -> None:
    """End to end through the policy: pushed=True reaches on_turn_settled -> quiet."""

    async def n(req: PushRequest) -> PushResult:
        return PushResult("sent", 1)

    s = _session(n)
    got: list[runtime.TurnSettled] = []

    async def cb(settled: runtime.TurnSettled) -> None:
        got.append(settled)

    s._on_turn_settled = cb

    async def stream(prompt, *, conversation_id=None, _token=None):  # noqa: ARG001
        await s._notify_user("T", "B", False)
        yield runtime.StreamEvent(kind="text", text="All done.")

    s.stream = stream  # type: ignore[method-assign]

    async def go() -> None:
        h = s.start_turn("p")
        async for _ in h.subscribe():
            pass
        await asyncio.sleep(0)
        await asyncio.gather(*runtime._SETTLED_TASKS)

    asyncio.run(go())
    assert got[0].pushed is True and got[0].outcome == "ok"
    assert push.settled_payload(got[0]) is None  # the agent's push was the only one


# --- default budget + journal visibility ---------------------------------------------
def test_default_budget_is_ten_per_hour(tmp_path) -> None:
    p, _, _ = _pusher(tmp_path, dedupe_seconds=0)  # no max_per_hour override
    for i in range(10):
        assert asyncio.run(p(_req(body=f"n{i}"))).status == "sent"
    assert asyncio.run(p(_req(body="n10"))).status == "rate_limited"


def test_each_agent_push_attempt_is_logged(tmp_path, caplog) -> None:
    import logging

    p, _, _ = _pusher(tmp_path, max_per_hour=1, dedupe_seconds=0)
    with caplog.at_level(logging.INFO, logger="paf.push"):
        asyncio.run(p(_req(title="Alpha")))
        asyncio.run(p(_req(title="Beta", body="x")))  # over the budget
    lines = [r.getMessage() for r in caplog.records if r.name == "paf.push"]
    assert "agent push assistant:c-1 sent sent=1 title='Alpha'" in lines
    assert "agent push assistant:c-1 rate_limited sent=0 title='Beta'" in lines


def test_sse_makes_push_info_logs_reach_the_journal() -> None:
    import logging

    from ui import sse

    for name in ("ui.sse", "paf.push"):
        lg = logging.getLogger(name)
        assert lg.isEnabledFor(logging.INFO)
        before = len(lg.handlers)
        sse._enable_info_logs(name)  # idempotent: no duplicate handler
        assert len(lg.handlers) == before >= 1


# --- agent push respects presence ----------------------------------------------------
def test_agent_push_is_not_sent_to_a_device_that_is_watching(tmp_path) -> None:
    p, sender, _ = _pusher(tmp_path, devices=2)
    svc = p._service()
    svc.presence.report(_sub(0)["endpoint"], "assistant", "c-1", True)
    assert asyncio.run(p(_req())).status == "sent"
    assert [c["subscription_info"]["endpoint"] for c in sender.calls] == [
        _sub(1)["endpoint"]
    ]


def test_agent_push_watching_status_when_all_devices_watch(tmp_path) -> None:
    p, sender, _ = _pusher(tmp_path, max_per_hour=1, dedupe_seconds=0)
    svc = p._service()
    svc.presence.report(_sub(0)["endpoint"], "assistant", "c-1", True)
    assert asyncio.run(p(_req())).status == "watching"
    assert sender.calls == []
    # It spent no budget and wasn't remembered: once they look away, it goes out.
    svc.presence.report(_sub(0)["endpoint"], "assistant", "c-1", False)
    assert asyncio.run(p(_req())).status == "sent"


def test_tool_tells_the_agent_the_user_is_watching() -> None:
    async def notifier(title: str, body: str, urgent: bool) -> PushResult:
        return PushResult("watching")

    out = asyncio.run(PushToolset(notifier).send_push("t", "b"))
    assert "looking at this conversation" in out and "No notification is needed" in out
