"""Approval flow P4: the SSE transport.

Browser-started turns are interactive; approval requests and their outcomes ride
the turn stream as `approval_requested` / `approval_resolved` frames;
`POST /api/approvals/{id}` decides one (first decision wins); and
`GET /api/conversations/{cid}/approvals` lists what is pending for a client that
re-attaches mid-turn. The endpoints are called directly, on the turn's own event
loop, as the server would run them.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness import ToolGuardrail

from paf import runtime
from paf.approvals import (
    ApprovalAudit,
    ApprovalBroker,
    ApprovalRequest,
    ApprovalRequested,
    ApprovalResolved,
    approvals_available,
)
from paf.capabilities.guardrails.gating_policy import make_gating_guard, parse_policy
from paf.runtime import StreamEvent
from ui import sse


def _run(coro):
    return asyncio.run(coro)


def _req(**kw) -> ApprovalRequest:
    base = dict(
        id="ap1",
        tool_call_id="call-1",
        tool_name="write_file",
        args='{"path": "notes.txt"}',
        reason="gating policy",
        preview={"deletes": "build: 3 files"},
        created_at=1.0,
        expires_at=601.0,
        conversation_id="c1",
    )
    base.update(kw)
    return ApprovalRequest(**base)


# --- frames -------------------------------------------------------------------------------
def test_requested_frame_carries_what_the_card_needs():
    name, data = sse._serialize_turn(ApprovalRequested(_req()))
    assert name == "approval_requested"
    assert data == {
        "approval_id": "ap1",
        "step_id": "call-1",
        "tool_name": "write_file",
        "args": '{"path": "notes.txt"}',
        "reason": "gating policy",
        "preview": {"deletes": "build: 3 files"},
        "created_at": 1.0,
        "expires_at": 601.0,
        "conversation_id": "c1",
    }
    json.dumps(data)  # frames are JSON


def test_resolved_frame():
    name, data = sse._serialize_turn(
        ApprovalResolved(
            "ap1", "call-1", "write_file", "denied", actor="web", note="no"
        )
    )
    assert name == "approval_resolved"
    assert data == {
        "approval_id": "ap1",
        "step_id": "call-1",
        "tool_name": "write_file",
        "outcome": "denied",
        "actor": "web",
        "note": "no",
    }


class _Request:
    async def is_disconnected(self) -> bool:
        return False


def test_browser_turns_are_interactive():
    seen = {}

    class Handle:
        error = None

        async def subscribe(self):
            if False:
                yield

    class Session:
        current_turn = None

        def start_turn(self, prompt, *, conversation_id=None, interactive=False):
            seen["interactive"] = interactive
            return Handle()

    async def go():
        async for _ in sse._turn_frames(Session(), "hi", "c1", _Request()):
            pass

    _run(go())
    assert seen["interactive"] is True


# --- endpoints ----------------------------------------------------------------------------
class FakeSession:
    def __init__(self, pending=(), resolve_ok=True) -> None:
        self.pending = list(pending)
        self.resolve_ok = resolve_ok
        self.calls: list[tuple] = []

    def pending_approvals(self):
        return list(self.pending)

    def resolve_approval(self, approval_id, approved, *, note="", actor="user"):
        self.calls.append((approval_id, approved, note, actor))
        return self.resolve_ok


@pytest.fixture
def sessions(monkeypatch):
    table: dict = {}
    monkeypatch.setattr(sse, "_sessions", table)
    return table


def test_get_lists_pending_without_creating_a_session(sessions):
    sessions["demo_assistant:c1"] = FakeSession([_req()])
    out = _run(sse.conversation_approvals("c1", agent="demo_assistant"))
    assert [a["approval_id"] for a in out["approvals"]] == ["ap1"]
    assert _run(sse.conversation_approvals("nobody", agent="demo_assistant")) == {
        "approvals": []
    }
    assert "demo_assistant:nobody" not in sessions


@pytest.mark.parametrize(("decision", "approved"), [("approve", True), ("deny", False)])
def test_post_decides_with_the_web_as_actor(sessions, decision, approved):
    s = FakeSession([_req()])
    sessions["demo_assistant:c1"] = s
    out = _run(sse.decide_approval("ap1", {"decision": decision, "note": "because"}))
    assert out == {"ok": True, "approved": approved}
    assert s.calls == [("ap1", approved, "because", "web")]


def test_post_narrows_to_the_named_conversation(sessions):
    a, b = FakeSession([_req()]), FakeSession([_req()])
    sessions["demo_assistant:c1"], sessions["demo_assistant:c2"] = a, b
    _run(
        sse.decide_approval(
            "ap1",
            {"decision": "approve", "agent": "demo_assistant", "conversation_id": "c2"},
        )
    )
    assert a.calls == [] and len(b.calls) == 1


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"decision": "maybe"}, 422),
        ({}, 422),
        ({"decision": "approve", "note": 5}, 422),
        ({"decision": "approve"}, 404),  # nothing pending with that id
    ],
)
def test_post_rejects_bad_requests(sessions, body, code):
    sessions["demo_assistant:c1"] = FakeSession([_req(id="other")])
    with pytest.raises(HTTPException) as exc:
        _run(sse.decide_approval("ap1", body))
    assert exc.value.status_code == code


def test_post_after_someone_else_decided_is_a_conflict(sessions):
    sessions["demo_assistant:c1"] = FakeSession([_req()], resolve_ok=False)
    with pytest.raises(HTTPException) as exc:
        _run(sse.decide_approval("ap1", {"decision": "approve"}))
    assert exc.value.status_code == 409


# --- end to end: real turn, real frames, decided over the endpoint -----------------------------
def _session(tmp_path, executed):
    def write_file(path: str, content: str) -> str:
        executed.append(path)
        return f"wrote {path}"

    calls = [
        ToolCallPart(
            "write_file", {"path": "notes.txt", "content": "x"}, tool_call_id="call-9"
        )
    ]

    def respond(messages, info):
        return (
            ModelResponse(parts=[calls.pop(0)])
            if calls
            else ModelResponse(parts=[TextPart("done")])
        )

    gating = ToolGuardrail(
        guard=make_gating_guard(
            parse_policy({"rules": [{"tools": ["write_file"], "action": "approve"}]}),
            approval_available=approvals_available,
        )
    )
    agent = Agent(
        FunctionModel(respond),
        capabilities=[gating, ApprovalBroker()],
        tools=[write_file],
    )
    s = object.__new__(runtime.Session)
    s._turn, s._token = None, None
    s.conversation_id = "c1"
    s.profile = None
    s.agent_name = "demo_assistant"
    s._approval_timeout = 5.0
    s._approval_audit = ApprovalAudit(tmp_path / "a.jsonl")

    async def stream(prompt, *, conversation_id=None, _token=None):
        result = await agent.run(prompt)
        yield StreamEvent(kind="text", text=str(result.output))

    s.stream = stream
    return s


def _parse(frame: str) -> tuple[str, dict]:
    head, data = frame.strip().split("\n", 1)
    return head.removeprefix("event: "), json.loads(data.removeprefix("data: "))


@pytest.mark.parametrize(("decision", "ran"), [("approve", True), ("deny", False)])
def test_e2e_browser_turn_decided_over_the_endpoint(tmp_path, sessions, decision, ran):
    executed: list[str] = []
    session = _session(tmp_path, executed)
    sessions["demo_assistant:c1"] = session

    async def go():
        names = []
        async for frame in sse._turn_frames(session, "save notes", "c1", _Request()):
            name, data = _parse(frame)
            names.append(name)
            if name == "approval_requested":
                assert data["step_id"] == "call-9" and data["tool_name"] == "write_file"
                pending = await sse.conversation_approvals("c1", agent="demo_assistant")
                assert [a["approval_id"] for a in pending["approvals"]] == [
                    data["approval_id"]
                ]
                await sse.decide_approval(
                    data["approval_id"], {"decision": decision, "note": "web said so"}
                )
            if name == "approval_resolved":
                assert data["outcome"] == ("approved" if ran else "denied")
                assert data["actor"] == "web"
        return names

    names = _run(go())
    assert names[:2] == ["approval_requested", "approval_resolved"]
    assert executed == (["notes.txt"] if ran else [])


def test_e2e_a_second_decision_loses(tmp_path, sessions):
    executed: list[str] = []
    session = _session(tmp_path, executed)
    sessions["demo_assistant:c1"] = session

    async def go():
        async for frame in sse._turn_frames(session, "save notes", "c1", _Request()):
            name, data = _parse(frame)
            if name == "approval_requested":
                assert session.resolve_approval(data["approval_id"], True, actor="cli")
                with pytest.raises(HTTPException) as exc:
                    await sse.decide_approval(data["approval_id"], {"decision": "deny"})
                assert exc.value.status_code == 404  # no longer pending

    _run(go())
    assert executed == ["notes.txt"]  # the CLI's approval stood
