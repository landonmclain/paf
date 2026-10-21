"""Approval flow P1: the runtime core (paf.approvals + Session/TurnHandle).

Covers the per-turn channel (decide, deny, timeout, stop, first decision wins,
audit, spans), the broker (deny-all when nobody is attached), real guards asking
through a real agent run (gating `approve`, InjectionGuard tightening, batches),
and the Session path end to end: an interactive turn publishes the request to
its subscribers, `Session.resolve_approval` settles it, a stop denies it, and
the SSE idle reaper treats a waiting turn as busy.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent, DeferredToolRequests, ToolDenied
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness import ToolGuardrail

import paf.approvals as approvals_mod
from paf import runtime
from paf.approvals import (
    DECISION_SPAN,
    REQUEST_SPAN,
    ApprovalAudit,
    ApprovalBroker,
    ApprovalChannel,
    ApprovalRequested,
    ApprovalResolved,
    approvals_available,
    bind_channel,
    current_channel,
)
from paf.capabilities.guardrails.gating_policy import make_gating_guard, parse_policy
from paf.capabilities.injection import InjectionGuard, Score, parse_injection_policy
from paf.runtime import StreamEvent


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def spans(monkeypatch) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(approvals_mod, "_TRACER", provider.get_tracer("paf"))
    return exporter


class Recorder:
    """An emit callback that records events and optionally answers requests."""

    def __init__(self, decide=None) -> None:
        self.events: list = []
        self.decide = decide  # (request) -> (approved, note) | None
        self.channel: ApprovalChannel | None = None

    def __call__(self, ev) -> None:
        self.events.append(ev)
        if isinstance(ev, ApprovalRequested) and self.decide is not None:
            answer = self.decide(ev.request)
            if answer is not None:
                approved, note = answer
                self.channel.resolve(ev.request.id, approved, note=note, actor="tester")


def _channel(decide=None, timeout=5.0, audit=None) -> tuple[ApprovalChannel, Recorder]:
    rec = Recorder(decide)
    ch = ApprovalChannel(emit=rec, timeout=timeout, conversation_id="c1", audit=audit)
    rec.channel = ch
    return ch, rec


def _request(ch: ApprovalChannel, call_id="t1", tool="write_file", args=None):
    return ch.new_request(
        tool_call_id=call_id, tool_name=tool, args=args or {"path": "x"}
    )


# --- the channel ---------------------------------------------------------------------------
def test_approve_and_events_in_order(spans):
    ch, rec = _channel(decide=lambda r: (True, ""))
    req = _request(ch)
    decisions = _run(ch.ask([req]))
    assert decisions[req.id].approved and decisions[req.id].outcome == "approved"
    assert [type(e) for e in rec.events] == [ApprovalRequested, ApprovalResolved]
    resolved = rec.events[1]
    assert resolved.outcome == "approved" and resolved.actor == "tester"
    assert resolved.tool_call_id == "t1"
    names = [s.name for s in spans.get_finished_spans()]
    assert names == [REQUEST_SPAN, DECISION_SPAN]


def test_deny_carries_the_note():
    ch, _ = _channel(decide=lambda r: (False, "not that file"))
    req = _request(ch)
    d = _run(ch.ask([req]))[req.id]
    assert not d.approved and d.outcome == "denied" and d.note == "not that file"


def test_nobody_answers_so_it_times_out():
    ch, rec = _channel(timeout=0.05)
    req = _request(ch)
    d = _run(ch.ask([req]))[req.id]
    assert d.outcome == "timeout" and not d.approved
    assert rec.events[-1].outcome == "timeout"
    assert ch.pending() == []


def test_first_decision_wins_and_unknown_ids_are_refused():
    ch, _ = _channel()

    async def scenario():
        req = _request(ch)
        task = asyncio.create_task(ch.ask([req]))
        await asyncio.sleep(0)
        assert [r.id for r in ch.pending()] == [req.id]
        assert ch.resolve(req.id, True, actor="cli")
        assert not ch.resolve(req.id, False, actor="web")  # too late
        assert not ch.resolve("nope", True)
        return (await task)[req.id]

    d = _run(scenario())
    assert d.approved and d.actor == "cli"


def test_cancel_all_denies_what_is_pending():
    ch, _ = _channel()

    async def scenario():
        req = _request(ch)
        task = asyncio.create_task(ch.ask([req]))
        await asyncio.sleep(0)
        ch.cancel_all("turn stopped")
        return (await task)[req.id]

    d = _run(scenario())
    assert d.outcome == "cancelled" and d.note == "turn stopped"


def test_an_aborted_wait_settles_everything_and_reraises():
    ch, rec = _channel()

    async def scenario():
        task = asyncio.create_task(ch.ask([_request(ch)]))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    _run(scenario())
    assert isinstance(rec.events[-1], ApprovalResolved)
    assert rec.events[-1].outcome == "cancelled" and ch.pending() == []


def test_reason_comes_from_the_guards_annotation():
    ch, _ = _channel()
    ch.annotate("t1", reason="because policy", preview={"deletes": 3})
    req = _request(ch)
    assert req.reason == "because policy" and req.preview == {"deletes": 3}
    assert _request(ch, call_id="t2").reason.startswith("a guard requires approval")


def test_args_are_shown_with_secrets_scrubbed(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-abcdef0123456789")
    ch, _ = _channel()
    req = _request(ch, args={"header": "Bearer sk-or-v1-abcdef0123456789"})
    assert "sk-or-v1" not in req.args and "$SECRET:OPENROUTER_API_KEY" in req.args
    assert json.loads(req.args)["header"].startswith("Bearer ")
    assert _request(ch, args='{"a": 1}').args == '{"a": 1}'


def test_audit_trail_records_request_and_decision(tmp_path):
    audit = ApprovalAudit(tmp_path / "agent.jsonl")
    ch, _ = _channel(decide=lambda r: (False, "no"), audit=audit)
    _run(ch.ask([_request(ch)]))
    records = [
        json.loads(line) for line in (tmp_path / "agent.jsonl").read_text().splitlines()
    ]
    assert [r["event"] for r in records] == ["request", "decision"]
    assert records[1]["decision"]["outcome"] == "denied"
    assert records[1]["decision"]["note"] == "no"
    assert records[0]["tool_name"] == "write_file" and "latency_ms" in records[1]


def test_audit_write_failure_never_breaks_the_turn(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    audit = ApprovalAudit(blocker / "nested" / "a.jsonl")  # parent is a file
    ch, _ = _channel(decide=lambda r: (True, ""), audit=audit)
    assert _run(ch.ask([_request(ch)]))


def test_binding_is_scoped():
    ch, _ = _channel()
    assert not approvals_available()
    with bind_channel(ch):
        assert approvals_available() and current_channel() is ch
    assert current_channel() is None


# --- the broker -------------------------------------------------------------------------------
class _Calls:
    def __init__(self, *calls):
        self.approvals = [
            ToolCallPart(name, args, tool_call_id=cid) for cid, name, args in calls
        ]


def test_broker_denies_everything_when_nobody_is_attached():
    results = _run(
        ApprovalBroker().handle_deferred_tool_calls(
            None, requests=_Calls(("a", "write_file", {}), ("b", "run_command", {}))
        )
    )
    assert set(results.approvals) == {"a", "b"}
    for denial in results.approvals.values():
        assert isinstance(denial, ToolDenied) and "no one is attached" in denial.message


@pytest.mark.parametrize(
    ("decide", "timeout", "expect"),
    [
        (lambda r: (True, ""), 5.0, True),
        (lambda r: (False, "use trash instead"), 5.0, "use trash instead"),
        (None, 0.05, "No one answered"),
    ],
)
def test_broker_maps_decisions(decide, timeout, expect):
    ch, _ = _channel(decide=decide, timeout=timeout)

    async def scenario():
        with bind_channel(ch):
            return await ApprovalBroker().handle_deferred_tool_calls(
                None, requests=_Calls(("a", "write_file", {"path": "x"}))
            )

    result = _run(scenario()).approvals["a"]
    if expect is True:
        assert result is True
    else:
        assert isinstance(result, ToolDenied) and expect in result.message


# --- real guards through a real agent run -------------------------------------------------------
def _scripted(*steps):
    """Each step is a list of (tool, args) called in ONE model response."""
    queue = [list(s) for s in steps]

    def respond(messages, info):
        if queue:
            return ModelResponse(parts=[ToolCallPart(t, a) for t, a in queue.pop(0)])
        return ModelResponse(parts=[TextPart("done")])

    return FunctionModel(respond)


def _gating():
    return ToolGuardrail(
        guard=make_gating_guard(
            parse_policy(
                {
                    "rules": [
                        {"tools": ["write_file"], "action": "allow", "inside": ["out"]},
                        {"tools": ["write_file"], "action": "approve"},
                    ]
                }
            ),
            approval_available=approvals_available,
        )
    )


def _agent(model, extra_caps=()):
    executed: list[str] = []

    def write_file(path: str, content: str) -> str:
        executed.append(path)
        return f"wrote {path}"

    def read_file(path: str) -> str:
        return "EVIL instructions" if path == "evil" else "fine"

    agent = Agent(
        model,
        capabilities=[_gating(), *extra_caps, ApprovalBroker()],
        tools=[write_file, read_file],
    )
    return agent, executed


def _returns(result, tool="write_file"):
    return [
        str(p.content)
        for m in result.all_messages()
        for p in m.parts
        if p.part_kind == "tool-return" and p.tool_name == tool
    ]


def _run_agent(agent, channel):
    async def go():
        with bind_channel(channel):
            return await agent.run("go")

    return _run(go())


def test_gating_approve_asks_the_person_and_runs_when_approved():
    ch, rec = _channel(decide=lambda r: (True, ""))
    agent, executed = _agent(
        _scripted([("write_file", {"path": "notes.txt", "content": "x"})])
    )
    result = _run_agent(agent, ch)
    assert executed == ["notes.txt"] and _returns(result) == ["wrote notes.txt"]
    (req_ev,) = [e for e in rec.events if isinstance(e, ApprovalRequested)]
    assert "gating policy requires approval for 'write_file'" in req_ev.request.reason
    assert '"notes.txt"' in req_ev.request.args


def test_gating_approve_denied_is_not_run_and_the_model_hears_why():
    ch, _ = _channel(decide=lambda r: (False, "write it under out/ instead"))
    agent, executed = _agent(
        _scripted([("write_file", {"path": "notes.txt", "content": "x"})])
    )
    result = _run_agent(agent, ch)
    assert executed == []
    (msg,) = _returns(result)
    assert "denied" in msg and "write it under out/ instead" in msg


def test_allowed_calls_never_ask():
    ch, rec = _channel(decide=lambda r: (False, ""))
    agent, executed = _agent(
        _scripted([("write_file", {"path": "out/a", "content": "x"})])
    )
    _run_agent(agent, ch)
    assert executed == ["out/a"] and rec.events == []


def test_nobody_attached_keeps_the_old_refusal():
    agent, executed = _agent(
        _scripted([("write_file", {"path": "notes.txt", "content": "x"})])
    )
    result = _run_agent(agent, None)
    assert executed == []
    assert "approval is not available in this environment" in _returns(result)[0]


class _EvilScorer:
    name = "fake"
    threshold = 0.5

    def warmup(self) -> None:
        pass

    async def score(self, value, tool_name):
        hit = "EVIL" in str(value)
        return Score(
            hit=hit,
            noteworthy=hit,
            risk_level="high",
            score=0.9 if hit else 0.1,
            diagnostics={},
        )

    async def localize(self, text, tool_name):
        return None


def test_injection_tightening_asks_with_the_flag_as_the_reason():
    ch, rec = _channel(decide=lambda r: (True, ""))
    guard = InjectionGuard(
        policy=parse_injection_policy(
            {"default_action": "flag", "on_flag": {"require_approval": ["write_file"]}}
        ),
        scorer=_EvilScorer(),
    )
    agent, executed = _agent(
        _scripted(
            [("read_file", {"path": "evil"})],
            [("write_file", {"path": "out/a", "content": "x"})],
        ),
        extra_caps=[guard],
    )
    _run_agent(agent, ch)
    assert executed == ["out/a"]  # approved
    (req_ev,) = [e for e in rec.events if isinstance(e, ApprovalRequested)]
    assert "flagged as a possible prompt injection" in req_ev.request.reason
    assert "read_file" in req_ev.request.reason


def test_a_batch_is_asked_together_and_decided_per_call():
    seen = []

    def decide(req):
        seen.append(req.tool_call_id)
        return None  # answer once both are pending

    ch, rec = _channel(decide=decide)
    agent, executed = _agent(
        _scripted(
            [
                ("write_file", {"path": "a.txt", "content": "x"}),
                ("write_file", {"path": "b.txt", "content": "y"}),
            ]
        )
    )

    async def go():
        with bind_channel(ch):
            task = asyncio.create_task(agent.run("go"))
            while len(ch.pending()) < 2:
                await asyncio.sleep(0.01)
            first, second = ch.pending()
            ch.resolve(first.id, True)
            ch.resolve(second.id, False, note="only one")
            return await task

    _run(go())
    assert len(seen) == 2 and len(executed) == 1


def test_the_run_never_ends_with_an_unhandled_deferred_request():
    """With the broker installed, an approval request never surfaces as a
    DeferredToolRequests output the runtime does not expect."""
    agent, _ = _agent(_scripted([("write_file", {"path": "x.txt", "content": "x"})]))
    ch, _ = _channel(timeout=0.05)
    result = _run_agent(agent, ch)
    assert not isinstance(result.output, DeferredToolRequests)
    assert result.output == "done"


# --- Session / TurnHandle -----------------------------------------------------------------------
def _session(tmp_path, agent: Agent | None = None):
    s = object.__new__(runtime.Session)
    s._turn = None
    s._token = None
    s.conversation_id = "conv-1"
    s._approval_timeout = 5.0
    s._approval_audit = ApprovalAudit(tmp_path / "approvals.jsonl")
    if agent is not None:

        async def stream(prompt, *, conversation_id=None, _token=None):
            result = await agent.run(prompt)
            yield StreamEvent(kind="text", text=str(result.output))

        s.stream = stream
    return s


def test_only_interactive_turns_get_a_channel(tmp_path):
    s = _session(tmp_path)
    seen = []

    async def stream(prompt, *, conversation_id=None, _token=None):
        seen.append(current_channel())
        yield StreamEvent(kind="text", text="x")

    s.stream = stream

    async def go():
        h = s.start_turn("hi")
        await h._task
        h2 = s.start_turn("hi", interactive=True)
        await h2._task

    _run(go())
    assert seen[0] is None and isinstance(seen[1], ApprovalChannel)


def test_session_turn_publishes_the_request_and_resolve_approval_settles_it(tmp_path):
    agent, executed = _agent(
        _scripted([("write_file", {"path": "notes.txt", "content": "x"})])
    )
    s = _session(tmp_path, agent)

    async def go():
        handle = s.start_turn("go", interactive=True)
        events = []
        async for ev in handle.subscribe():
            events.append(ev)
            if isinstance(ev, ApprovalRequested):
                assert [r.id for r in s.pending_approvals()] == [ev.request.id]
                assert s.has_pending_approvals
                assert s.resolve_approval(ev.request.id, True, actor="cli")
        return events

    events = _run(go())
    kinds = [type(e).__name__ for e in events]
    assert kinds[:2] == ["ApprovalRequested", "ApprovalResolved"]
    assert executed == ["notes.txt"]
    assert not s.has_pending_approvals
    lines = (tmp_path / "approvals.jsonl").read_text().splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["request", "decision"]


def test_stopping_the_turn_while_it_waits_denies_the_request(tmp_path):
    agent, executed = _agent(
        _scripted([("write_file", {"path": "notes.txt", "content": "x"})])
    )
    s = _session(tmp_path, agent)
    s._token = None

    async def go():
        handle = s.start_turn("go", interactive=True)
        outcomes = []
        async for ev in handle.subscribe():
            if isinstance(ev, ApprovalRequested):
                await s.cancel()
            if isinstance(ev, ApprovalResolved):
                outcomes.append(ev.outcome)
        return outcomes

    assert _run(go()) == ["cancelled"]
    assert executed == []


def test_a_waiting_turn_is_busy_for_the_sse_idle_reaper(tmp_path):
    from ui.sse import _session_is_busy

    agent, _ = _agent(
        _scripted([("write_file", {"path": "notes.txt", "content": "x"})])
    )
    s = _session(tmp_path, agent)

    async def go():
        handle = s.start_turn("go", interactive=True)
        busy = None
        async for ev in handle.subscribe():
            if isinstance(ev, ApprovalRequested):
                handle._subscribers.clear()  # even with no subscriber attached...
                busy = _session_is_busy(s)
                s.resolve_approval(ev.request.id, False)
                break
        await handle._task
        return busy

    assert _run(go()) is True


def test_resolve_with_no_turn_or_no_channel_is_a_noop(tmp_path):
    s = _session(tmp_path)
    assert s.pending_approvals() == [] and not s.resolve_approval("x", True)


# --- spec ------------------------------------------------------------------------------------------
def test_spec_timeout(tmp_path):
    from paf.specs import spec_approval_timeout

    (tmp_path / "a.yaml").write_text("name: a\napprovals:\n  timeout: 30\n")
    (tmp_path / "b.yaml").write_text("name: b\n")
    (tmp_path / "c.yaml").write_text("name: c\napprovals:\n  timeout: -5\n")
    assert spec_approval_timeout("a", tmp_path) == 30.0
    assert spec_approval_timeout("b", tmp_path) == 600.0
    assert spec_approval_timeout("c", tmp_path) == 600.0
    assert spec_approval_timeout("missing", tmp_path) == 600.0
