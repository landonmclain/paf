"""Approval flow P2: the CLI asks the person at the terminal.

Covers the answer parser, the request panel and outcome lines, the printer
marking a call that was not approved as "not run" (instead of a misleading ✓),
the prompt flow (approve, deny with a note, Ctrl-C, settled elsewhere), that
only a terminal session makes turns interactive, and the real turn loop end to
end against a real agent and gating rule.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness import ToolGuardrail

import paf.cli.chat as chat_mod
from paf import runtime
from paf.approvals import (
    ApprovalAudit,
    ApprovalBroker,
    ApprovalRequest,
    ApprovalResolved,
    approvals_available,
)
from paf.capabilities.guardrails.gating_policy import make_gating_guard, parse_policy
from paf.cli._render import (
    TurnPrinter,
    parse_approval_answer,
    render_approval_request,
    render_approval_resolved,
)
from paf.runtime import StreamEvent


def _run(coro):
    return asyncio.run(coro)


def _req(**kw) -> ApprovalRequest:
    base = dict(
        id="ap1",
        tool_call_id="t1",
        tool_name="run_command",
        args=json.dumps({"command": "rm -rf build/"}),
        reason="this agent's gating policy requires approval for 'run_command'",
        preview=None,
        created_at=1000.0,
        expires_at=1600.0,
    )
    base.update(kw)
    return ApprovalRequest(**base)


# --- answers -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("y", (True, "")),
        ("YES", (True, "")),
        ("  y  ", (True, "")),
        ("n", (False, "")),
        ("n use the trash instead", (False, "use the trash instead")),
        ("no, keep the logs", (False, "no, keep the logs")),
        ("", (False, "")),
        (None, (False, "")),
        ("only delete dist/", (False, "only delete dist/")),
        ("yolo", (False, "yolo")),  # only y/yes approves
    ],
)
def test_parse_answer(text, expected):
    assert parse_approval_answer(text) == expected


# --- rendering ---------------------------------------------------------------------------
def test_request_panel_shows_what_why_and_how_long():
    panel = render_approval_request(
        _req(preview={"deletes": "2 directories, 340 MB"}), now=1000.0
    )
    assert "approval needed" in panel and "run_command" in panel
    assert "command: rm -rf build/" in panel
    assert "why: this agent's gating policy" in panel
    assert "deletes: 2 directories, 340 MB" in panel
    assert "expires in 10 min" in panel
    assert "y approve" in panel


@pytest.mark.parametrize(
    ("outcome", "text"),
    [
        ("approved", "approved"),
        ("denied", "denied"),
        ("timeout", "no answer in time"),
        ("cancelled", "cancelled"),
    ],
)
def test_resolved_lines(outcome, text):
    line = render_approval_resolved(
        ApprovalResolved("ap1", "t1", "run_command", outcome, actor="cli", note="nope")
    )
    assert text in line and "run_command" in line


def test_a_web_decision_names_who_made_it():
    line = render_approval_resolved(
        ApprovalResolved("ap1", "t1", "run_command", "approved", actor="web")
    )
    assert "by web" in line


def test_printer_marks_a_denied_call_as_not_run(capsys):
    p = TurnPrinter(show_thinking=False, markdown=False)
    p.feed(
        StreamEvent(
            kind="tool_call", tool_name="run_command", args="rm -rf build", step_id="t1"
        )
    )
    p.pause()
    p.feed(ApprovalResolved("ap1", "t1", "run_command", "denied", actor="cli"))
    p.feed(
        StreamEvent(
            kind="tool_result", tool_name="run_command", text="denied", step_id="t1"
        )
    )
    p.finish()
    out = capsys.readouterr().out
    assert "denied, not run" in out and "✓" not in out


def test_printer_keeps_the_tick_for_an_approved_call(capsys):
    p = TurnPrinter(show_thinking=False, markdown=False)
    p.feed(
        StreamEvent(
            kind="tool_call", tool_name="run_command", args="rm -rf build", step_id="t1"
        )
    )
    p.pause()
    p.feed(ApprovalResolved("ap1", "t1", "run_command", "approved", actor="cli"))
    p.feed(
        StreamEvent(
            kind="tool_result", tool_name="run_command", text="ok", step_id="t1"
        )
    )
    p.feed(StreamEvent(kind="text", text="done"))
    p.finish()
    out = capsys.readouterr().out
    assert "approved" in out and "not run" not in out


# --- the prompt --------------------------------------------------------------------------
class FakeApp:
    def __init__(self) -> None:
        self.is_running = True
        self.done: asyncio.Future | None = None

    def exit(self, result=None) -> None:
        if self.done is not None and not self.done.done():
            self.done.set_result(result)


class FakePrompt:
    """Answers immediately with `answer`, raises `raises`, or waits for app.exit."""

    def __init__(self, answer=None, raises=None) -> None:
        self.answer, self.raises = answer, raises
        self.app = FakeApp()

    async def prompt_async(self, *a, **k):
        if self.raises is not None:
            raise self.raises
        if self.answer is not None:
            return self.answer
        self.app.done = asyncio.get_running_loop().create_future()
        return await self.app.done


class FakeSession:
    def __init__(self, pending=True) -> None:
        self.pending = [_req()] if pending else []
        self.resolved: list[tuple] = []

    def pending_approvals(self):
        return list(self.pending)

    def resolve_approval(self, approval_id, approved, *, note="", actor="user"):
        self.resolved.append((approval_id, approved, note, actor))
        self.pending = []
        return True


def _app(session, prompt) -> chat_mod.ChatApp:
    app = object.__new__(chat_mod.ChatApp)
    app.session = session
    app.approve_ps = prompt
    return app


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (FakePrompt(answer="y"), ("ap1", True, "", "cli")),
        (FakePrompt(answer="n keep build/"), ("ap1", False, "keep build/", "cli")),
        (FakePrompt(answer=""), ("ap1", False, "", "cli")),
        (FakePrompt(raises=KeyboardInterrupt()), ("ap1", False, "", "cli")),
        (FakePrompt(raises=EOFError()), ("ap1", False, "", "cli")),
    ],
)
def test_prompt_answers_settle_the_request(prompt, expected, capsys):
    session = FakeSession()
    _run(_app(session, prompt)._approve(_req()))
    assert session.resolved == [expected]
    assert "approval needed" in capsys.readouterr().out


def test_prompt_closes_itself_when_settled_elsewhere():
    session = FakeSession()
    prompt = FakePrompt()  # waits: the user never types

    async def scenario():
        task = asyncio.create_task(_app(session, prompt)._approve(_req()))
        await asyncio.sleep(0.05)
        session.pending = []  # e.g. it timed out, or the web UI answered
        await asyncio.wait_for(task, timeout=2)

    _run(scenario())
    assert session.resolved == []  # the CLI did not also decide it


# --- interactive only at a terminal ------------------------------------------------------------
@pytest.mark.parametrize(
    ("has_terminal", "interactive"), [(True, True), (False, False)]
)
def test_turns_are_interactive_only_with_a_terminal(has_terminal, interactive):
    started = {}

    class Session:
        current_turn = None

        def start_turn(self, prompt, *, conversation_id=None, interactive=False):
            started["interactive"] = interactive
            return "handle"

    app = object.__new__(chat_mod.ChatApp)
    app.session = Session()
    app.ps = object() if has_terminal else None
    app.cid = "c1"
    app.queued = []

    async def stream(handle):
        pass

    app.stream = stream
    _run(app.send("hello"))
    assert started["interactive"] is interactive


# --- the real turn loop ---------------------------------------------------------------------
class _NoKeys:
    def __init__(self, inbox) -> None:
        pass

    def resume(self) -> None:
        pass

    def pause(self) -> None:
        pass


def _session_with_agent(tmp_path, executed):
    def run_command(command: str) -> str:
        executed.append(command)
        return "ok"

    calls = [ToolCallPart("run_command", {"command": "rm -rf build/"})]

    def respond(messages, info):
        return (
            ModelResponse(parts=[calls.pop(0)])
            if calls
            else ModelResponse(parts=[TextPart("done")])
        )

    gating = ToolGuardrail(
        guard=make_gating_guard(
            parse_policy({"rules": [{"tools": ["run_command"], "action": "approve"}]}),
            approval_available=approvals_available,
        )
    )
    agent = Agent(
        FunctionModel(respond),
        capabilities=[gating, ApprovalBroker()],
        tools=[run_command],
    )
    s = object.__new__(runtime.Session)
    s._turn, s._token = None, None
    s.conversation_id = "c1"
    s._approval_timeout = 5.0
    s._approval_audit = ApprovalAudit(tmp_path / "a.jsonl")

    async def stream(prompt, *, conversation_id=None, _token=None):
        result = await agent.run(prompt)
        yield StreamEvent(kind="text", text=str(result.output))

    s.stream = stream
    return s


@pytest.mark.parametrize(("answer", "ran"), [("y", True), ("n not now", False)])
def test_cli_turn_loop_end_to_end(tmp_path, monkeypatch, capsys, answer, ran):
    monkeypatch.setattr(chat_mod, "_KeyWatch", _NoKeys)
    executed: list[str] = []
    session = _session_with_agent(tmp_path, executed)
    app = object.__new__(chat_mod.ChatApp)
    app.session = session
    app.ps = object()
    app.approve_ps = FakePrompt(answer=answer)
    app.show_thinking, app.markdown = False, False
    app._turn, app._sigints = None, 0

    async def scenario():
        await app.stream(session.start_turn("clean up", interactive=True))

    _run(scenario())
    out = capsys.readouterr().out
    assert "approval needed" in out and "rm -rf build/" in out
    assert executed == (["rm -rf build/"] if ran else [])
    assert ("approved" in out) is ran
    assert ("denied" in out) is (not ran)
