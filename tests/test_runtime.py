"""Offline unit tests for ``paf.runtime`` — the production Session event contract.

These run WITHOUT a model or API key. They exercise the pure, deterministic
seam of runtime: :func:`_dispatch_stream_event` (the Phase 1+2 turn-event
mapping, fed SYNTHETIC pydantic-ai events) and the background-task registry
(``subscribe_tasks`` replay + ``_publish_task`` transitions). Live model turns
belong in the ``live``-marked tests under ``tests/`` with ``-m live``.

Verified contract (Phase 2):
- ``FunctionToolCallEvent``  -> tool_call StreamEvent + ToolRunning (same step_id)
- ``FunctionToolResultEvent`` success -> tool_result (no ToolError)
- ``FunctionToolResultEvent`` failed/retried -> tool_result + ToolError (same step_id)
- PartStart/PartDelta -> text/thinking StreamEvents (Phase 1 parity)
- BackgroundTask registry replays active (queued/running) tasks on subscribe.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest
from pydantic_ai import (
    CancellationToken,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
)
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from paf import runtime


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _call_event(
    tool: str = "run_command", sid: str = "call_123"
) -> FunctionToolCallEvent:
    part = ToolCallPart(
        tool_name=tool,
        args='{"command": "pwd"}',
        tool_call_id=sid,
    )
    return FunctionToolCallEvent(part=part, args_valid=True)


def _result_event(
    *,
    tool: str = "run_command",
    sid: str = "call_123",
    outcome: str = "success",
    content: str = "ok",
) -> FunctionToolResultEvent:
    part = ToolReturnPart(
        tool_name=tool,
        content=content,
        tool_call_id=sid,
        outcome=outcome,
        timestamp=_now(),
    )
    return FunctionToolResultEvent(part=part, content=content)


def _retry_event(
    tool: str = "run_command", sid: str = "call_999", content: str = "model_retry"
) -> FunctionToolResultEvent:
    part = RetryPromptPart(
        tool_name=tool, tool_call_id=sid, content=content, timestamp=_now()
    )
    return FunctionToolResultEvent(part=part, content=content)


# --- _dispatch_stream_event: tool lifecycle -------------------------------


def test_tool_call_emits_tool_running_with_matching_step_id() -> None:
    out = runtime._dispatch_stream_event(_call_event(sid="call_step_1"))
    kinds = [type(e).__name__ for e in out]
    assert kinds == ["StreamEvent", "ToolRunning"]
    assert out[0].kind == "tool_call"
    assert out[0].tool_name == "run_command"
    running = out[1]
    assert isinstance(running, runtime.ToolRunning)
    assert running.step_id == "call_step_1"  # stable id -> UI flips ONE row
    assert running.tool_name == "run_command"


def test_successful_tool_result_has_no_tool_error() -> None:
    out = runtime._dispatch_stream_event(
        _result_event(sid="call_step_1", outcome="success", content="done")
    )
    assert [type(e).__name__ for e in out] == ["StreamEvent"]
    assert out[0].kind == "tool_result"
    assert out[0].text == "done"


def test_failed_tool_result_emits_tool_error_same_step_id() -> None:
    out = runtime._dispatch_stream_event(
        _result_event(sid="call_fail_7", outcome="failed", content="boom")
    )
    res, err = out
    assert res.kind == "tool_result"
    assert isinstance(err, runtime.ToolError)
    assert err.step_id == "call_fail_7"  # correlates with ToolRunning
    assert "boom" in err.error


def test_retry_prompt_result_emits_tool_error() -> None:
    out = runtime._dispatch_stream_event(_retry_event(sid="call_retry"))
    assert isinstance(out[-1], runtime.ToolError)
    assert out[-1].step_id == "call_retry"


# --- _dispatch_stream_event: Phase 1 text/thinking parity ------------------


def test_part_start_text_leading() -> None:
    from pydantic_ai import PartStartEvent

    ev = PartStartEvent(part=TextPart(content="hello"), index=0)
    out = runtime._dispatch_stream_event(ev)
    assert [e.kind for e in out] == ["text"]
    assert out[0].text == "hello"


def test_part_delta_text_and_thinking() -> None:
    from pydantic_ai import PartDeltaEvent

    out = runtime._dispatch_stream_event(
        PartDeltaEvent(delta=TextPartDelta(content_delta="abc"), index=0)
    )
    assert [e.kind for e in out] == ["text"]
    assert out[0].text == "abc"

    out2 = runtime._dispatch_stream_event(
        PartDeltaEvent(delta=ThinkingPartDelta(content_delta="hmm"), index=0)
    )
    assert [e.kind for e in out2] == ["thinking"]
    assert out2[0].text == "hmm"


# --- background-task registry (no agent build) ----------------------------
# Imported here (not at module top) so tests above never trigger a paf.background
# / session-build import at module scope.
from paf.background import BackgroundCoordinator  # noqa: E402
from paf.background_processes._events import get_agent_emitter  # noqa: E402


def _bare_coordinator() -> BackgroundCoordinator:
    """A coordinator with no store built (pure registry tests)."""
    c = object.__new__(BackgroundCoordinator)
    c._background_tasks = {}
    c._consolidation_seq = 0
    c._run_events = None
    c._emitter = get_agent_emitter("assistant")
    c.lane = "agent"
    c.conversation_id = None
    return c


def test_subscribe_replays_active_task() -> None:
    c = _bare_coordinator()
    seen: list[tuple[str, str]] = []

    def cb(ev) -> None:
        seen.append((ev.label, ev.status.value))

    c.subscribe_tasks(cb)
    t = runtime.BackgroundTask(id="consolidation.c.1", label="Consolidating memory (c)")
    asyncio.run(c._publish_task(t))  # queued (default)

    # Late subscriber must see the in-flight (queued) task replayed.
    seen2: list[tuple[str, str]] = []
    c.subscribe_tasks(lambda ev: seen2.append((ev.label, ev.status.value)))
    assert ("Consolidating memory (c)", "queued") in seen2

    # Transition to done reaches both subscribers.
    t.status = runtime.DONE
    asyncio.run(c._publish_task(t))
    assert ("Consolidating memory (c)", "done") in seen
    assert ("Consolidating memory (c)", "done") in seen2


def test_done_task_not_replayed() -> None:
    c = _bare_coordinator()
    c.subscribe_tasks(lambda ev: None)
    t = runtime.BackgroundTask(id="consolidation.d.1", label="L")
    t.status = runtime.DONE
    asyncio.run(c._publish_task(t))
    seen: list = []
    c.subscribe_tasks(lambda ev: seen.append(ev))
    assert seen == []  # only queued/running are active; done is not replayed


# --- profile namespace (conversation identity scoping) -------------------


def test_scoped_cid_prefixes_with_profile_name() -> None:
    s = object.__new__(runtime.Session)
    s.profile = _FakeProfile(name="assistant")
    assert s._scoped_cid("cli-1") == "assistant:cli-1"


def test_scoped_cid_returns_cid_when_no_profile() -> None:
    s = object.__new__(runtime.Session)
    s.profile = None
    assert s._scoped_cid("cli-1") == "cli-1"
    assert s._scoped_cid(None) is None


def test_scoped_cid_handles_bare_name_profile() -> None:
    s = object.__new__(runtime.Session)
    s.profile = "test"  # raw string profile fallback
    assert s._scoped_cid("web-1") == "test:web-1"


def test_two_profiles_same_cid_do_not_collide() -> None:
    """The namespace invariant: identical plain cids scope apart by profile."""
    a = _FakeProfile("assistant")
    b = _FakeProfile("demo")
    sa, sb = object.__new__(runtime.Session), object.__new__(runtime.Session)
    sa.profile, sb.profile = a, b
    assert sa._scoped_cid("conv-9") != sb._scoped_cid("conv-9")


# --- profile wiring --------------------------------------------------------


def test_session_init_injects_spec_profile_for_keyword_agent_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: Session(agent_name=...) must resolve+inject the spec's
    declared profile when none is passed.

    This guards the bug where the constructor read ``agent_name`` from
    positional args only, so a keyword ``agent_name="test"`` silently fell back
    to the default profile (building a default-profile agent under a test-agent
    name instead of the lean Shell-only toolset).
    """
    monkeypatch.setattr(runtime, "resolve_profile_name", lambda name: "test")
    monkeypatch.setattr(runtime, "PROFILES", {"test": _FAKE_SHELL_PROFILE})
    # Stub the heavy build paths so constructing Session needs no API key.
    monkeypatch.setattr(
        runtime, "build_agent", lambda name, profile=None, on_complete=None: object()
    )
    monkeypatch.setattr(runtime, "_build_store", lambda: object())
    monkeypatch.setattr(runtime, "setup_observability", lambda: None)

    s = runtime.Session(agent_name="test", trace=False)
    # Session.__init__ must have injected the Shell-only profile for the
    # keyword agent_name (not the default profile).
    assert s.profile == _FAKE_SHELL_PROFILE


class _FakeRun:
    """Minimal AgentRun stand-in for steering tests."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str]] = []

    def enqueue(self, content: str, priority: str = "asap") -> str:
        self.enqueued.append((content, priority))
        return "enq-1"


def test_enqueue_forwards_to_live_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 3: enqueue() must forward to the active AgentRun's enqueue."""
    monkeypatch.setattr(runtime.Session, "__init__", lambda self, *a, **k: None)

    s = runtime.Session(_FakeRun.__module__ + ".x", agent_name="x", trace=False)
    fake = _FakeRun()
    s._run = fake  # type: ignore[attr-defined]

    enq_id = asyncio.run(s.enqueue("steer me", priority="asap"))
    assert enq_id == "enq-1"
    assert fake.enqueued == [("steer me", "asap")]


def test_enqueue_returns_none_when_no_run_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Steering with nothing in flight must no-op (return None), not raise."""
    monkeypatch.setattr(runtime.Session, "__init__", lambda self, *a, **k: None)

    s = runtime.Session("x", agent_name="x", trace=False)
    s._run = None  # type: ignore[attr-defined]

    assert asyncio.run(s.enqueue("steer me")) is None


class _FakeProfile:
    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeProfile) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


class _FakeToolStore:
    """Duck-typed store exposing list_unresolved_tool_effects for offline tests."""

    def __init__(self, unresolved: set[str]) -> None:
        self._unresolved = unresolved

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[object]:
        del run_id
        return [type("Rec", (), {"tool_call_id": tid})() for tid in self._unresolved]


def _trim_interrupted(unresolved: set[str], history: list[object]) -> list[object]:
    """Run Session._recover_interrupted against a fake store, synchronously."""

    async def _run() -> list[object]:
        s = object.__new__(runtime.Session)
        s._store = _FakeToolStore(unresolved)  # type: ignore[attr-defined]
        return await s._recover_interrupted("run-1", history)  # type: ignore[attr-defined]

    return asyncio.run(_run())


def test_recover_interrupted_trims_unresolved_tool_tail() -> None:
    """A recovered interrupted snapshot must drop the unresolved tool tail."""
    user_ok = ModelRequest(
        parts=[UserPromptPart(content="hello", part_kind="user-prompt")]
    )
    resp_ok = ModelResponse(parts=[TextPart(content="partial done")])
    resp_bad = ModelResponse(
        parts=[ToolCallPart(tool_name="shell", tool_call_id="call_unresolved")]
    )
    # History ends on the unresolved tool call -> that tail must be trimmed.
    trimmed = _trim_interrupted({"call_unresolved"}, [user_ok, resp_ok, resp_bad])
    assert len(trimmed) == 2
    assert trimmed[0] is user_ok
    assert trimmed[1] is resp_ok


def test_recover_interrupted_keeps_history_when_no_unresolved() -> None:
    """No unresolved tool effects -> recovered history returned intact."""
    user_ok = ModelRequest(
        parts=[UserPromptPart(content="hi", part_kind="user-prompt")]
    )
    resp = ModelResponse(
        parts=[ToolCallPart(tool_name="shell", tool_call_id="call_resolved")]
    )
    trimmed = _trim_interrupted({"call_other"}, [user_ok, resp])
    assert len(trimmed) == 2
    assert trimmed[0] is user_ok
    assert trimmed[1] is resp


async def _collect(agen):
    """Drain an async generator, propagating exceptions, into a list."""
    out: list[object] = []
    async for x in agen:
        out.append(x)
    return out


def test_stream_preserves_partial_history_on_stop(monkeypatch) -> None:
    """Hitting Stop must keep the cancelled run's partial history, end cleanly.

    ``stream()`` catches ``RunCancelled`` (raised when a CancellationToken
    cancel ends the agent run), appends the run's partial ``new_messages()`` to
    ``self.history`` so the NEXT message in this session continues from it, and
    ends the stream WITHOUT surfacing an error frame ("The agent run was
    cancelled."). External cancellation (``asyncio.CancelledError``) is NOT
    swallowed — only first-party ``RunCancelled``.
    """
    s = object.__new__(runtime.Session)  # type: ignore[attr-defined]
    s.agent_name = "x"
    s.profile = None
    s.conversation_id = "conv-1"
    s._token = CancellationToken()
    s.history = []

    @contextmanager
    def _root_span(*a, **k):
        del a, k
        yield type("Span", (), {"set_attribute": lambda *x, **y: None})()

    monkeypatch.setattr(runtime, "root_span", _root_span)

    async def _noop(*a, **k):
        del a, k

    s._ensure_resumed = _noop  # type: ignore[attr-defined]

    partial = ModelResponse(parts=[TextPart(content="partial answer")])

    class _CancellingRun:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RunCancelled("cancelled", messages=[partial])

    class _CM:
        async def __aenter__(self):
            return _CancellingRun()

        async def __aexit__(self, *exc):
            del exc
            return False

    class _FakeAgent:
        def iter(self, *a, **k):
            del a, k
            return _CM()

    s._agent = _FakeAgent()  # type: ignore[attr-defined]

    events = asyncio.run(
        _collect(s.stream("hi", conversation_id="conv-1"))  # type: ignore[attr-defined]
    )
    # Stream ends cleanly (no error/exception event) and partial history kept.
    assert events == []
    assert s.history == [partial]


def test_stream_propagates_real_cancellation(monkeypatch) -> None:
    """External ``asyncio.CancelledError`` must NOT be swallowed."""

    s = object.__new__(runtime.Session)  # type: ignore[attr-defined]
    s.agent_name = "x"
    s.profile = None
    s.conversation_id = "conv-1"
    s._token = CancellationToken()
    s.history = []

    @contextmanager
    def _root_span(*a, **k):
        del a, k
        yield type("Span", (), {"set_attribute": lambda *x, **y: None})()

    monkeypatch.setattr(runtime, "root_span", _root_span)

    async def _noop(*a, **k):
        del a, k

    s._ensure_resumed = _noop  # type: ignore[attr-defined]

    class _CancellingRun:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise asyncio.CancelledError()

    class _CM:
        async def __aenter__(self):
            return _CancellingRun()

        async def __aexit__(self, *exc):
            del exc
            return False

    class _FakeAgent:
        def iter(self, *a, **k):
            del a, k
            return _CM()

    s._agent = _FakeAgent()  # type: ignore[attr-defined]

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            _collect(s.stream("hi", conversation_id="conv-1"))  # type: ignore[attr-defined]
        )


_FAKE_SHELL_PROFILE = _FakeProfile(name="test")


# --- Session-owned turn handle (durable turns, no agent build) -------------


def _fake_session() -> runtime.Session:
    """A Session shell (no agent build) for turn-handle mechanics."""
    s = object.__new__(runtime.Session)
    s._turn = None
    s._token = None
    return s


def test_start_turn_reattaches_and_fans_out() -> None:
    s = _fake_session()

    async def fake_stream(prompt, *, conversation_id=None, _token=None):  # noqa: ARG001
        for text in ["hi", "there"]:
            yield runtime.StreamEvent(kind="text", text=text)

    s.stream = fake_stream  # type: ignore[method-assign]

    async def scenario() -> None:
        h1 = s.start_turn("prompt", conversation_id="c1")
        assert s._turn is h1
        # A second start_turn re-attaches; the prompt is ignored.
        h2 = s.start_turn("ignored", conversation_id="c1")
        assert h2 is h1
        events = [ev async for ev in h1.subscribe()]
        assert [e.text for e in events] == ["hi", "there"]
        assert h1.done
        assert s._turn is None  # cleared once the turn settles

    asyncio.run(scenario())


def test_turn_handle_records_error_and_clears() -> None:
    s = _fake_session()

    async def boom(prompt, *, conversation_id=None, _token=None):  # noqa: ARG001
        yield runtime.StreamEvent(kind="text", text="x")
        raise ValueError("boom")

    s.stream = boom  # type: ignore[method-assign]

    async def scenario() -> None:
        h = s.start_turn("p")
        events = [ev async for ev in h.subscribe()]
        assert [e.text for e in events] == ["x"]
        assert isinstance(h.error, ValueError)
        assert h.done
        assert s._turn is None

    asyncio.run(scenario())
