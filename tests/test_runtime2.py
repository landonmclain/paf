"""Offline unit tests for ``paf.runtime2`` — next-gen Session2 event contract.

These run WITHOUT a model or API key. They exercise the pure, deterministic
seam of runtime2: :func:`_dispatch_stream_event` (the Phase 1+2 turn-event
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

from datetime import UTC, datetime

from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai.messages import (
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)

from paf import runtime2


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _call_event(tool: str = "run_command", sid: str = "call_123") -> FunctionToolCallEvent:
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
    out = runtime2._dispatch_stream_event(_call_event(sid="call_step_1"))
    kinds = [type(e).__name__ for e in out]
    assert kinds == ["StreamEvent", "ToolRunning"]
    assert out[0].kind == "tool_call"
    assert out[0].tool_name == "run_command"
    running = out[1]
    assert isinstance(running, runtime2.ToolRunning)
    assert running.step_id == "call_step_1"  # stable id -> UI flips ONE row
    assert running.tool_name == "run_command"


def test_successful_tool_result_has_no_tool_error() -> None:
    out = runtime2._dispatch_stream_event(
        _result_event(sid="call_step_1", outcome="success", content="done")
    )
    assert [type(e).__name__ for e in out] == ["StreamEvent"]
    assert out[0].kind == "tool_result"
    assert out[0].text == "done"


def test_failed_tool_result_emits_tool_error_same_step_id() -> None:
    out = runtime2._dispatch_stream_event(
        _result_event(sid="call_fail_7", outcome="failed", content="boom")
    )
    res, err = out
    assert res.kind == "tool_result"
    assert isinstance(err, runtime2.ToolError)
    assert err.step_id == "call_fail_7"  # correlates with ToolRunning
    assert "boom" in err.error


def test_retry_prompt_result_emits_tool_error() -> None:
    out = runtime2._dispatch_stream_event(_retry_event(sid="call_retry"))
    assert isinstance(out[-1], runtime2.ToolError)
    assert out[-1].step_id == "call_retry"


# --- _dispatch_stream_event: Phase 1 text/thinking parity ------------------


def test_part_start_text_leading() -> None:
    from pydantic_ai import PartStartEvent

    ev = PartStartEvent(part=TextPart(content="hello"), index=0)
    out = runtime2._dispatch_stream_event(ev)
    assert [e.kind for e in out] == ["text"]
    assert out[0].text == "hello"


def test_part_delta_text_and_thinking() -> None:
    from pydantic_ai import PartDeltaEvent

    out = runtime2._dispatch_stream_event(
        PartDeltaEvent(delta=TextPartDelta(content_delta="abc"), index=0)
    )
    assert [e.kind for e in out] == ["text"]
    assert out[0].text == "abc"

    out2 = runtime2._dispatch_stream_event(
        PartDeltaEvent(delta=ThinkingPartDelta(content_delta="hmm"), index=0)
    )
    assert [e.kind for e in out2] == ["thinking"]
    assert out2[0].text == "hmm"


# --- background-task registry (no agent build) ----------------------------


def _bare_session2() -> runtime2.Session2:
    s = object.__new__(runtime2.Session2)
    s._background_tasks = {}
    s._task_subscribers = []
    s._consolidation_seq = 0
    return s


def test_subscribe_replays_active_task() -> None:
    s = _bare_session2()
    seen: list[tuple[str, str]] = []

    def cb(ev: runtime2.BackgroundEvent) -> None:
        seen.append((ev.kind, ev.status))

    s.subscribe_tasks(cb)
    t = runtime2.BackgroundTask(id="consolidation.c.1", label="Consolidating memory (c)")
    s._publish_task(t)  # queued (default)

    # Late subscriber must see the in-flight (queued) task replayed.
    seen2: list[tuple[str, str]] = []
    s.subscribe_tasks(lambda ev: seen2.append((ev.kind, ev.status)))
    assert ("task:Consolidating memory (c)", "queued") in seen2

    # Transition to done reaches both subscribers.
    t.status = runtime2.DONE
    s._publish_task(t)
    assert ("task:Consolidating memory (c)", "done") in seen
    assert ("task:Consolidating memory (c)", "done") in seen2


def test_done_task_not_replayed() -> None:
    s = _bare_session2()
    s.subscribe_tasks(lambda ev: None)
    t = runtime2.BackgroundTask(id="consolidation.d.1", label="L")
    t.status = runtime2.DONE
    s._publish_task(t)
    seen: list = []
    s.subscribe_tasks(lambda ev: seen.append(ev))
    assert seen == []  # only queued/running are active; done is not replayed


# --- profile wiring --------------------------------------------------------


def test_runtime2_test_profile_is_shell_only() -> None:
    from paf.capabilities import PROFILES

    prof = PROFILES["runtime2_test"]
    assert prof.capability_names == ("Shell",)
    # Capability resolves to the Shell type (buildable without a model).
    types = prof.capability_types()
    assert len(types) == 1
    assert types[0].__name__ == "Shell"


def test_runtime2_test_spec_resolves_to_profile() -> None:
    from paf.specs import resolve_profile_name

    assert resolve_profile_name("runtime2_test") == "runtime2_test"
