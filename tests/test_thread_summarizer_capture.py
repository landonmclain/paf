"""Curation + window hook tests for the thread summarizer input.

The 2026-09-08 re-architecture removed the eviction **capture WAL** (no
``EvictionBlock`` / ``attach_to_window`` / evictions.jsonl). What survived and
matters for the thread summarizer is the *curation contract*: the input fed to
the summarizer holds ONLY the durable user-prompt + turn-closing final answer —
thinking, intermediate tool-narration, and tool returns are dropped (see
``paf.capabilities._window._first_text``). The ``capture_hook`` on
``TurnBoundSlidingWindow`` remains as a generic eviction hook (no longer bound
by the summarizer) plus the per-agent capture directory helper.

These tests cover: the capture-dir helper, the window hook mechanics (correct
refs / OFF by default / no-eviction-no-hook / active-turn survival), and the
curation contract that drives the summarizer prompt.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from paf.capabilities._window import TurnBoundSlidingWindow
from paf.capabilities.thread_summarizer import capture_directory_for


def _turn(label: str, run_id: str | None = "run-z") -> list[ModelMessage]:
    """One (user, assistant) turn with ``run_id`` stamped on both messages."""
    req = ModelRequest(parts=[UserPromptPart(content=label)])
    resp = ModelResponse(parts=[TextPart(f"reply:{label}")])
    if run_id is not None:
        req.run_id = run_id
        resp.run_id = run_id
    return [req, resp]


def _ctx_like(conversation_id: str | None):
    """A minimal stand-in exposing ``conversation_id`` (RunContext has it)."""
    return type("Ctx", (), {"conversation_id": conversation_id})


def test_capture_directory_for_is_per_agent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # relative to CWD (mirrors memory's DEFAULT_MEMORY_DIR data/memory/<agent>)
    assert capture_directory_for("assistant") == Path("data") / "summaries" / "assistant"
    assert capture_directory_for(None) == Path("data") / "summaries" / "agent"


def test_window_fires_hook_with_correct_refs():
    # history long enough that keep_messages=2 evicts the oldest turns
    msgs: list[ModelMessage] = []
    for i in range(5):
        msgs.extend(_turn(f"q{i}", run_id="run-a" if i < 3 else "run-b"))
    captured: list[tuple[list[dict], int]] = []
    window = TurnBoundSlidingWindow(max_messages=100, keep_messages=2, capture_hook=lambda rows, keep: captured.append((rows, keep)))

    asyncio.run(window.compact(msgs, _ctx_like("prof:cid")))

    assert len(captured) == 1
    rows, keep = captured[0]
    assert keep == 2
    refs = [r["ref"] for r in rows]
    assert refs == sorted(refs)
    assert set(refs) == {"run-a:0", "run-a:1", "run-a:2"}
    assert refs[0] == refs[1]           # first turn's req+resp share run-a:0
    assert all("role" in r and "text" in r for r in rows)


def test_window_hook_off_by_default_no_capture():
    window = TurnBoundSlidingWindow(max_messages=100, keep_messages=2)
    msgs: list[ModelMessage] = []
    for i in range(5):
        msgs.extend(_turn(f"q{i}"))
    out = asyncio.run(window.compact(msgs, _ctx_like("prof:cid")))
    assert len(out) < len(msgs)


def test_window_no_eviction_no_hook():
    captured: list = []
    window = TurnBoundSlidingWindow(max_messages=100, keep_messages=2, capture_hook=lambda rows, keep: captured.append((rows, keep)))
    msgs = _turn("only-one-turn")
    asyncio.run(window.compact(msgs, _ctx_like("prof:cid")))
    assert captured == []  # nothing evictable -> hook not called


def test_window_never_evicts_active_turn():
    captured: list = []
    window = TurnBoundSlidingWindow(max_messages=100, keep_messages=2, capture_hook=lambda rows, keep: captured.append((rows, keep)))
    msgs: list[ModelMessage] = []
    for i in range(3):
        msgs.extend(_turn(f"q{i}"))
    out = asyncio.run(window.compact(msgs, _ctx_like("prof:cid")))
    assert out[-2].parts[0].content == "q2"
    assert out[-1].parts[0].content == "reply:q2"


# ---------------------------------------------------------------------------
# Curation contract — the summarizer is fed user prompt + final answer ONLY
# ---------------------------------------------------------------------------


def test_capture_drops_thinking_and_intermediate_keeps_final_text():
    """The curated span holds the model's FINAL answer, never thinking/narration.

    Regression guard for the thread-summarizer bloat fix: the summarizer input
    must carry only durable signal — the user prompt (cleaned) and the
    turn-closing model response. `ThinkingPart` (reasoning), intermediate
    responses that still call a tool, and tool results are dropped so summaries
    are never fed the model's mid-step narration.
    """
    from pydantic_ai.messages import (
        ThinkingPart,
        ToolCallPart,
        ToolReturnPart,
    )

    from paf.capabilities._window import _first_text

    user0 = ModelRequest(parts=[UserPromptPart(content="actual user question")], run_id="run-z")
    intermediate = ModelResponse(
        parts=[
            ThinkingPart("let me think about this..."),
            TextPart("Let me check the codebase first."),
            ToolCallPart(tool_name="read_file", args={"path": "x"}),
        ],
        run_id="run-z",
    )
    toolret = ModelRequest(
        parts=[ToolReturnPart(tool_name="read_file", content="file contents here")],
        run_id="run-z",
    )
    final0 = ModelResponse(
        parts=[
            ThinkingPart("let me finalize..."),
            TextPart("Let me check the codebase first."),
            TextPart("Here is the final answer."),
        ],
        run_id="run-z",
    )
    user1 = ModelRequest(parts=[UserPromptPart(content="next question")], run_id="run-z")

    msgs: list[ModelMessage] = [user0, intermediate, toolret, final0, user1]
    captured: list[tuple[list[dict], int]] = []
    window = TurnBoundSlidingWindow(max_messages=100, keep_messages=0, capture_hook=lambda rows, keep: captured.append((rows, keep)))
    asyncio.run(window.compact(msgs, _ctx_like("prof:cid")))

    assert captured  # something was evicted
    rows = captured[0][0]
    texts = [r["text"] for r in rows if r["text"] is not None]
    assert "actual user question" in texts
    assert "Here is the final answer." in texts
    for t in texts:
        assert "Let me check the codebase" not in (t or "")
        assert "file contents here" not in (t or "")
    assert "let me think about this..." not in texts

    # Same contract at the _first_text layer the SUMMARIZER uses directly:
    for msg in msgs:
        curated = _first_text(msg)
        if isinstance(msg, ModelResponse):
            assert curated is None or "Let me check the codebase" not in curated
