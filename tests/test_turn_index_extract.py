"""Tests for `_extract`: rendering a full run into the summarizer's context.

The key contract (Landon 2026-08-08): **send the whole run**, truncating ONLY
tool-call arguments and tool results; the user prompt and assistant's final
text pass through whole.
"""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from paf.capabilities.turn_index._extract import (
    ARG_CAP,
    RESULT_CAP,
    build_turn_context,
    truncate,
)


def _question_run() -> list[object]:
    return [
        ModelRequest(parts=[UserPromptPart(content="what is my passport codename?")]),
        ModelResponse(
            parts=[ToolCallPart(tool_name="search", args='{"query": "codename"}')]
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="search", content="found ZEPHYR-7741")]
        ),
        ModelResponse(parts=[TextPart(content="Your codename is **ZEPHYR-7741**.")]),
    ]


def test_truncate_passthrough_when_short_or_unbounded() -> None:
    assert truncate("hello", ARG_CAP) == "hello"  # under arg cap
    assert truncate("short", None) == "short"  # unbounded
    assert truncate("x", RESULT_CAP) == "x"


def test_truncate_applies_marker_when_over_limit() -> None:
    big = "Z" * 9000
    out = truncate(big, RESULT_CAP)
    assert len(out) <= RESULT_CAP
    assert "[truncated]" in out
    # Marker is suffix so a reviewer can always see an elision happened.
    assert out.endswith("[truncated]")


def test_context_keeps_user_prompt_whole() -> None:
    prompt = "what is my passport codename?"
    ctx = build_turn_context(_question_run())
    assert prompt in ctx


def test_context_keeps_assistant_final_text_whole() -> None:
    ctx = build_turn_context(_question_run())
    assert "Your codename is **ZEPHYR-7741**." in ctx


def test_context_renders_tool_call_with_args() -> None:
    ctx = build_turn_context(_question_run())
    assert 'tool call: search({"query": "codename"})' in ctx


def test_context_caps_tool_result_only() -> None:
    big = "RESULT_" + ("X" * 9000)
    run = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name="big_tool", content=big),
                UserPromptPart(content="keep me whole"),
            ]
        )
    ]
    ctx = build_turn_context(run)
    assert "keep me whole" in ctx          # user prompt untouched
    assert "[truncated]" in ctx            # result capped
    # Neither the assistant text nor the user prompt got the marker/suffix.
    assert "user: keep me whole[truncated]" not in ctx


def test_context_caps_tool_call_args_only() -> None:
    big = '{"q": "' + ("A" * 9000) + '"}'
    run = [
        ModelResponse(parts=[ToolCallPart(tool_name="search", args=big)]),
    ]
    ctx = build_turn_context(run)
    assert 'tool call: search(' in ctx
    assert "[truncated]" in ctx


def test_context_keeps_every_part_ordered() -> None:
    ctx = build_turn_context(_question_run())
    lines = [ln for ln in ctx.splitlines() if ln.strip()]
    # user -> tool call -> tool result -> assistant
    assert lines[0].startswith("user:")
    assert lines[1].startswith("tool call:")
    assert lines[2].startswith("tool result")
    assert lines[3].startswith("assistant:")


def test_context_unknown_kind_not_dropped() -> None:
    # A message that is neither a ModelRequest nor ModelResponse must still appear.
    class _Bogus:
        def __str__(self) -> str:  # pragma: no cover - simple fake
            return "BOGUS-MESSAGE"

    ctx = build_turn_context([_Bogus()])
    assert "BOGUS-MESSAGE" in ctx


def test_context_strips_injected_turn_index_block() -> None:
    """The summarizer must NOT see the injected `<turn-index>` block."""
    from pydantic_ai.messages import TextContent

    run = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "what is my codename?",
                        TextContent(
                            "<turn-index>\n[abc:1] user asked earlier\n</turn-index>",
                            metadata="turn-index.v1",
                        ),
                    ]
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content="Your codename is ZEPHYR-7741.")]),
    ]
    ctx = build_turn_context(run)
    assert "<turn-index>" not in ctx
    assert "my codename" in ctx  # real user prompt survives
    assert "ZEPHYR-7741" in ctx  # assistant text survives