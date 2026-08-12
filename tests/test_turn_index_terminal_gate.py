"""Regression test: TurnIndexCapability skips indexing tool-call responses.

The after_model_request hook fires once per *model request*, which in an
agentic tool loop happens once per tool iteration — not once per completed user
turn. Without a gate, intermediate "here's the tool to call next" responses get
summarized and written to the index as fake turns like
`"Searched turn index for ..."`, which then re-retrieve and confuse later turns.

Terminal-response gate (2026-08-12): skip when the response carries a
`ToolCallPart`; only index responses that close a user turn (final text /
structured output with no tool call).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)

from paf.capabilities.turn_index import TurnIndexCapability, TurnIndexStore, TurnSummary


class _StubSummarizer:
    """Returns a fixed summary; no model/API key required."""

    def __init__(self) -> None:
        self.calls = 0

    async def summarize(self, rendered: str) -> TurnSummary:
        self.calls += 1
        # Keep the stub's summary short but *embed* the rendered turn content
        # so tests can assert that the actual turn was fed to the summarizer.
        return TurnSummary(
            sentence=f"STUB({self.calls}): {rendered}",
            key_terms=["term"],
            outcome="resolved",
        )


def _ctx(messages: list, run_id: str, conversation_id: str):
    return type("Ctx", (), {"messages": messages, "run_id": run_id, "conversation_id": conversation_id})()


def _history_with_one_user_turn() -> list:
    """A minimal history: one completed user prompt (no prior model response)."""
    return [
        ModelRequest(
            parts=[UserPromptPart(content="what is my 5K PB?")],
        )
    ]


def _make_cap(tmp_path: Path) -> tuple[TurnIndexCapability, _StubSummarizer]:
    cap = TurnIndexCapability(
        source=None,  # read path not exercised here
        index_store=TurnIndexStore(tmp_path),
    )
    stub = _StubSummarizer()
    cap.summarizer = stub
    return cap, stub


def test_tool_call_response_not_indexed(tmp_path: Path) -> None:
    """An intermediate tool-call response must NOT write an index entry."""
    cap, stub = _make_cap(tmp_path)
    ctx = _ctx(_history_with_one_user_turn(), run_id="run-a", conversation_id="conv-1")

    # A model response whose only role is "call tool X" — the middle of a loop.
    tool_resp = ModelResponse(
        parts=[ToolCallPart(tool_name="search_index", args={"q": "fence"})]
    )

    out = asyncio.run(
        cap.after_model_request(ctx, request_context=None, response=tool_resp)
    )

    # Returns the response unchanged (hook contract), and records nothing.
    assert out is tool_resp
    assert cap.index_store.count(conversation_id="conv-1") == 0
    assert stub.calls == 0


def test_final_text_response_indexed(tmp_path: Path) -> None:
    """A response that closes the user turn (text, no tool call) IS indexed."""
    cap, stub = _make_cap(tmp_path)
    ctx = _ctx(_history_with_one_user_turn(), run_id="run-a", conversation_id="conv-1")

    final_resp = ModelResponse(parts=[TextPart(content="Your personal best is 25:50.")])

    asyncio.run(cap.after_model_request(ctx, request_context=None, response=final_resp))

    # Exactly one turn indexed, at the anchor ordinal 0, with the real turn fed
    # to the summarizer (the stub embeds the rendered user+assistant content).
    assert stub.calls == 1
    entries = cap.index_store.read_all(conversation_id="conv-1")
    assert len(entries) == 1
    assert entries[0].turn_id == 0
    assert "what is my 5K PB?" in entries[0].summary
    assert "25:50" in entries[0].summary
