"""Recall retrieval surface — deterministic, no live model.

Covers the retrieval TOOLS the built assistant exposes and the `read_turn`
full-content read primitive (owned by `search`). The turn-index-specific tools
(`search_index`, index ladder, escalation geometry) were retired with the
turn-index experiment — see `tests/retired/` and `archive/turn_index/`.

No live model. Runs in default `pytest`. Uses a real FileStepStore in a tmp dir.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import FileStepStore

from paf.capabilities.search._reader import TurnFinder


def test_all_retrieval_tools_exposed_on_built_agent() -> None:
    """The retrieval mechanisms must be reachable from a built agent.

    The assistant profile uses the memory + BM25 stack, and its memory surface is
    the M2 INJECT-ONLY contract (`expose_tools=False`): the journal-rendered
    notebook is injected in-context and the recall ladder is
    `search_conversation_history` / `read_turn` — the subject carries NO memory
    tools at all (not even read/search). Memory is injected, not tooled.
    """
    from paf.specs import build_agent

    agent = build_agent("assistant")
    combined = next(
        ts for ts in agent.toolsets if type(ts).__name__ == "CombinedToolset"
    )
    names: set[str] = set()
    for ts in combined.toolsets:
        wrapped = getattr(ts, "wrapped", ts)
        names.update(wrapped.tools.keys())
    # The BM25 recall rungs must be exposed.
    assert "search_conversation_history" in names, (
        f"search_conversation_history missing; got {sorted(names)}"
    )
    # `read_turn` is owned by the search capability (ConversationSearch -> ReadTurn).
    assert "read_turn" in names, f"read_turn missing; got {sorted(names)}"
    # M2 INJECT-ONLY memory: the subject must carry NO memory tools.
    for mtool in (
        "search_memory",
        "read_memory",
        "write_memory",
        "delete_memory",
        "append_memory",
    ):
        assert mtool not in names, (
            f"{mtool} should be absent (inject-only); got {sorted(names)}"
        )
    # `search_index` was the retired TurnIndex-only tool — gone.
    assert "search_index" not in names, (
        f"search_index should be absent; got {sorted(names)}"
    )


def test_read_turn_resolves_and_renders_turn(tmp_path: Path) -> None:
    """`read_turn` resolves a ref back to the full original turn via persistence."""
    store = FileStepStore(directory=tmp_path / "steps")
    from pydantic_ai_harness.step_persistence import ContinuableSnapshot

    msgs = [
        ModelRequest(parts=[UserPromptPart("the deploy port is 8080")]),
        ModelResponse(parts=[TextPart("ok")]),
    ]

    async def go():
        await store.save_snapshot(
            ContinuableSnapshot(
                run_id="run-abc", step_index=0, messages=msgs, conversation_id="c1"
            )
        )
        source = SnapshotHistorySource(store)
        finder = TurnFinder(source)
        out = await finder.resolve("run-abc:0", run_id="run-abc", conversation_id="c1")
        assert "the deploy port is 8080" in out, out
        # malformed ref
        out = await finder.resolve(
            "run-abc:nope", run_id="run-abc", conversation_id="c1"
        )
        assert "Malformed" in out, out


def test_read_turn_truncates_by_default_and_full_is_byte_complete(
    tmp_path: Path,
) -> None:
    """`read_turn` returns tool calls/results (not just user/assistant text) and
    truncates them by default; `full=True` renders the byte-complete originals."""
    store = FileStepStore(directory=tmp_path / "steps")
    from pydantic_ai_harness.step_persistence import ContinuableSnapshot

    LONG = "x" * 2000
    LONG_ARG = '{"query": "' + "y" * 900 + '"}'

    msgs = [
        ModelRequest(parts=[UserPromptPart("help " + "z" * 900)]),
        # ModelResponse: assistant decides to call a tool (long args).
        ModelResponse(parts=[ToolCallPart(tool_name="read_file", args=LONG_ARG)]),
        # ModelRequest carrying the tool RETURN (long payload) back to the model.
        ModelRequest(parts=[ToolReturnPart(tool_name="read_file", content=LONG)]),
    ]

    async def go():
        await store.save_snapshot(
            ContinuableSnapshot(
                run_id="run-tr", step_index=0, messages=msgs, conversation_id="c1"
            )
        )
        source = SnapshotHistorySource(store)
        finder = TurnFinder(source)

        # Default (bounded) read: tool call + tool return present, but truncated.
        bounded = await finder.resolve(
            "run-tr:0", run_id="run-tr", conversation_id="c1", full=False
        )
        assert "Tool [read_file]" in bounded, bounded  # tool return no longer dropped
        assert "Tool Call [read_file]" in bounded, bounded  # tool call rendered
        assert "z" * 900 not in bounded, bounded  # user text capped at 500
        assert "y" * 900 not in bounded, bounded  # tool call args capped at 200
        assert "x" * 2000 not in bounded, bounded  # tool return capped at 500
        assert bounded.count("...") >= 3, bounded

        # full=True: byte-complete originals.
        full = await finder.resolve(
            "run-tr:0", run_id="run-tr", conversation_id="c1", full=True
        )
        assert "z" * 900 in full, "user text should be full when full=True"
        assert "y" * 900 in full, "tool call args should be full when full=True"
        assert "x" * 2000 in full, "tool return payload should be full when full=True"
        assert "..." not in full, full

    asyncio.run(go())
