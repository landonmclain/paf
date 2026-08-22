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

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
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
    combined = next(ts for ts in agent.toolsets if type(ts).__name__ == "CombinedToolset")
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
    for mtool in ("search_memory", "read_memory", "write_memory", "delete_memory", "append_memory"):
        assert mtool not in names, f"{mtool} should be absent (inject-only); got {sorted(names)}"
    # `search_index` was the retired TurnIndex-only tool — gone.
    assert "search_index" not in names, f"search_index should be absent; got {sorted(names)}"


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
            ContinuableSnapshot(run_id="run-abc", step_index=0, messages=msgs, conversation_id="c1")
        )
        source = SnapshotHistorySource(store)
        finder = TurnFinder(source)
        out = await finder.resolve("run-abc:0", run_id="run-abc", conversation_id="c1")
        assert "the deploy port is 8080" in out, out
        # malformed ref
        out = await finder.resolve("run-abc:nope", run_id="run-abc", conversation_id="c1")
        assert "Malformed" in out, out

    asyncio.run(go())
