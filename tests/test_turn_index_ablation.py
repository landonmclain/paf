"""Regression: ABLATE_TURN_INDEX removes ALL of TurnIndex from the pipeline.

The ablation's point is NOT just dropping TurnIndex's tools — it's that NO
TurnIndex instance may survive anywhere in the agent, because TurnIndex injects
a `<turn-index>` block into context (before_model_request). Landon: "all of
turn index needs to be gone. otherwise it will show in context." These tests
assert, for the ablated pipeline built through the shared `_build_agent`:

  1. zero `TurnIndexCapability` instances anywhere in agent.root_capability,
  2. `read_turn` / `search_index` tools absent from the built toolset,
  3. `search_conversation_history` still present (ConversationSearch is the ladder),
  4. ConversationSearch.add_instructions == True (it owns the steering),
  and, for the baseline, the mirror: TurnIndex present with add_instructions=False
  on ConversationSearch (TurnIndex owns the ladder) — so an over-eager ablation
  can't silently strip the default pipeline.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai_harness.step_persistence import FileStepStore

from paf.capabilities.search._capability import ConversationSearch
from paf.capabilities.turn_index._capability import TurnIndexCapability
from tests import framework as R


def _root(agent: Agent) -> Any:
    return getattr(agent, "root_capability", None) or getattr(
        agent, "_root_capability", None
    )


def _collect(node: Any, kind: type, found: list) -> None:
    if node is None:
        return
    if isinstance(node, kind):
        found.append(node)
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            _collect(item, kind, found)
    elif isinstance(node, dict):
        for value in node.values():
            _collect(value, kind, found)
    elif hasattr(node, "__dataclass_fields__"):
        for f in node.__dataclass_fields__.values():
            _collect(getattr(node, f.name, None), kind, found)


def _build(ablate: bool) -> Agent:
    R.ABLATE_TURN_INDEX = ablate
    with tempfile.TemporaryDirectory() as d:
        store = FileStepStore(directory=d + "/persist")
        return R._build_agent(store, Path(d) / "index")


def _turn_index_caps(agent: Agent) -> list[TurnIndexCapability]:
    found: list[TurnIndexCapability] = []
    _collect(_root(agent), TurnIndexCapability, found)
    return found


def _conversation_caps(agent: Agent) -> list[ConversationSearch]:
    found: list[ConversationSearch] = []
    _collect(_root(agent), ConversationSearch, found)
    return found


def _tool_names(agent: Any) -> set[str]:
    combined = next(
        ts for ts in agent.toolsets if type(ts).__name__ == "CombinedToolset"
    )
    names: set[str] = set()
    for ts in combined.toolsets:
        wrapped = getattr(ts, "wrapped", ts)
        names.update(wrapped.tools.keys())
    return names


def test_ablation_removes_turn_index_entirely() -> None:
    agent = _build(ablate=True)
    # No TurnIndex instance at all -> no <turn-index> injection, no hooks.
    assert _turn_index_caps(agent) == [], "TurnIndex must be fully absent when ablated"
    tools = _tool_names(agent)
    assert "read_turn" not in tools
    assert "search_index" not in tools
    assert "search_conversation_history" in tools
    # ConversationSearch inherits the ladder.
    assert all(c.add_instructions for c in _conversation_caps(agent))


def test_baseline_still_carries_turn_index() -> None:
    agent = _build(ablate=False)
    assert _turn_index_caps(agent), "default pipeline must keep TurnIndex"
    tools = _tool_names(agent)
    assert "read_turn" in tools
    assert "search_index" in tools
    assert "search_conversation_history" in tools
    # TurnIndex owns the ladder; ConversationSearch stays silent.
    assert all(not c.add_instructions for c in _conversation_caps(agent))
