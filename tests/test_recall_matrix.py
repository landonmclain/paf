"""Deterministic Tier-1 Challenge matrix for the Turn Index retrieval tools.

The escalation ladder under test (see Turn-Index-Testing.md):
    index read (in context, no tool)
      -> read_turn(ref)                  narrow, gated behind detected relevance
         -> search_index(q)              substring over one-liners + key_terms
            -> search_conversation_history(q)   expensive BM25 wide fallback

Tier-1 goals, per the testing doc:
  * Regression-safety: catch the "wiring scare" class of bug -- tools silently
    not exposed on the built agent -- WITHOUT a live model or API key.
  * Prove the *mechanics* of each tool (correct ref resolution, substring match
    across summary AND key_terms, conversation scoping) deterministically.
  * Prove the *escalation geometry*: the Challenge rows C1-C4 force exactly one
    mechanism to be able to answer. We gate on the TOOL CALL the agent makes
    (via pydantic-ai's FunctionModel emitting a ToolCallPart for the registered
    tool), not on the answer wording -- the doc's core lesson.

No live model. Runs in default `pytest`. Uses FunctionModel (scripted tool calls)
and a real FileStepStore + TurnIndexStore in a tmp dir.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import FileStepStore

from paf.capabilities.turn_index import TurnIndexCapability, TurnIndexStore, TurnIndexToolset


def _toolcall_model(tool_name: str, args: dict | None = None):
    """FunctionModel that FIRST calls `tool_name` with `args`, then answers when a tool result is present."""
    from pydantic_ai.models.function import FunctionModel

    call_args = args if args is not None else {}

    def handler(messages, info) -> ModelResponse:
        has_result = any(
            isinstance(m, ModelRequest)
            and any(isinstance(p, ToolReturnPart) for p in m.parts)
            for m in messages
        )
        if not has_result:
            return ModelResponse(parts=[ToolCallPart(tool_call_id="t1", tool_name=tool_name, args=call_args)])
        return ModelResponse(parts=[TextPart("answer")])

    return FunctionModel(handler, model_name=f"tier1-call-{tool_name}")


def _called_tools(result) -> list[str]:
    """Tool names ACTUALLY invoked by the agent during the run (from ToolReturnPart)."""
    called = []
    for m in result.all_messages():
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, ToolReturnPart):
                    called.append(p.tool_name)
    return called


# ---------------------------------------------------------------------------
# Tool exposure — the regression guard that catches the "wiring" class of bug
# ---------------------------------------------------------------------------


def test_all_retrieval_tools_exposed_on_built_agent() -> None:
    """The three retrieval mechanisms must be reachable from a built agent.

    This is the deterministic guard that would have caught the "search_index isn't
    registered" wiring scare (doc §Problem): assert the FunctionTools are present
    on the toolset the agent actually runs with.
    """
    from paf.specs import build_agent

    agent = build_agent("assistant")
    combined = next(ts for ts in agent.toolsets if type(ts).__name__ == "CombinedToolset")
    names: set[str] = set()
    for ts in combined.toolsets:
        wrapped = getattr(ts, "wrapped", ts)
        names.update(wrapped.tools.keys())
    # The escalation ladder's named tools must all be exposed.
    assert "read_turn" in names, f"read_turn missing; got {sorted(names)}"
    assert "search_index" in names, f"search_index missing; got {sorted(names)}"
    assert "search_conversation_history" in names, (
        f"search_conversation_history missing; got {sorted(names)}"
    )


def test_single_recall_ladder_when_memory_owns_instructions() -> None:
    """Memory owns the ONE composite recall ladder when co-present; TurnIndex and
    ConversationSearch suppress their instruction scraps so the model sees a single
    ladder, while all recall tools stay registered (suppressing instructions never
    suppresses tools).

    This is the regression guard for the unified-ladder change: the live assistant
    profile builds StepPersistence + ConversationSearch + TurnIndex + Memory +
    Compaction, and `CombinedCapability.get_instructions()` concatenates every
    capability's instructions. Exactly one must be non-empty, and it must be Memory's.
    """
    from paf.capabilities.build import build_runtime_capabilities
    from paf.capabilities.profiles import ASSISTANT_PROFILE

    caps = build_runtime_capabilities(ASSISTANT_PROFILE)
    instructions: list[str] = []
    for c in caps:
        inst = c.get_instructions()
        if not inst:
            continue
        parts = list(inst) if isinstance(inst, (list, tuple)) else [inst]
        for p in parts:
            if not callable(p):
                instructions.append(str(p))

    assert len(instructions) == 1, (
        f"expected exactly ONE recall ladder; got {len(instructions)}: {instructions!r}"
    )
    ladder = instructions[0]
    # Memory's composite ladder names the durable-fact rung first and the last-rung
    # history search last — that is the branch-by-intent ordering.
    assert "search_memory" in ladder, "Memory ladder must name search_memory"
    assert "search_index" in ladder, "Memory ladder must name search_index"
    assert "search_conversation_history" in ladder, (
        "Memory ladder must name search_conversation_history as the last rung"
    )
    assert "LAST RUNG" in ladder, "search_conversation_history must be marked last rung"
    # The suppressed capabilities must not leak a second, conflicting ladder.
    assert "TO RECALL PAST TURNS" not in ladder.upper()


def test_turn_index_owns_ladder_without_memory() -> None:
    """Back-compat: when Memory is NOT in the profile, TurnIndex still emits its own
    self-contained ladder (recognition -> search_index -> read_turn ->
    search_conversation_history) and ConversationSearch stays suppressed. This guards
    the "don't weaken the TurnIndex-alone path" risk of the unified-ladder change."""
    from paf.capabilities.build import build_runtime_capabilities
    from paf.capabilities.profiles import AgentProfile

    profile = AgentProfile(
        name="no-memory",
        capability_names=("StepPersistence", "ConversationSearch", "TurnIndex"),
    )
    caps = build_runtime_capabilities(profile)
    instructions: list[str] = []
    for c in caps:
        inst = c.get_instructions()
        if not inst:
            continue
        parts = list(inst) if isinstance(inst, (list, tuple)) else [inst]
        for p in parts:
            if not callable(p):
                instructions.append(str(p))

    assert len(instructions) == 1, (
        f"expected exactly ONE recall ladder; got {len(instructions)}: {instructions!r}"
    )
    ladder = instructions[0]
    assert "search_index" in ladder, "TurnIndex ladder must name search_index"
    assert "read_turn" in ladder, "TurnIndex ladder must name read_turn"
    assert "search_conversation_history" in ladder, (
        "TurnIndex ladder must name search_conversation_history"
    )


# ---------------------------------------------------------------------------
# Tool mechanics — deterministic, no model drive
# ---------------------------------------------------------------------------


class _FakeCtx:
    def __init__(self, conversation_id):
        self.conversation_id = conversation_id


def test_search_index_matches_summary_and_key_terms(tmp_path: Path) -> None:
    """`search_index` matches against BOTH the one-liner and key_terms."""
    idx = TurnIndexStore(tmp_path / "index")
    idx.append(conversation_id="c1", run_id="r1", turn_id=0, summary="the database is slow", key_terms=["postgres"])
    idx.append(conversation_id="c1", run_id="r1", turn_id=1, summary="unrelated chat about weather")
    source = SnapshotHistorySource(FileStepStore(directory=tmp_path / "steps"))
    ts = TurnIndexToolset(source, idx)

    async def go():
        # key_term match
        out = await ts.search_index(_FakeCtx("c1"), "postgres")
        assert "the database is slow" in out, out
        # summary substring match
        out = await ts.search_index(_FakeCtx("c1"), "database")
        assert "the database is slow" in out, out
        # no match
        out = await ts.search_index(_FakeCtx("c1"), "nonexistent-term")
        assert "No index entries" in out, out

    asyncio.run(go())


def test_search_index_scopes_to_conversation(tmp_path: Path) -> None:
    """`search_index` only returns entries for the CURRENT conversation."""
    idx = TurnIndexStore(tmp_path / "index")
    idx.append(conversation_id="c1", run_id="r1", turn_id=0, summary="alpha secret", key_terms=["leak"])
    idx.append(conversation_id="c2", run_id="r2", turn_id=0, summary="alpha secret", key_terms=["leak"])
    source = SnapshotHistorySource(FileStepStore(directory=tmp_path / "steps"))
    ts = TurnIndexToolset(source, idx)

    async def go():
        out = await ts.search_index(_FakeCtx("c1"), "leak")
        # c1 has 1 hit; c2's entry must NOT leak in.
        assert out.count("alpha secret") == 1, out

    asyncio.run(go())


def test_read_turn_resolves_and_renders_turn(tmp_path: Path) -> None:
    """`read_turn` resolves a ref back to the full original turn via persistence."""
    store = FileStepStore(directory=tmp_path / "steps")
    from pydantic_ai_harness.step_persistence import ContinuableSnapshot

    msgs = [
        ModelRequest(parts=[UserPromptPart("the deploy port is 8080")]),
        ModelResponse(parts=[TextPart("ok")]),
    ]
    idx = TurnIndexStore(tmp_path / "index")
    idx.append(conversation_id="c1", run_id="run-abc", turn_id=0, summary="deploy port")

    async def go():
        await store.save_snapshot(
            ContinuableSnapshot(run_id="run-abc", step_index=0, messages=msgs, conversation_id="c1")
        )
        source = SnapshotHistorySource(store)
        ts = TurnIndexToolset(source, idx)
        out = await ts._finder.resolve("run-abc:0", run_id="run-abc", conversation_id="c1")
        assert "the deploy port is 8080" in out, out
        # malformed ref
        out = await ts._finder.resolve("run-abc:nope", run_id="run-abc", conversation_id="c1")
        assert "Malformed" in out, out

    asyncio.run(go())


# ---------------------------------------------------------------------------
# Challenge geometry — escalation ladder forced by FunctionModel tool calls
# ---------------------------------------------------------------------------


def test_c2_search_index_escalation_forces_tool(tmp_path: Path) -> None:
    """C2: fact in key_terms past the injected tail => the agent MUST call search_index.

    We seed an index whose only matching vocabulary is a `key_term`, register the
    search tool, and drive a FunctionModel whose scripted FIRST response is a
    search_index call. Gate on the ToolReturnPart: the tool was invoked.

    NOTE: we do NOT pass `tools=` here -- TurnIndexCapability already registers
    `read_turn` / `search_index` via its own toolset, and the FunctionModel's
    emitted call targets those.
    """
    store = FileStepStore(directory=tmp_path / "steps")
    idx = TurnIndexStore(tmp_path / "index")
    idx.append(conversation_id="c2", run_id="r1", turn_id=3, summary="the lb address", key_terms=["blue-server-7"])
    agent = Agent(
        model=_toolcall_model("search_index", {"query": "blue-server-7"}),
        name="tier1-c2",
        capabilities=[TurnIndexCapability(source=SnapshotHistorySource(store), index_store=idx, max_lines=4)],
    )
    r = agent.run_sync("what server?", conversation_id="c2")
    called = _called_tools(r)
    assert "search_index" in called, f"expected search_index call; got {called}"


def test_c3_read_turn_escalation_forces_read(tmp_path: Path) -> None:
    """C3: one-liner names topic but the VALUE is only in the turn body => read_turn.

    The FunctionModel is scripted to call read_turn (it must, per geometry, since
    search_index alone can't surface the value stored only in the full turn).
    """
    store = FileStepStore(directory=tmp_path / "steps")
    idx = TurnIndexStore(tmp_path / "index")
    idx.append(conversation_id="c3", run_id="r1", turn_id=0, summary="the token is mentioned", key_terms=["token"])
    agent = Agent(
        model=_toolcall_model("read_turn", {"ref": "r1:0"}),
        name="tier1-c3",
        capabilities=[TurnIndexCapability(source=SnapshotHistorySource(store), index_store=idx, max_lines=4)],
    )
    r = agent.run_sync("what is the token value?", conversation_id="c3")
    called = _called_tools(r)
    assert "read_turn" in called, f"expected read_turn call; got {called}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-q", "-s", __file__]))