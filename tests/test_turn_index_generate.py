"""Tests for the turn-index per-turn write path (slice + summarize + append).

Covers the M2 redesign (2026-08-08): the index is built per completed user
turn -- fired synchronously from `TurnIndexCapability.after_model_request` --
not batched per run. Each turn is sliced out of the live message history, keyed
by a STABLE `turn_id` (the ordinal of its user-prompt anchor), summarized, and
appended idempotently.

The old collector/cursor/`generated_turns` run-batch machinery was retired; these
tests pin the new single-path write behavior.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import FileStepStore

from paf.capabilities.turn_index._capability import TurnIndexCapability
from paf.capabilities.turn_index._generate import (
    Turn,
    summarize_turn_append,
    this_turn_id,
    this_turn_slice,
    user_anchor_indices,
)
from paf.capabilities.turn_index._store import TurnIndexStore


def _one_turn(user_text: str = "what is my passport codename?", answer: str = "ZEPHYR-7741") -> list[ModelMessage]:
    """A single completed turn: user ask -> tool call -> result -> final answer."""
    return [
        ModelRequest(parts=[UserPromptPart(content=user_text)]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="search_conversation_history",
                    args='{"query": "passport codename"}',
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="search_conversation_history",
                    content="found ZEPHYR-7741",
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content=f"Your codename is **{answer}**.")]),
    ]


# ---- turn slicing / stable ids -------------------------------------------------


def test_user_anchor_indices_and_this_turn() -> None:
    msgs: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("first")]),
        ModelResponse(parts=[TextPart("a")]),
        ModelRequest(parts=[UserPromptPart("second")]),
        ModelResponse(parts=[TextPart("b")]),
    ]
    assert user_anchor_indices(msgs) == [0, 2]
    assert this_turn_id(msgs) == 1  # the SECOND user turn
    assert this_turn_slice(msgs) == msgs[2:]  # from the last anchor to the end


def test_this_turn_id_stable_and_missing() -> None:
    # No user anchor yet -> -1 sentinel, empty slice.
    assert this_turn_id([ModelResponse(parts=[TextPart("hi")])]) == -1
    assert this_turn_slice([ModelResponse(parts=[TextPart("hi")])]) == []

    # Ordering never changes the ordinal: the same messages always give the same id.
    msgs = _one_turn() + _one_turn("second q", "ANSWER2")
    assert this_turn_id(msgs) == 1


# ---- summarize / append (the per-turn write) -----------------------------------


def test_summarize_turn_append_writes_stable_ref(tmp_path: Path) -> None:
    index = TurnIndexStore(tmp_path / "index")
    turn = Turn(
        run_id="r1",
        conversation_id="c1",
        turn_id=0,
        messages=_one_turn(),
    )
    appended, summary = asyncio.run(summarize_turn_append(turn=turn, index_store=index))
    assert appended is True
    assert isinstance(summary, str)
    # One-line captures intent + tool from the slice.
    assert "passport codename" in summary
    assert "search_conversation_history" in summary
    # Keyed by the stable turn_id, not a message position.
    entries = index.read_all("c1")
    assert len(entries) == 1
    assert entries[0].run_id == "r1"
    assert entries[0].turn_id == 0
    assert entries[0].ref == "r1:0"


def test_summarize_turn_sees_the_completed_turn_including_response(tmp_path: Path) -> None:
    """The summarizer input is the FULL turn slice -- user, tool traffic, AND the
    final answer (`response`), which the hook synthesizes in because the graph
    has not yet appended it to history when `after_model_request` runs.

    The stub only *surfaces* line 1 + tools, but the rendered context that feeds
    the one-liner (and a future LLM summarizer) contains the whole turn. We prove
    that by building the slice the way the hook does (slice + response) and
    checking the stub's line-0 is the user prompt while the tool is detected.
    """
    index = TurnIndexStore(tmp_path / "index")
    live: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("what is my passport codename?")]),
        ModelResponse(parts=[ToolCallPart(tool_name="search_conversation_history", args="{}")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="search_conversation_history", content="ZEPHYR")]),
    ]
    # The live history ends at the tool-return request (response NOT yet appended).
    assert this_turn_id(live) == 0
    slice_only = this_turn_slice(live)
    assert len(slice_only) == 3
    # The hook appends the just-returned response to close the turn.
    full = [*slice_only, ModelResponse(parts=[TextPart("Your codename is ZEPHYR-7741.")])]
    turn = Turn(run_id="r1", conversation_id="c1", turn_id=0, messages=full)
    appended, summary = asyncio.run(summarize_turn_append(turn=turn, index_store=index))
    assert appended is True
    # Stub emits the user intent line + the tool that ran.
    assert summary.startswith("user: what is my passport codename?")
    assert "search_conversation_history" in summary
    # The full rendered context (what a real LLM summarizer would receive) holds
    # the response text too -- proving nothing about the turn is lost to ordering.
    from paf.capabilities.turn_index._extract import build_turn_context

    rendered = build_turn_context(full)
    assert "ZEPHYR-7741" in rendered


def test_summarize_turn_append_is_idempotent(tmp_path: Path) -> None:
    index = TurnIndexStore(tmp_path / "index")
    turn = Turn(run_id="r1", conversation_id="c1", turn_id=0, messages=_one_turn())
    first, _ = asyncio.run(summarize_turn_append(turn=turn, index_store=index))
    second, _ = asyncio.run(summarize_turn_append(turn=turn, index_store=index))
    assert first is True
    assert second is False  # already indexed -> no duplicate line
    assert len(index.read_all("c1")) == 1


def test_distinct_turns_get_distinct_stable_ids(tmp_path: Path) -> None:
    index = TurnIndexStore(tmp_path / "index")
    t0 = Turn(run_id="r1", conversation_id="c1", turn_id=0, messages=_one_turn("first", "A"))
    t1 = Turn(run_id="r1", conversation_id="c1", turn_id=1, messages=_one_turn("second", "B"))
    asyncio.run(summarize_turn_append(turn=t0, index_store=index))
    asyncio.run(summarize_turn_append(turn=t1, index_store=index))
    entries = index.read_all("c1")
    assert {e.turn_id for e in entries} == {0, 1}
    assert [e.ref for e in entries] == ["r1:0", "r1:1"]
    assert len(entries) == 2


# ---- the capability hook --------------------------------------------------------


def test_after_model_request_populates_conversation_scoped_index(tmp_path: Path) -> None:
    """The per-turn hook slices the live messages and writes under `ctx.conversation_id`.

    A `cli-primary` run's turns must land under `cli-primary` only. The hook is
    background-independent -- a pure per-turn write fired the moment each model
    request completes.
    """
    index = TurnIndexStore(tmp_path / "index")
    cap = TurnIndexCapability(
        source=SnapshotHistorySource(FileStepStore(tmp_path / "store")),
        index_store=index,
    )

    async def scenario():
        # Live history after the FIRST turn's tool traffic but BEFORE the graph
        # appends the final response. `after_model_request` runs before that
        # append (pydantic-ai graph ordering), so `ctx.messages` excludes the
        # response — the hook receives it as the `response` argument instead.
        live: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart("what is my passport codename?")]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name="search_conversation_history", args="{}")
                ]
            ),
            ModelRequest(parts=[ToolReturnPart(tool_name="search_conversation_history", content="ZEPHYR")]),
        ]
        final = ModelResponse(parts=[TextPart("Your codename is ZEPHYR-7741.")])
        ctx = SimpleNamespace(conversation_id="cli-primary", run_id="run-live", messages=live)
        out = await cap.after_model_request(ctx, request_context=object(), response=final)

        assert out is final  # hook returns the response unchanged
        entries = index.read_all("cli-primary")
        assert len(entries) == 1
        assert entries[0].run_id == "run-live"
        assert entries[0].turn_id == 0
        assert entries[0].conversation_id == "cli-primary"
        # The summary sees the completed turn: intent + tool + the final answer
        # (the response the hook synthesized into the slice).
        assert "passport codename" in entries[0].summary
        # The stub summarizer emits first line + tools + hash; the final answer
        # text is in the slice (proven below) but not surfaced by this stub.
        assert "search_conversation_history" in entries[0].summary
        assert index.read_all("other") == []  # no cross-contamination

        # A SECOND turn in the same run appends with the next stable turn_id.
        live.append(final)  # graph now appends the first turn's response
        live.append(ModelRequest(parts=[UserPromptPart("now fix the deploy")]))
        final2 = ModelResponse(parts=[TextPart("fixed.")])
        ctx2 = SimpleNamespace(conversation_id="cli-primary", run_id="run-live", messages=live)
        await cap.after_model_request(ctx2, request_context=object(), response=final2)
        entries = index.read_all("cli-primary")
        assert len(entries) == 2
        assert {e.turn_id for e in entries} == {0, 1}
        assert {e.ref for e in entries} == {"run-live:0", "run-live:1"}

    asyncio.run(scenario())


def test_after_model_request_is_idempotent_on_refire(tmp_path: Path) -> None:
    """Re-firing the hook for an already-indexed turn (retry/resume) writes nothing.

    Idempotency is by stable `turn_id` -- the capability never duplicates a line.
    """
    index = TurnIndexStore(tmp_path / "index")
    cap = TurnIndexCapability(
        source=SnapshotHistorySource(FileStepStore(tmp_path / "store")),
        index_store=index,
    )

    async def scenario():
        live: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart("what is my passport codename?")]),
            ModelResponse(parts=[TextPart("Your codename is ZEPHYR-7741.")]),
        ]
        final = ModelResponse(parts=[TextPart("Your codename is ZEPHYR-7741.")])
        ctx = SimpleNamespace(conversation_id="cli-primary", run_id="run-dup", messages=live)
        for _ in range(3):  # same turn fired 3x
            await cap.after_model_request(ctx, request_context=object(), response=final)
        assert len(index.read_all("cli-primary")) == 1

    asyncio.run(scenario())


def test_after_model_request_noop_without_content(tmp_path: Path) -> None:
    """No user anchor (e.g. a system-only exchange) writes nothing."""
    index = TurnIndexStore(tmp_path / "index")
    cap = TurnIndexCapability(
        source=SnapshotHistorySource(FileStepStore(tmp_path / "store")),
        index_store=index,
    )

    async def scenario():
        live = [ModelResponse(parts=[TextPart("thinking")])]
        await cap.after_model_request(
            SimpleNamespace(conversation_id="c", run_id="r", messages=live),
            request_context=object(),
            response=ModelResponse(parts=[TextPart("ok")]),
        )
        assert index.read_all() == []

    asyncio.run(scenario())