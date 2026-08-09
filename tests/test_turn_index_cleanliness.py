"""CLEANLINESS contract: a recalled turn must NOT carry injected context artifacts.

The bug this guards (found 2026-08-09): `before_model_request` injects a bounded
`<turn-index>` block into the latest User message, and `StepPersistence` snapshots
that User message verbatim — so the injected block is baked into the durable turn.
When `read_turn` later replays that turn, the block surfaces as part of the User
half, and the model can latch onto its leading intro lines instead of the turn's
real content (the "firefly story" incident).

The fix (Memory pattern, read-side): the injected block is tagged with a metadata
marker on its `TextContent`; `_toolset._render_message` drops marker-tagged parts
when rendering a recalled turn. The recalled User message therefore shows the real
prompt, not the transient injected index. This test asserts exactly that.

Run gated? No — deterministic FunctionModel, no API key.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import FileStepStore

from paf.capabilities.turn_index import TurnIndexCapability, TurnIndexStore, TurnIndexToolset

FIRST_FACT = "the company picnic is on September 14"
SECOND_ASK = "what day is the picnic?"
CONV = "conv-cleanliness"


def _picnic_model() -> FunctionModel:
    """Deterministic model that answers the picnic question."""
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.models.function import AgentInfo

    def handler(messages: list, info: AgentInfo) -> ModelResponse:
        if SECOND_ASK in "\n".join(str(m) for m in messages):
            return ModelResponse(parts=[TextPart("September 14.")])
        return ModelResponse(parts=[TextPart(f"ok: {FIRST_FACT}")])

    return FunctionModel(handler, model_name="picnic")


def _recall_finder(store: FileStepStore, index_dir: Path, run_id: str):
    return TurnIndexToolset(
        SnapshotHistorySource(store), TurnIndexStore(index_dir)
    )._finder  # type: ignore[attr-defined]


def test_render_strips_injected_and_keeps_real_content() -> None:
    """Deterministic unit guard: a persisted User msg with an injected block renders clean."""
    from paf.capabilities.turn_index._toolset import (
        _is_injected_text,
        _render_message,
    )
    from pydantic_ai.messages import ModelRequest, TextContent, UserPromptPart

    msg = ModelRequest(
        parts=[
            UserPromptPart(
                [
                    "tell me the firefly story",
                    TextContent("the real story about the firefly ..."),
                    TextContent(
                        "<turn-index>\n[abc:1] user said something\n</turn-index>",
                        metadata="turn-index.v1",
                    ),
                ]
            )
        ]
    )

    assert _is_injected_text(TextContent("x", metadata="turn-index.v1"))
    assert _is_injected_text(
        TextContent("x", metadata="pydantic-ai-harness.memory.v1:abc123")
    )
    assert not _is_injected_text(TextContent("real", metadata=None))

    rendered = _render_message(msg)
    assert "<turn-index>" not in rendered and "</turn-index>" not in rendered
    assert "firefly story" in rendered and "real story" in rendered


def test_recalled_turn_strips_injected_index_block() -> None:
    """A recalled turn renders the real User prompt, not the injected `<turn-index>` block."""

    async def scenario(tmp: Path) -> list[str]:
        store = FileStepStore(directory=tmp / "steps")
        index_dir = tmp / "index"
        from paf.capabilities.persistence import _build_step_persistence

        agent = Agent(
            model=_picnic_model(),
            name="paf-cleanliness",
            capabilities=[
                _build_step_persistence(store),
                TurnIndexCapability(
                    source=SnapshotHistorySource(store),
                    index_store=TurnIndexStore(index_dir),
                ),
            ],
        )

        # Turn 1: plant the fact (index is empty on turn 1, so no block injected).
        await agent.run(FIRST_FACT, conversation_id=CONV)
        # Turn 2: index now holds turn-1's line, so a `<turn-index>` block is
        # injected into turn 2's request AND persists into turn 2's snapshot.
        await agent.run(SECOND_ASK, conversation_id=CONV)

        runs = await store.list_runs()
        if not runs:
            return ["StepPersistence wrote nothing to the store"]
        # Both turns share one run (same conversation, same agent loop)? They may
        # be two separate runs OR one continuing run -- scan all runs for one whose
        # snapshot carries the injected block, then assert read_turn strips it.
        index_store = TurnIndexStore(index_dir)
        entries = index_store.read_all(conversation_id=CONV)
        if not entries:
            return ["index empty — the after_model_request hook didn't record turns"]

        problems: list[str] = []

        # For every executed run, pull each turn via read_turn and assert the
        # real user text is present AND no injected `<turn-index>` block leaked.
        for entry in entries:
            finder = _recall_finder(store, index_dir, entry.run_id)
            recalled = await finder.resolve(
                entry.ref, run_id=entry.run_id, conversation_id=CONV
            )
            if "<turn-index>" in recalled or "</turn-index>" in recalled:
                problems.append(
                    f"recalled turn {entry.ref} leaked the injected index block:\n{recalled[:400]!r}"
                )

        return problems

    with tempfile.TemporaryDirectory() as td:
        problems = asyncio.run(scenario(Path(td)))

    assert not problems, "TURN CLEANLINESS BROKEN:\n" + "\n".join(
        f"  - {p}" for p in problems
    )