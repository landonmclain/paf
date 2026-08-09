"""Tier-2 live recall test — Option-B-lite (cf. Turn-Index-Testing.md).

The value-add tier: it tests REAL model judgment over the escalation ladder.
Unlike Tier 1 (FunctionModel, deterministic), here a real chat model must
*recognize* it needs a retrieval tool and invoke the right one.

What "live" buys us (the doc's two-part assertion):
  1. TOOL INVOCATION is the PASS/FAIL gate (not the answer string). We collect
     `FunctionToolCallEvent` from `run_stream_events` and assert the model
     actually called the expected tool -- catching the "claimed it searched but
     didn't" failure mode that a single-utterance check can dodge.
  2. COLD-START PERSISTENCE: we SEED with a real model turn (persisted via
     StepPersistence into the shared store + index), then build a FRESH agent
     over the SAME store + SAME cid and recall. A rebuild re-reads the store
     from disk, so this catches durability / scoping / run-id regressions a
     same-session check cannot.

Crucial invariant (from the read_turn bug we hit): every index row must point at
REAL persisted history. So we seed by RUNNING a real turn (StepPersistence writes
the transcript, TurnIndex indexes it) -- never by faking an index row that has no
underlying store record, or `read_turn` correctly reports "No persisted history".

Marked `live` -> excluded from default run (pytest addopts `-m 'not live'`).
Invoke with `pytest -m live` (needs OPENROUTER_API_KEY / OPENAI_API_KEY; load
.env). Unique cid per run + temp-dir cleanup keeps it replayable.
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import pytest
from pydantic_ai.messages import FunctionToolCallEvent
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence

from paf.capabilities.turn_index import TurnIndexCapability, TurnIndexStore

pytestmark = pytest.mark.live

PASSPHRASE = "zephyr-quartz-4471"


def _build_agent(*, store: FileStepStore, index_dir: Path, max_lines: int = 4):
    """Agent wired like production (StepPersistence + TurnIndex pair), low max_lines.

    Rebuilt fresh per call over the same store+index => cold-start read.
    """
    from pydantic_ai import Agent

    return Agent(
        "openrouter:deepseek/deepseek-v4-flash-0731:nitro",
        name="tier2-live",
        capabilities=[
            StepPersistence(store=store),
            TurnIndexCapability(
                source=SnapshotHistorySource(store),
                index_store=TurnIndexStore(index_dir),
                max_lines=max_lines,
            ),
        ],
    )


def _run_turn(agent, prompt: str, cid: str) -> tuple[list[str], str]:
    """Run one turn via run_stream_events; return (tools_invoked, text)."""
    import asyncio

    tool_calls: list[str] = []
    text_parts: list[str] = []

    async def run():
        async with agent.run_stream_events(prompt, conversation_id=cid) as es:
            async for event in es:
                if isinstance(event, FunctionToolCallEvent):
                    tool_calls.append(str(event.part.tool_name or ""))
                elif hasattr(event, "part") and event.part is not None:
                    from pydantic_ai.messages import TextPart

                    if isinstance(event.part, TextPart):
                        text_parts.append(event.part.content)

    asyncio.run(run())
    return tool_calls, "".join(text_parts)


def test_live_recall_via_read_turn() -> None:
    """A real, persisted fact survives a cold start and is recalled via read_turn.

    Geometry: seed the passphrase in a REAL turn (so it's in persisted history
    AND indexed under the SAME run_id -- the run-id invariant). A fresh agent over
    the same store/cid must (a) recognize it needs the retrieval tool, and
    (b) surface the exact passphrase. We gate on read_turn being invoked.
    """
    cid = f"live-recall-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        store = FileStepStore(directory=base / "steps")
        index_dir = base / "index"

        # SEED with a real model turn that plants the passphrase in the transcript.
        seed_agent = _build_agent(store=store, index_dir=index_dir)
        seed_tools, seed_text = _run_turn(
            seed_agent,
            f"Remember this for later: the encrypted volume passphrase is {PASSPHRASE}. "
            "Do not do anything else, just acknowledge.",
            cid,
        )
        # Sanity: seeding actually produced an index row referencing real history.
        assert TurnIndexStore(index_dir).count(conversation_id=cid) >= 1, (
            "seed turn was not indexed"
        )

        # COLD-START recall in a FRESH agent over the same store + cid.
        recall_agent = _build_agent(store=store, index_dir=index_dir)
        tools, text = _run_turn(
            recall_agent,
            "From your turn index / history (you may need a tool), what was the "
            "encrypted volume passphrase? Reply with just the passphrase.",
            cid,
        )

        # 1. The model must actually invoke a retrieval tool (grounding check).
        assert any(t in tools for t in ("read_turn", "search_index")), (
            f"model did not invoke a retrieval tool; tools={tools!r} text={text!r}"
        )
        # 2. The value re-covered from persistence surfaces in the answer.
        assert PASSPHRASE in text, f"answer missing passphrase; tools={tools!r} text={text!r}"