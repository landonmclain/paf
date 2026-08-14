"""RECALL CONTRACT: prove the full write->index->read loop recalls the SAME turn.

This is a recall tool. The unit tests elsewhere hand-pin a single run_id onto both
the store snapshot and the index entry, so they can't catch the run-id namespace
divergence that actually broke `read_turn` (StepPersistence built WITH agent_name
keys the store by `assistant-<hex>` while TurnIndex.after_model_request records
pydantic-ai's `ctx.run_id` ``019fe…``).

This test drives a REAL `Agent.run()` with a deterministic `FunctionModel` (no API
key), through the SAME builder wiring production uses:
    StepPersistence(store) + TurnIndexCapability(source=SnapshotHistorySource(store), index_store)
so the framework genuinely generates `ctx.run_id` ONCE, and every layer (the store
write, the index ref) must key by THAT id — or the recall fails. We never supply
the run_id ourselves; it is opaque and framework-generated.

  * If store and index diverge again, `read_turn(ref)` returns
    "No persisted history for run ..." and the asserted content is absent.
  * If `read_turn` can't resolve a user-prompt anchor to a turn, the recalled
    body won't contain the seeded text.

Run ungated (FunctionModel needs no provider key).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence

from paf.capabilities.search._reader import TurnFinder
from paf.capabilities.turn_index import TurnIndexCapability, TurnIndexStore

SEED = "staging ftp host is sftp-mars.example.com"
CONV = "conv-recall"


def _echo_model() -> FunctionModel:
    """Deterministic model. FunctionDef signature: (messages, info) -> ModelResponse."""
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.models.function import AgentInfo

    def handler(messages: list, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(f"ok: {SEED}")])

    return FunctionModel(handler, model_name="echo")


def test_recall_contract_end_to_end() -> None:
    """Real Agent.run persists a turn; read_turn recalls exactly that turn."""

    async def scenario(tmp: Path) -> list[str]:
        store = FileStepStore(directory=tmp / "steps")
        index_dir = tmp / "index"
        # Use the ACTUAL production builders so a regression in persistence.py
        # (e.g. re-adding agent_name) fails this test — wiring it inline would
        # only proxy the wiring, not guard the shipped code.
        from paf.capabilities.persistence import _build_step_persistence

        agent = Agent(
            model=_echo_model(),
            name="paf-recall",
            capabilities=[
                _build_step_persistence(store),
                TurnIndexCapability(
                    source=SnapshotHistorySource(store),
                    index_store=TurnIndexStore(index_dir),
                ),
            ],
        )

        # Drive one real turn. Only `conversation_id` matters for scoping; run_id
        # is framework-generated and opaque to us — we must NOT pin it.
        await agent.run(SEED, conversation_id=CONV)

        # 1. The store write: find the run_id StepPersistence actually persisted.
        store_run_ids = await store.list_runs()
        if not store_run_ids:
            return ["StepPersistence wrote nothing to the store"]
        store_run_id = store_run_ids[0].run_id

        # 2. The index write: find the run_id TurnIndex.after_model_request recorded.
        index_store = TurnIndexStore(index_dir)
        entries = index_store.read_all(conversation_id=CONV)
        if not entries:
            return ["index empty — the after_model_request hook didn't record a turn"]
        index_run_id = entries[0].run_id

        problems: list[str] = []
        if store_run_id != index_run_id:
            problems.append(
                f"run-id mismatch: store keys {store_run_id!r}, index records {index_run_id!r}"
            )

        # 3. The recall: resolve the index's ref through read_turn. read_turn is
        #    owned by `search` (TurnFinder); the index only emits refs into it.
        #    If store and index already diverge, or the anchor resolution fails,
        #    the seeded content will not be reproduced and we flag it here.
        finder = TurnFinder(SnapshotHistorySource(store))
        recalled = await finder.resolve(
            f"{index_run_id}:0", run_id=index_run_id, conversation_id=CONV
        )
        if "No persisted history" in recalled:
            problems.append(f"read_turn couldn't resolve the stored ref at all: {recalled[:200]!r}")
        elif SEED not in recalled:
            problems.append(
                f"read_turn recalled a turn MISSING the seeded content:\n{recalled[:400]!r}"
            )
        return problems

    with tempfile.TemporaryDirectory() as td:
        problems = asyncio.run(scenario(Path(td)))

    assert not problems, "RECALL BROKEN:\n" + "\n".join(f"  - {p}" for p in problems)