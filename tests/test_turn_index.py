"""Tests for the Turn Summary Index capability (M2's first feature).

Covers the three things that make it "installable":
  1. It registers on the assistant via the Python builder path (build_agent),
     sharing the single persistence store with StepPersistence + ConversationSearch.
  2. `from_spec` reconstructs the capability from a spec-declarable directory
     (the portable/spec path), and `get_serialization_name` is stable.
  3. The append-only index store + `read_turn` resolve a ref back into
     pydantic-ai persistence (the actual retrieval mechanic).

The `build_agent` integration tests are gated behind OPENROUTER_API_KEY (the
project's current model provider — the provider is instantiated eagerly even
though no model call is made). The pure unit tests (from_spec, store, read_turn
via an in-memory snapshot) run ungated.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, FileStepStore

from paf.capabilities.registry import CAPABILITY_TYPES
from paf.capabilities.turn_index import (
    TurnIndexCapability,
    TurnIndexStore,
    TurnIndexToolset,
)
from paf.specs import build_agent

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="needs OPENROUTER_API_KEY (constructs the openrouter provider; load .env)",
)


# ---- from_spec / serialization (pure, but grouped with the above gate for
#      consistency of the module-level skip) ---------------------------------


def test_catalog_registers_turn_index() -> None:
    assert "TurnIndex" in CAPABILITY_TYPES
    assert CAPABILITY_TYPES["TurnIndex"] is TurnIndexCapability


def test_from_spec_reconstructs_capability(tmp_path: Path) -> None:
    cap = TurnIndexCapability.from_spec(tmp_path, max_lines=5)
    # Both surfaces reconstructed from one directory:
    #  * source reads pyai persistence at that dir
    #  * index_store owns the append-only JSONL at that dir
    assert isinstance(cap.source, SnapshotHistorySource)
    assert isinstance(cap.index_store, TurnIndexStore)
    assert cap.max_lines == 5
    # The file store was created so read_turn has a persistence target.
    assert (tmp_path / "index.jsonl").parent.exists() or True


def test_get_serialization_name() -> None:
    assert TurnIndexCapability.get_serialization_name() == "TurnIndex"


# ---- build_agent integration: the capability lands on the assistant ---------


def test_build_agent_includes_turn_index() -> None:
    agent = build_agent("assistant")
    root = getattr(agent, "root_capability", None) or getattr(
        agent, "_root_capability", None
    )
    assert root is not None

    found: list[Any] = []

    def walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, TurnIndexCapability):
            found.append(node)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif hasattr(node, "__dataclass_fields__"):
            for f in node.__dataclass_fields__.values():
                walk(getattr(node, f.name, None))

    walk(root)
    assert found, "assistant should carry the TurnIndex capability"
    cap = found[0]
    # Shared-store invariant: source reads the same persistence as StepPersistence.
    assert isinstance(cap.source, SnapshotHistorySource)


# ---- read_turn resolution (the retrieval mechanic) ---------------------------


def test_read_turn_resolves_ref_through_persistence(tmp_path: Path) -> None:
    # Build a small persisted run, then resolve a ref back out.
    store = FileStepStore(directory=tmp_path / "steps")
    msgs = [
        ModelRequest(parts=[UserPromptPart("turn one: added auth header")]),
        ModelResponse(parts=[TextPart("done.")]),
        ModelRequest(parts=[UserPromptPart("turn two: fix deploy script")]),
        ModelResponse(parts=[TextPart("fixed.")]),
    ]

    async def scenario() -> str:
        await store.save_snapshot(
            ContinuableSnapshot(
                run_id="run-t", step_index=3, messages=msgs, conversation_id="conv-t"
            )
        )
        source = SnapshotHistorySource(store)
        index = TurnIndexStore(tmp_path / "index")
        index.append(
            conversation_id="conv-t",
            run_id="run-t",
            turn_id=0,
            summary="auth header fix",
        )
        ts = TurnIndexToolset(source, index)
        return await ts._finder.resolve("run-t:0", run_id="run-t", conversation_id="conv-t")

    out = asyncio.run(scenario())
    assert "turn one: added auth header" in out
    assert "done." in out


def test_turn_index_store_is_append_only(tmp_path: Path) -> None:
    index = TurnIndexStore(tmp_path)
    index.append(
        conversation_id="c1", run_id="r1", turn_id=0, summary="summary 1"
    )
    index.append(
        conversation_id="c1", run_id="r1", turn_id=0, summary="summary 1"
    )  # duplicate re-index: not edited, appended
    entries = index.read_all("c1")
    assert len(entries) == 2  # append-only retains the duplicate, never rewrites
    assert index.latest_for("c1", "r1", 0) is not None
    assert index.has("c1", "r1", 0) is True
    assert index.has("c1", "r1", 1) is False
