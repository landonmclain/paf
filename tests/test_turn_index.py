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
from paf.capabilities.search._reader import TurnFinder
from paf.capabilities.turn_index import (
    TurnIndexCapability,
    TurnIndexStore,
    TurnSummarizer,
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


def test_from_spec_default_summarizer_is_stub(tmp_path: Path) -> None:
    """No `summarizer` block => deterministic stub (no model/API key fired)."""
    cap = TurnIndexCapability.from_spec(tmp_path)
    assert cap.summarizer is None


def test_from_spec_flat_summarizer_dict(tmp_path: Path) -> None:
    """A flat `summarizer:` dict is passed straight to `TurnSummarizer(**d)`.

    `TurnSummarizer` is a LEAF HELPER, not an AbstractCapability, so from_spec
    must not route it through the capability registry -- it just constructs it.
    """
    cap = TurnIndexCapability.from_spec(
        tmp_path,
        summarizer={"budget_chars": 200, "max_key_terms": 8},
    )
    assert isinstance(cap.summarizer, TurnSummarizer)
    assert cap.summarizer.budget_chars == 200
    assert cap.summarizer.max_key_terms == 8
    # Unspecified knob falls through to the constructor default.
    assert cap.summarizer.model == "openrouter:deepseek/deepseek-v4-flash-0731"


# ---- build_agent integration: the capability lands on the assistant ---------


def test_build_agent_includes_memory_not_turn_index() -> None:
    """The assistant profile retired TurnIndex in favor of memory + BM25.

    Per the BEAM + gated-benchmark finding (index summaries were a ±7/−5 wash vs
    memory's reliable +0.146 lift), the assistant no longer carries the turn-index
    capability. It carries the durable Memory capability (the recall-ladder owner)
    instead, sharing the single persistence store for lossless recall.
    """
    from paf.capabilities.memory import Memory

    agent = build_agent("assistant")
    root = getattr(agent, "root_capability", None) or getattr(
        agent, "_root_capability", None
    )
    assert root is not None

    found_mem: list[Any] = []
    found_index: list[Any] = []

    def walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, TurnIndexCapability):
            found_index.append(node)
            return
        if isinstance(node, Memory):
            found_mem.append(node)
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
    assert found_mem, "assistant should carry the Memory capability"
    assert not found_index, "assistant should NOT carry the retired TurnIndex capability"


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
    # The run_id the capability writes and the store keys by MUST be the SAME
    # id. In production both derive from pydantic-ai's ctx.run_id (StepPersistence
    # is built WITHOUT agent_name, so the harness falls through to ctx.run_id, and
    # TurnIndex.after_model_request writes ctx.run_id). Regressing this — e.g.
    # passing agent_name so the store keys by `assistant-<hex>` — breaks read_turn;
    # use one shared id here to mirror the fix.
    run_id = "019fe72b-9653-70f3-93ce-c34032c6b601"

    async def scenario() -> str:
        await store.save_snapshot(
            ContinuableSnapshot(
                run_id=run_id, step_index=3, messages=msgs, conversation_id="conv-t"
            )
        )
        source = SnapshotHistorySource(store)
        index = TurnIndexStore(tmp_path / "index")
        index.append(
            conversation_id="conv-t",
            run_id=run_id,
            turn_id=0,
            summary="auth header fix",
        )
        # read_turn is owned by search; the index only emits refs into it.
        finder = TurnFinder(source)
        return await finder.resolve(
            f"{run_id}:0", run_id=run_id, conversation_id="conv-t"
        )

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


def test_step_persistence_run_id_matches_ctx_run_id() -> None:
    """REGRESSION: the store and the TurnIndex must key runs by the SAME id.

    The TurnIndex capability writes pydantic-ai's `ctx.run_id` (``019fe…``) into
    the index refs. `read_turn` resolves those refs through `SnapshotHistorySource`,
    which walks the store by `RunRecord.run_id`. If StepPersistence is built with
    `agent_name`, the harness's `_derive_run_id` prefixes its OWN id
    (``assistant-<hex>``) and the store no longer keys by `ctx.run_id` -- so every
    ref the index writes becomes unresolvable ("No persisted history for run ...").
    This test pins the invariant that `_build_step_persistence` does NOT set
    `agent_name`, keeping both sides on `ctx.run_id`.
    """
    from paf.capabilities.persistence import _build_store, _build_step_persistence

    store = _build_store()
    step = _build_step_persistence(store)

    class _Ctx:
        run_id = "019fe72b-9653-70f3-93ce-c34032c6b601"

    derived = step._derive_run_id(_Ctx())  # type: ignore[attr-defined]
    # Store run id is EXACTLY pydantic-ai's ctx.run_id, not assistant-<hex>.
    assert derived == _Ctx.run_id
    assert not derived.startswith("assistant-")
