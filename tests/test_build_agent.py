"""Integration test: build_agent resolves the hybrid capability merge.

Pins the M2 contract: building the assistant from specs/assistant.yaml (with the
TurnIndex profile) yields an Agent carrying —
  * a spec-declared StepPersistence (resolved from YAML via from_spec),
  * a Python-assembled TurnIndexCapability (the `capabilities=` runtime seam),
  * and NO builtin compaction — turn-index IS the summary layer; turn-index and
    SummarizingCompaction are mutually exclusive (see build.py + vault).

This only *builds* the agent — it makes no model call — but constructing an
`openai:` Agent requires `OPENAI_API_KEY` to be present (the provider is
instantiated eagerly, even though no request is made). So this file is gated
behind that env var: bare `pytest` (no `.env`) skips it; run with the key
present (e.g. `uv run --env-file .env pytest`) to actually assert the merge.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import pytest
from pydantic_ai_harness.compaction import TieredCompaction
from pydantic_ai_harness.conversation_search import ConversationSearch
from pydantic_ai_harness.step_persistence import StepPersistence

from paf.capabilities.profiles import ASSISTANT_PROFILE
from paf.capabilities.turn_index import TurnIndexCapability
from paf.specs import build_agent

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="needs OPENAI_API_KEY (constructs the openai provider; load .env)",
)


def _collect_capabilities(node: Any, seen: set[type] | None = None) -> set[type]:
    """Recursively collect capability classes reachable from a (possibly composed)
    root capability. Anticipates that `root_capability` may wrap or list children."""
    seen = seen if seen is not None else set()
    if node is None:
        return seen
    if isinstance(node, (StepPersistence, TieredCompaction, TurnIndexCapability)):
        seen.add(type(node))
        return seen
    if isinstance(node, (list, tuple)):
        for item in node:
            _collect_capabilities(item, seen)
        return seen
    if isinstance(node, dict):
        for value in node.values():
            _collect_capabilities(value, seen)
        return seen
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        for field in dataclasses.fields(node):
            _collect_capabilities(getattr(node, field.name, None), seen)
        return seen
    return seen


def test_build_agent_merges_spec_and_runtime_capabilities() -> None:
    agent = build_agent("assistant")
    root = getattr(agent, "root_capability", None) or getattr(
        agent, "_root_capability", None
    )
    assert root is not None, "agent exposes a root capability to inspect"

    caps = _collect_capabilities(root)

    # Persistence leg (spec-declared) + turn-index leg (Python runtime seam).
    assert StepPersistence in caps
    assert TurnIndexCapability in caps
    # Builtin compaction is REMOVED — turn-index is the sole summary layer.
    assert TieredCompaction not in caps


def test_build_agent_excludes_compaction_when_turn_index() -> None:
    """Turn-index and SummarizingCompaction are mutually exclusive summary layers.

    With the TurnIndex profile, no TieredCompaction / compacting tier may be wired
    into the agent — otherwise the model gets BOTH the <turn-index> one-liners and
    a `Summary of previous conversation` in the same request (redundant/confusing).
    """
    agent = build_agent("assistant")
    root = getattr(agent, "root_capability", None) or getattr(
        agent, "_root_capability", None
    )
    assert root is not None

    compaction: Any = None

    def find_compaction(node: Any) -> None:
        nonlocal compaction
        if node is None:
            return
        if isinstance(node, TieredCompaction):
            compaction = node
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                find_compaction(item)
        elif isinstance(node, dict):
            for value in node.values():
                find_compaction(value)
        elif dataclasses.is_dataclass(node) and not isinstance(node, type):
            for field in dataclasses.fields(node):
                find_compaction(getattr(node, field.name, None))

    find_compaction(agent._root_capability)
    assert compaction is None


def _runtime_capabilities_for(names: tuple[str, ...]):
    """Build a throwaway profile over `names` and return what build_runtime_capabilities
    constructs for it — the resolution of the profile allowlist, bypassing the model/API."""
    from paf.capabilities.build import build_runtime_capabilities
    from paf.capabilities.profiles import AgentProfile

    return build_runtime_capabilities(AgentProfile(name="tmp", capability_names=names))


def test_profile_is_the_gatekeeper_for_runtime_capabilities() -> None:
    """The profile allowlist decides exactly what reaches the agent via the
    ``capabilities=`` seam (the only seam now that we don't declare capabilities
    in YAML). A minimal profile must NOT receive capabilities it didn't allow.

    This pins the contract that sparked the M2 fix: build_runtime_capabilities()
    used to hardcode a fixed set and ignore the profile — so every agent got
    everything regardless of its allowlist. Now it is the resolution of the profile.
    """
    # The real assistant profile: persistence + recall + turn-index, and NOTHING
    # else (TieredCompaction/compaction is deliberately absent — mutex with index).
    assistant = _runtime_capabilities_for(ASSISTANT_PROFILE.capability_names)
    types = {type(c) for c in assistant}
    assert StepPersistence in types
    assert ConversationSearch in types
    assert TurnIndexCapability in types
    assert TieredCompaction not in types

    # A minimal profile with only StepPersistence must get ONLY StepPersistence.
    minimal = _runtime_capabilities_for(("StepPersistence",))
    assert {type(c) for c in minimal} == {StepPersistence}

    # A profile that allows persistence + recall (but not turn-index) gets both,
    # none of the un-allowed ones.
    no_index = _runtime_capabilities_for(("StepPersistence", "ConversationSearch"))
    assert {type(c) for c in no_index} == {StepPersistence, ConversationSearch}
