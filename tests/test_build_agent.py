"""Integration test: build_agent resolves the hybrid capability merge.

Pins the M2 contract: building the assistant from specs/assistant.yaml yields an
Agent carrying —
  * a spec-declared StepPersistence (resolved from YAML via from_spec),
  * Python-assembled recall + memory capabilities (the `capabilities=` runtime seam),
  * and a no-summary compaction ladder (context management only — the non-summary
    tiers clamp / clear-tool / sliding-window; long-term recall lives in the
    store-backed memory system, so there is no summarized-history tier).

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
from pydantic_ai_harness.step_persistence import StepPersistence

from paf.capabilities.memory import Memory
from paf.capabilities.search import ConversationSearch
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
    if isinstance(node, (StepPersistence, TieredCompaction, Memory)):
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


def _find_compaction(node: Any) -> TieredCompaction | None:
    """Locate the (single) TieredCompaction anywhere in a composed root capability."""
    if node is None:
        return None
    if isinstance(node, TieredCompaction):
        return node
    if isinstance(node, (list, tuple)):
        for item in node:
            found = _find_compaction(item)
            if found is not None:
                return found
        return None
    if isinstance(node, dict):
        for value in node.values():
            found = _find_compaction(value)
            if found is not None:
                return found
        return None
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        for field in dataclasses.fields(node):
            found = _find_compaction(getattr(node, field.name, None))
            if found is not None:
                return found
    return None


def test_build_agent_merges_spec_and_runtime_capabilities() -> None:
    agent = build_agent("assistant")
    root = getattr(agent, "root_capability", None) or getattr(
        agent, "_root_capability", None
    )
    assert root is not None, "agent exposes a root capability to inspect"

    caps = _collect_capabilities(root)

    # Persistence leg (spec-declared) + memory + recall legs (Python runtime seam) +
    # the no-summary compaction ladder (profile-gated "Compaction").
    assert StepPersistence in caps
    assert Memory in caps
    assert TieredCompaction in caps


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
    # A profile allowlisting persistence + recall + memory + compaction gets
    # exactly those (the no-summary ladder). Everything the profile lists is present.
    from paf.capabilities import AgentProfile

    profile = AgentProfile(
        name="merge-on",
        capability_names=(
            "StepPersistence",
            "ConversationSearch",
            "Memory",
            "Compaction",
        ),
    )
    assistant = _runtime_capabilities_for(profile.capability_names)
    types = {type(c) for c in assistant}
    assert StepPersistence in types
    assert ConversationSearch in types
    assert Memory in types
    assert TieredCompaction in types

    # A minimal profile with only StepPersistence must get ONLY StepPersistence.
    minimal = _runtime_capabilities_for(("StepPersistence",))
    assert {type(c) for c in minimal} == {StepPersistence}

    # A profile that allows persistence + recall gets both, none of the others.
    no_index = _runtime_capabilities_for(("StepPersistence", "ConversationSearch"))
    assert {type(c) for c in no_index} == {StepPersistence, ConversationSearch}

    # A profile WITHOUT "Compaction" gets no compaction at all.
    no_compaction = _runtime_capabilities_for(("StepPersistence", "ConversationSearch"))
    assert TieredCompaction not in {type(c) for c in no_compaction}


def _build_recall_agent():
    """Drive tests.framework._build_agent (the recall-agent builder).

    Returns the composed root capability.
    """
    import tests.framework as _rl

    old_mem = _rl.MEMORY
    _rl.MEMORY = True  # the memory arm only exists under MEMORY=1
    try:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from pydantic_ai_harness.step_persistence import FileStepStore

        with TemporaryDirectory() as td:
            store = FileStepStore(directory=Path(td) / "store")
            agent = _rl._build_agent(
                store,
                memory_dir=Path(td) / "memory",
            )
            return getattr(agent, "_root_capability", None) or getattr(
                agent, "root_capability", None
            )
    finally:
        _rl.MEMORY = old_mem


def test_recall_agent_has_no_compaction() -> None:
    """The recall agent has NO compaction.

    The subject is cold-started per question (question + injected notebook +
    on-demand read_turn) and never receives message history in context, so there
    is nothing to compact — no TieredCompaction ladder is wired.
    """
    root = _build_recall_agent()
    comp = _find_compaction(root)
    assert comp is None
