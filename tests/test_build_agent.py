"""Integration test: build_agent resolves the hybrid capability merge.

Pins the core M1 contract: building the assistant from specs/assistant.yaml
yields an Agent carrying BOTH —
  * a spec-declared StepPersistence (resolved from YAML via from_spec), and
  * a Python-assembled TieredCompaction (the `capabilities=` runtime seam).
If the spec loader ever stops resolving one leg, this fails.

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
    if isinstance(node, (StepPersistence, TieredCompaction)):
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

    # The two-leg merge: spec-declared persistence + Python-built compaction.
    assert StepPersistence in caps
    assert TieredCompaction in caps


def test_build_agent_tiers_are_objects_not_dicts() -> None:
    """The compaction leg must carry real strategy objects — the regression
    that prompted moving compaction out of YAML into Python."""
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
    assert compaction is not None
    assert len(compaction.tiers) == 4
    # Real strategy objects, not raw dicts (the YAML-declared-tier crash).
    assert all(not isinstance(tier, dict) for tier in compaction.tiers)
