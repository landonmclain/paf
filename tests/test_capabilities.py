"""Tests for the profile↔catalog allowlist integrity and behavior."""

from __future__ import annotations

import pytest
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    SlidingWindowCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.step_persistence import StepPersistence
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.warn_on_cache_busts import WarnOnCacheBusts

from paf.capabilities import ASSISTANT_PROFILE, AgentProfile
from paf.capabilities.compaction import build_compaction
from paf.capabilities.memory._window import MemoryConsolidationWindow
from paf.capabilities.profiles import DEMO_PROFILE, PROFILES
from paf.capabilities.registry import CAPABILITY_TYPES


def test_assistant_profile_names_all_exist_in_catalog() -> None:
    """Every allowlisted name must resolve in the catalog (no dead refs)."""
    missing = [
        name
        for name in ASSISTANT_PROFILE.capability_names
        if name not in CAPABILITY_TYPES
    ]
    assert missing == []


def test_catalog_covers_everything_build_uses() -> None:
    """The catalog is complete enough to resolve the built capabilities.

    build_compaction emits a TieredCompaction whose tier types are all in the
    catalog, and the spec-declared StepPersistence is cataloged too.
    """
    compaction = build_compaction()
    tier_names = {type(tier).__name__ for tier in compaction.tiers}
    assert tier_names <= set(CAPABILITY_TYPES)
    assert "StepPersistence" in CAPABILITY_TYPES
    assert CAPABILITY_TYPES["StepPersistence"] is StepPersistence
    # "Compaction" is the profile-gated name that resolves the no-summary ladder.
    assert "Compaction" in CAPABILITY_TYPES
    assert CAPABILITY_TYPES["Compaction"] is TieredCompaction


def test_capability_types_resolves_and_raises_on_missing() -> None:
    """A valid name resolves to its type; an unknown one fails loudly,
    naming the offending entry (profiles.py contract: fail at build, not run)."""
    profile = AgentProfile(name="probe", capability_names=("StepPersistence",))
    types = profile.capability_types()
    assert types == (StepPersistence,)

    bad = AgentProfile(name="probe", capability_names=("NoSuchCapability",))
    with pytest.raises(KeyError) as excinfo:
        bad.capability_types()
    assert "NoSuchCapability" in str(excinfo.value)


def test_warn_on_cache_busts_cataloged_and_wired() -> None:
    """WarnOnCacheBusts is a cataloged BUILTIN and resolves on the demo profile.

    Same drop-in pattern as ToolOutputLimits: cataloged (so the allowlist + schema
    resolve) and built in Python (so the profile gate controls it with real config
    objects). Observer-only, conservative defaults — no tools/instructions/settings.
    """
    assert CAPABILITY_TYPES["WarnOnCacheBusts"] is WarnOnCacheBusts

    from paf.capabilities.build import build_runtime_capabilities

    built = build_runtime_capabilities(DEMO_PROFILE)
    monitor = next(
        (c for c in built if type(c).__name__ == "WarnOnCacheBusts"), None
    )
    assert monitor is not None
    assert isinstance(monitor, WarnOnCacheBusts)
    assert monitor.collapse_ratio == 0.5
    assert monitor.min_prefix_tokens == 1024
    assert monitor.cache_ttl_seconds == 300.0


def test_subagents_cataloged_and_wired() -> None:
    """Subagents is a cataloged BUILTIN and resolves on the demo profile with child agents.

    Same BUILTIN drop-in pattern as WarnOnCacheBusts, but NOT spec-serializable
    (holds live `Agent` instances; get_serialization_name -> None), so it must be
    built in Python (`_build_subagents`) with explicit child entries. Assert the
    demo build attaches a SubAgents capability whose toolset exposes exactly the
    two named demo leaf agents.
    """
    assert CAPABILITY_TYPES["Subagents"] is SubAgents

    from paf.capabilities.build import build_runtime_capabilities

    built = build_runtime_capabilities(DEMO_PROFILE)
    subagents = next((c for c in built if type(c).__name__ == "SubAgents"), None)
    assert subagents is not None
    assert isinstance(subagents, SubAgents)
    # _by_name is built in __post_init__ from the two explicit SubAgent entries.
    assert set(subagents._by_name) == {"researcher", "writer"}
    assert all(isinstance(sa, SubAgent) for sa in subagents._by_name.values())
    # A live toolset is produced so the delegate_task tool is actually exposed.
    toolset = subagents.get_toolset()
    assert toolset is not None


def test_only_one_real_profile_for_m1() -> None:
    """Two known profiles: assistant (clean prod) and demo (the everything sandbox).

    ``demo_assistant`` is the surface where we build out and tune the full stack
    (persistence, recall, memory, compaction) via the ``demo`` profile, which
    allowlists the same harness capabilities as the production assistant. Once the
    experiment proves out, demo's tuned values promote to the clean ``assistant``
    spec and the sandbox resets. This is the complete, intentional set — no further
    blank profile scaffolding.
    """
    assert set(PROFILES) == {"assistant", "demo"}
    assert PROFILES["assistant"] is ASSISTANT_PROFILE


def test_compaction_builds_real_strategy_objects() -> None:
    """TieredCompaction must hold real strategy objects — the exact regression
    that bit us when tiers were YAML dicts (AttributeError 'model')."""
    compaction = build_compaction()
    assert isinstance(compaction, TieredCompaction)

    # `CompactionStrategy` is a bare Protocol (not runtime_checkable), so we can't
    # isinstance against it. Assert each tier is a real concrete strategy (a class,
    # not a raw dict) drawn from the no-summary ladder's known types. The sliding
    # window may be the plain `SlidingWindowCompaction` or, when consolidation is
    # enabled (Settings.consolidate / CONSOLIDATE=1), its subclass
    # `MemoryConsolidationWindow` — both are real strategy objects.
    strategy_types = (
        ClampOversizedMessages,
        ClearToolResults,
        SlidingWindowCompaction,
        MemoryConsolidationWindow,
    )
    assert len(compaction.tiers) == 3
    assert all(type(tier) in strategy_types for tier in compaction.tiers)
