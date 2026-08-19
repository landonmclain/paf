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

from paf.capabilities import ASSISTANT_PROFILE, AgentProfile
from paf.capabilities.compaction import build_compaction
from paf.capabilities.memory._window import MemoryConsolidationWindow
from paf.capabilities.profiles import PROFILES
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


def test_compaction_excludes_the_summarizing_tier() -> None:
    """The summarizing tier is mutually exclusive with the turn-index one-liner:
    with turn-index ON, compaction is the no-summary ladder (clamp / clear-tool /
    sliding-window) and must NOT emit SummarizingCompaction."""
    compaction = build_compaction()
    from pydantic_ai_harness.compaction import SummarizingCompaction

    assert not any(type(tier) is SummarizingCompaction for tier in compaction.tiers)
