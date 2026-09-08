"""Tests for the ThreadSummarizer capability shell (Phase 0).

Verifies the capability registers in the catalog with a serialization name, can
be built via ``from_spec``, and is OFF by default for the assistant profile
(the profile allowlist is the ON/OFF switch — see OFF-switch correctness).
"""

from __future__ import annotations

from pydantic_ai.capabilities import AbstractCapability

from paf.capabilities import ASSISTANT_PROFILE
from paf.capabilities.registry import CAPABILITY_TYPES
from paf.capabilities.thread_summarizer import ThreadSummarizer


def test_thread_summarizer_is_registered_in_catalog() -> None:
    """The catalog keyed by the spec's CamelCase name resolves the type."""
    assert "ThreadSummarizer" in CAPABILITY_TYPES
    assert CAPABILITY_TYPES["ThreadSummarizer"] is ThreadSummarizer


def test_thread_summarizer_is_spec_serializable() -> None:
    """A non-None serialization name means it can be spec-declared (like Memory)."""
    assert ThreadSummarizer.get_serialization_name() == "ThreadSummarizer"


def test_from_spec_builds_instance() -> None:
    """from_spec returns a real ThreadSummarizer instance."""
    cap = ThreadSummarizer.from_spec()
    assert isinstance(cap, ThreadSummarizer)
    assert isinstance(cap, AbstractCapability)


def test_thread_summarizer_is_abstract_capability() -> None:
    """It is a first-class pydantic-ai abstract capability."""
    assert issubclass(ThreadSummarizer, AbstractCapability)


def test_off_by_default_for_assistant_profile() -> None:
    """ThreadSummarizer is NOT allowlisted by default — the OFF switch.

    A profile that omits it simply never builds it (build_runtime_capabilities
    skips any name not in capability_names).
    """
    assert "ThreadSummarizer" not in ASSISTANT_PROFILE.capability_names


def test_not_required_by_profile_resolution() -> None:
    """Resolving the assistant profile's types succeeds and excludes it."""
    # Should not raise KeyError (no dead refs) and not include ThreadSummarizer.
    resolved = ASSISTANT_PROFILE.capability_types()
    names = {t.get_serialization_name() for t in resolved}
    assert "ThreadSummarizer" not in names


def test_built_when_profile_allowlists_it() -> None:
    """build_runtime_capabilities builds it iff the profile allowlists it."""
    from paf.capabilities import AgentProfile
    from paf.capabilities.build import build_runtime_capabilities

    # A profile that opts the capability IN.
    on = AgentProfile(
        name="thread-on",
        capability_names=(
            "StepPersistence",
            "ConversationSearch",
            "Memory",
            "ThreadSummarizer",
        ),
    )
    caps = build_runtime_capabilities(on)
    assert any(type(c).__name__ == "ThreadSummarizer" for c in caps)


def test_off_profile_never_builds_it() -> None:
    """OFF is total at build time: a profile omitting it yields no instance."""
    from paf.capabilities.build import build_runtime_capabilities

    caps = build_runtime_capabilities(ASSISTANT_PROFILE)
    assert all(type(c).__name__ != "ThreadSummarizer" for c in caps)
