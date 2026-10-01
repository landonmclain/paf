"""Agent profiles — named capability allowlists resolved to ``custom_capability_types``.

A profile maps to an agent kind, not a spec file. It is the safety boundary
separating policy (which capability types an agent of this kind may enable) from
the declarative spec (what one agent instance is configured with). The loader
resolves a profile to the ``custom_capability_types`` it passes to
``Agent.from_file``; the spec YAML then selects concrete capability instances
from that allowlisted set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from paf.capabilities.registry import CAPABILITY_TYPES


@dataclass(frozen=True, slots=True)
class AgentProfile:
    """A named allowlist of capability types for one agent kind."""

    name: str
    capability_names: tuple[str, ...]

    # Lazily build; frozen dataclass can't set in __post_init__, so cache the
    # resolved tuple via a class-level memo or rebuild each call. The catalog is
    # tiny, so rebuild-per-call is fine and keeps the type honest.
    def capability_types(self) -> tuple[type[Any], ...]:
        """Resolve allowlisted names to capability type objects.

        Raises ``KeyError`` naming the offending entry if a profile references a
        capability not in the catalog — fail loudly at build time, not at run.

        Capabilities that opt out of spec serialization (``get_serialization_name()``
        returns ``None``, e.g. ``Subagents``) are excluded: ``Agent.from_spec``
        requires every type in ``custom_capability_types`` to be spec-serializable,
        and a serialization-opted-out capability can only be supplied as a live
        object via ``build_runtime_capabilities`` (gated by ``capability_names``).
        """
        missing = [n for n in self.capability_names if n not in CAPABILITY_TYPES]
        if missing:
            raise KeyError(
                f"Profile {self.name!r} references capabilities not in the catalog: "
                f"{missing}. Known: {sorted(CAPABILITY_TYPES)}"
            )
        types = (CAPABILITY_TYPES[name] for name in self.capability_names)
        return tuple(t for t in types if t.get_serialization_name() is not None)


ASSISTANT_PROFILE = AgentProfile(
    name="assistant",
    capability_names=(
        "StepPersistence",
        "ConversationSearch",
        "Memory",
        "Compaction",
        "FileSystem",
        "Shell",
        "Git",
        "CodebaseMap",
        "Vault",
        "Workspace",
        "Web",
        "Visual",
        "Browser",
        "Skills",
        "ToolOutputLimits",
        "WarnOnCacheBusts",
        "Subagents",
        "Planning",
        "ThreadSummarizer",
        # "Secrets",
        # "Guardrails",
        # "ToolGating",
    ),
)


# The demo/live-testing profile — the "EVERYTHING" sandbox where new/experimental
# capabilities are tried before being promoted to the production assistant.
DEMO_PROFILE = AgentProfile(
    name="demo",
    capability_names=(
        "StepPersistence",
        "ConversationSearch",
        "Memory",
        "Compaction",
        "FileSystem",
        "Shell",
        "Git",
        "CodebaseMap",
        "Vault",
        "Workspace",
        "Web",
        "Visual",
        "Browser",
        "Skills",
        "ToolOutputLimits",
        "WarnOnCacheBusts",
        "Subagents",
        "Planning",
        "ThreadSummarizer",
        # "Secrets",
        # "Guardrails",
        # "ToolGating",
        # "MCP",  # experimental — enabled in the DEMO sandbox, absent from assistant
    ),
)


# The ContextBench benchmark-runner profile — the codebase-map × benchmark A/B
# harness agent. Deliberately NOT the everything-sandbox: no Web/Browser/Vault/
# Visual/Skills/Subagents/Planning — the benchmark agent should not wander off to
# unrelated capability surfaces. It gets the codebrowser + the shell/git/
# filesystem it needs to inspect and patch a snapshot.
CONTEXTBENCH_PROFILE = AgentProfile(
    name="contextbench",
    capability_names=(
        # "CodebaseMap",
        "Shell",
        "FileSystem",
        "Git",
        "Workspace",
    ),
)


# The LongMemEval benchmark-runner profile — the memory × benchmark harness
# agent. Deliberately NOT the everything-sandbox: no Web/Browser/Vault/Visual/
# Skills/Subagents/Planning — the benchmark agent should not wander off to
# unrelated capability surfaces. It gets exactly the three capabilities the
# memory pipeline needs: persistence, conversation search, and memory. No
# Compaction — the subject is cold-started per question (question + injected
# notebook + on-demand read_turn), so there is no message history to compact.
#
# NOTE: the LongMemEval subject is NOT built via build_agent with this profile —
# the test harness (tests/framework.py) builds it manually, because it needs a
# per-run store and a per-question memory dir that build_agent can't inject. This
# profile is registered in PROFILES for completeness (asserted in
# test_capabilities.py) but its allowlist does not gate the subject's actual
# build. Comment out Memory for the search-only (no-memory) arm.
LONGMEMEVAL_PROFILE = AgentProfile(
    name="longmemeval",
    capability_names=(
        "StepPersistence",
        "ConversationSearch",
        "Memory",
    ),
)


# Name lookup so a spec/CLI can reference a profile by a string that resolves
# to the allowlist. Only used by spec-driven loaders; keep small.
# Tool profile for the `researcher` SUBAGENT leaf (specs/sub_agents/
# researcher_subagent.yaml), not a subject agent: web search/fetch plus the
# scoped filesystem (root defaults to the parent's). Leaf builds always strip
# persistence/recall/memory/summarizer/subagents regardless of the profile.
RESEARCHER_PROFILE = AgentProfile(
    name="researcher",
    capability_names=(
        "Web",
        "FileSystem",
        "Workspace",
        "ToolOutputLimits",
    ),
)


PROFILES: dict[str, AgentProfile] = {
    profile.name: profile
    for profile in (
        ASSISTANT_PROFILE,
        DEMO_PROFILE,
        CONTEXTBENCH_PROFILE,
        LONGMEMEVAL_PROFILE,
    )
}

# Tool profiles for SUBAGENT leaves (a leaf spec's `profile:`), kept apart from
# PROFILES — they are never selectable as a subject agent's profile. Leaf
# profile lookup checks here first, then PROFILES.
SUBAGENT_PROFILES: dict[str, AgentProfile] = {
    profile.name: profile for profile in (RESEARCHER_PROFILE,)
}
