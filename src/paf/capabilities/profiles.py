"""Agent profiles — named capability allowlists resolved to ``custom_capability_types``.

A profile maps to *an agent kind*, not to a spec file. It is the safety boundary
separating policy (which capability types may an agent of this kind enable) from
the declarative spec (what one agent instance is configured with). The loader
resolves a profile to the ``custom_capability_types`` it passes to
``Agent.from_file``; the spec YAML then selects concrete capability instances from
that allowlisted set.

The only real profile: **assistant** (allowlists ``StepPersistence``, lossless
recall over that persisted history (``ConversationSearch``), the durable-facts
``Memory`` notebook (recall-ladder owner), and the compaction tier types — M1's
core storage point + recall + context management). There is no ``summary``
profile: compaction's summarizing tier is not an agent, so its
model and prompt are configured in ``compaction.py`` (the summarizer inherits the
assistant's model by default). Per the M1 plan, do not scaffold a blank summary
profile until summary earns a capability of its own (YAGNI).
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
    # resolved tuple via a class-level memo or rebuild each call. For M1 the
    # catalog is tiny, so rebuild-per-call is fine and keeps the type honest.
    def capability_types(self) -> tuple[type[Any], ...]:
        """Resolve allowlisted names to capability type objects.

        Raises ``KeyError`` naming the offending entry if a profile references a
        capability not in the catalog — fail loudly at build time, not at run.
        """
        missing = [n for n in self.capability_names if n not in CAPABILITY_TYPES]
        if missing:
            raise KeyError(
                f"Profile {self.name!r} references capabilities not in the catalog: "
                f"{missing}. Known: {sorted(CAPABILITY_TYPES)}"
            )
        return tuple(CAPABILITY_TYPES[name] for name in self.capability_names)


# The single real M1 profile. The assistant may enable persistence, lossless
# recall of that persisted history (ConversationSearch), and context compaction
# (all harness capabilities).
ASSISTANT_PROFILE = AgentProfile(
    name="assistant",
    capability_names=(
        "StepPersistence",
        "ConversationSearch",
        # Durable-facts notebook (MEMORY.md + topic files). Recall-ladder OWNER
        # for the assistant: its composite guidance (durable-fact -> past-turn ->
        # history) is the single recall ladder, so TurnIndex is NOT allowlisted
        # here (BEAM+gated benchmark showed the truncation-index summaries are a
        # ±7/−5 wash vs. memory's reliable +0.146 lift → memory + BM25 search is
        # the proven stack). Re-add "TurnIndex" to restore the index rung.
        "Memory",
        # Context compaction WITHOUT the summarizing tier (the LLM rewrite is
        # mutually exclusive with the <turn-index> one-liner block). Clamp /
        # clear-tool-results / sliding-window are pure context management and
        # stay; see compaction.py.
        "Compaction",
    ),
)


# The demo/live-testing profile — deliberately LEAN. Its job is a surface to
# poke at capabilities in the CLI (ui/cli2.py <agent>) WITHOUT dragging in the
# assistant's persistence/recall/compaction stack. Capabilities that resolve
# natively from a spec's YAML `capabilities:` (WebSearch, WebFetch, XSearch,
# Thinking, MCP, ToolSearch, ImageGeneration — pydantic-ai's own CAPABILITY_TYPES)
# do NOT belong here: they need no builder and no allowlist row. This allowlist
# should only grow with harness capabilities that need a Python build step
# (e.g. ToolOutputLimits, FileSystem, Shell) as we wire them in. Empty for now —
# the first demo pokes are all native, spec-declarable capabilities.
DEMO_PROFILE = AgentProfile(
    name="demo",
    capability_names=(),
)


# Name lookup so a spec/CLI can reference a profile by a string that resolves
# to the allowlist. Only used by spec-driven loaders; keep small.
PROFILES: dict[str, AgentProfile] = {
    profile.name: profile for profile in (ASSISTANT_PROFILE, DEMO_PROFILE)
}
