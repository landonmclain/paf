"""Agent profiles — named capability allowlists resolved to ``custom_capability_types``.

A profile maps to *an agent kind*, not to a spec file. It is the safety boundary
separating policy (which capability types may an agent of this kind enable) from
the declarative spec (what one agent instance is configured with). The loader
resolves a profile to the ``custom_capability_types`` it passes to
``Agent.from_file``; the spec YAML then selects concrete capability instances from
that allowlisted set.

M1 has exactly one real profile: **assistant** (allowlists ``StepPersistence``,
M1's core storage point). There is no ``summary`` profile — ``summary.yaml`` is a
*model recipe* (model / instructions / settings) for the compaction capability,
not an agent, so it has nothing to whitelist. Per the M1 plan, do not scaffold a
blank summary profile until summary earns a capability of its own (YAGNI).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .registry import CAPABILITY_TYPES


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


# The single real M1 profile. The assistant may enable persistence (and later,
# per the M1 roadmap, compaction + ConversationSearch as those land in its spec).
ASSISTANT_PROFILE = AgentProfile(
    name="assistant",
    capability_names=("StepPersistence",),
)


# Name lookup so a spec/CLI can reference a profile by a string that resolves
# to the allowlist. Only used by spec-driven loaders; keep small.
PROFILES: dict[str, AgentProfile] = {
    profile.name: profile for profile in (ASSISTANT_PROFILE,)
}
