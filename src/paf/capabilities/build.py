"""Runtime capability builders — the ``capabilities=`` seam for Agent.from_spec.

Spec files (`specs/*.yaml`) go through pydantic-ai's ``AgentSpec``/``Agent.from_spec``,
which resolves each top-level ``capabilities:`` entry via that class's ``from_spec()`` and
— crucially — only ONE level deep (see ``_capabilities_from_spec`` in pydantic-ai's
``agent/__init__.py``). Capabilities that can't be expressed that way — because a class
doesn't override ``from_spec`` or needs real objects as arguments — are assembled here and
injected via the ``capabilities=`` kwarg, which ``Agent.from_spec`` MERGES with any
spec-declared capabilities.

**The profile is the gatekeeper.** ``build_runtime_capabilities(profile)`` builds exactly
the capabilities the profile's ``capability_names`` allowlists (resolved through
:data:`paf.capabilities.persistence.PERSISTENCE_BUILDERS`) sharing ONE store. We no longer
declare capabilities in YAML, so this Python ``capabilities=`` seam is the ONLY thing that
reaches the agent — which means the profile's allowlist is what decides "what this agent
gets." No hardcoded list here; it is the resolution of the profile.

With turn-index ON (the M2 summary layer), the builtin compaction stack is REMOVED — the two
are mutually exclusive summary layers (see vault ``Turn-Summary-Index.md`` → "Comparison with
compaction"). So compaction has no builder in this seam. A profile simply omits it (or any
capability) from its ``capability_names`` and that agent won't get it.
"""

from __future__ import annotations

from paf.capabilities.persistence import _build_store, PERSISTENCE_BUILDERS
from paf.capabilities.profiles import AgentProfile


def build_runtime_capabilities(profile: AgentProfile) -> list:
    """Build the capabilities the profile allowlists, over one shared store.

    Selects a builder for each of ``profile.capability_names`` that has one in
    :data:`PERSISTENCE_BUILDERS`, handing each the SAME store (so write /
    search / read_turn provably resolve the same history). A profile may list
    fewer — e.g. a minimal agent with only ``StepPersistence`` — and that is
    exactly what it gets. Capability names without a Python builder are left
    for the spec-declared (YAML) path.
    """
    store = _build_store()
    return [
        builder(store)
        for name, builder in PERSISTENCE_BUILDERS.items()
        if name in profile.capability_names
    ]
