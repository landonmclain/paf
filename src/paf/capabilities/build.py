"""Runtime capability builders — the ``capabilities=`` seam for Agent.from_spec.

Spec files (`specs/*.yaml`) go through pydantic-ai's ``AgentSpec``/``Agent.from_spec``,
which resolves each top-level ``capabilities:`` entry via that class's ``from_spec()`` and
— crucially — only ONE level deep (see ``_capabilities_from_spec`` in pydantic-ai's
``agent/__init__.py``). Capabilities that can't be expressed that way — because a class
doesn't override ``from_spec`` or needs real objects as arguments — are assembled here and
injected via the ``capabilities=`` kwarg, which ``Agent.from_spec`` MERGES with any
spec-declared capabilities.

**The profile is the gatekeeper.** ``build_runtime_capabilities(profile)`` builds exactly
the capabilities the profile's ``capability_names`` allowlists, over one shared store.
We no longer declare capabilities in YAML, so this Python ``capabilities=`` seam is the
ONLY thing that reaches the agent — which means the profile's allowlist is what decides
"what this agent gets." No hardcoded list here; it is the resolution of the profile.

Persistent capabilities (StepPersistence, ConversationSearch, TurnIndex) select their
builders from :data:`paf.capabilities.persistence.PERSISTENCE_BUILDERS`. Context
compaction builds via ``build_compaction()`` **iff the profile allowlists ``Compaction``**
— the ladder WITHOUT the summarizing tier (the LLM rewrite is mutually exclusive with the
`<turn-index>` one-liner block; see ``compaction.py``). A profile simply omits a name from
its ``capability_names`` and that agent won't get it.
"""

from __future__ import annotations

from pydantic_ai_harness.conversation_search import ConversationSearch, SnapshotHistorySource

from paf.capabilities.compaction import build_compaction
from paf.capabilities.persistence import PERSISTENCE_BUILDERS, _build_store
from paf.capabilities.profiles import AgentProfile
from paf.capabilities.turn_index import build_turn_index

# The profile name that gates context compaction in build_runtime_capabilities.
_COMPACTION_CAPABILITY = "Compaction"


def _conversation_search_suppressed(store):
    """ConversationSearch with its own instruction scrap disabled.

    Used when TurnIndex is co-present: TurnIndex's `get_instructions` emits the
    composite recall ladder that already names search_conversation_history's slot,
    so the harness's standalone "a search_conversation_history tool can retrieve
    exact details…" fragment would be duplicative and potentially conflicting.
    Same shared store either way — this only toggles instruction emission.
    """
    return ConversationSearch(
        source=SnapshotHistorySource(store),
        scope="conversation",
        add_instructions=False,
    )


def build_runtime_capabilities(profile: AgentProfile) -> list:
    """Build the capabilities the profile allowlists, over one shared store.

    Persistence + recall select per :data:`PERSISTENCE_BUILDERS`; ``TurnIndex``
    builds itself via :func:`turn_index.build_turn_index`, but all are handed the
    SAME store (so write / search / read_turn provably resolve the same history).
    ``Compaction`` (the no-summary ladder) is appended iff the profile allowlists
    it. A profile may list fewer — e.g. a minimal agent with only
    ``StepPersistence`` — and that is exactly what it gets. Capability names
    without a Python builder are left for the spec-declared (YAML) path.

    **Recall-instruction ownership:** when the profile allowlists BOTH ``TurnIndex``
    and ``ConversationSearch``, ``TurnIndex`` owns the composite recall ladder
    (recognition → search_index → read_turn → search_conversation_history) in its
    ``get_instructions``, so ``ConversationSearch`` gets ``add_instructions=False``
    here to keep that single ladder the ONLY recall guidance the model sees.
    """
    # Both builders are keyed by the SAME names a profile allowlists, so the
    # profile is the gatekeeper for each.
    builders = {
        **PERSISTENCE_BUILDERS,
        "TurnIndex": build_turn_index,
    }
    store = _build_store()
    # TurnIndex emits the composite recall ladder that already covers
    # search_conversation_history's slot, so when both are present suppress the
    # harness ConversationSearch's own instruction scrap to avoid two conflicting
    # fragments (the model was reproducing them back in registration order).
    has_turn_index = "TurnIndex" in profile.capability_names
    capabilities = []
    for name, builder in builders.items():
        if name not in profile.capability_names:
            continue
        if name == "ConversationSearch" and has_turn_index:
            capabilities.append(_conversation_search_suppressed(store))
        else:
            capabilities.append(builder(store))
    if _COMPACTION_CAPABILITY in profile.capability_names:
        capabilities.append(build_compaction())
    return capabilities
