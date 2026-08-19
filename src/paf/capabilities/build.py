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

from pathlib import Path

from paf.capabilities.compaction import build_compaction
from paf.capabilities.memory import Memory
from paf.capabilities.persistence import PERSISTENCE_BUILDERS, _build_store
from paf.capabilities.profiles import AgentProfile
from paf.capabilities.search import ConversationSearch, SnapshotHistorySource
from paf.capabilities.turn_index import build_turn_index
from paf.config import PROJECT_ROOT, settings

# The profile name that gates context compaction in build_runtime_capabilities.
_COMPACTION_CAPABILITY = "Compaction"

# Where the Memory capability writes its readable `.md` bodies + `.memory-state.json`
# CAS sidecar, anchored to the repo root (same convention as persistence/_STORE_DIR).
# Directory comes from the shared Settings object (MEMORY_DIR / .env), so the
# consolidation owner and the assistant always target the same store.
_MEMORY_DIR = PROJECT_ROOT / Path(settings.memory_dir)


def _build_memory(store) -> Memory:
    """M2 journal-view durable memory, INJECT-ONLY for the assistant.

    Mirrors the benchmark assembly in `tests/framework.py`: the model carries
    the rendered journal (MEMORY.md) injected into every request via
    before_model_request, and is exposed ZERO memory tools (expose_tools=False)
    -- not even read/search -- because a separate consolidation owner sub-agent
    is the sole writer. This converges prod's memory surface on the M2 contract
    (JournalViewStore, read-only, inject-only), the same shape the LongMemEval
    harness measures.

    `store` (the shared persistence store) is accepted for a uniform builder
    signature but unused. inject_memory=True, writable=False, and an empty
    guidance string mirror _build_agent; max_lines=1_000_000 / max_tokens=65_536
    are high, non-evicting caps so the full rendered journal always shows.
    """
    from paf.capabilities.memory._view import JournalViewStore

    return Memory(
        store=JournalViewStore(_MEMORY_DIR),
        inject_memory=True,
        writable=False,
        expose_tools=False,
        max_lines=1_000_000,
        max_tokens=65_536,
        guidance="",
    )


def _conversation_search_suppressed(store):
    """ConversationSearch with its own instruction scrap disabled.

    Used when TurnIndex OR Memory owns the composite recall ladder: TurnIndex's
    `get_instructions` (when Memory absent) or Memory's composite `_DEFAULT_GUIDANCE`
    (when present) already name `search_conversation_history`'s slot, so this
    capability's standalone "a search_conversation_history tool can retrieve
    exact details…" fragment would be duplicative and potentially conflicting.
    Same shared store either way — this only toggles instruction emission.
    """
    return ConversationSearch(
        source=SnapshotHistorySource(store),
        scope="conversation",
        add_instructions=False,
    )


def _turn_index_suppressed(store):
    """TurnIndex with its own instruction scrap disabled (Memory owns the ladder).

    When the profile also allowlists ``Memory``, Memory's composite
    ``_DEFAULT_GUIDANCE`` is the single recall ladder (durable-fact + past-turn +
    last-rung history). It already names ``search_index`` / ``read_turn`` /
    ``search_conversation_history``, so TurnIndex's own self-contained ladder would
    be a conflicting duplicate. Same shared store — this only toggles instruction
    emission; the `search_index` / `read_turn` tools stay registered.
    """
    from dataclasses import replace

    from paf.capabilities.turn_index import build_turn_index

    return replace(build_turn_index(store), add_instructions=False)


def build_runtime_capabilities(
    profile: AgentProfile,
    compaction_config=None,
) -> list:
    """Build the capabilities the profile allowlists, over one shared store.

    Persistence + recall select per :data:`PERSISTENCE_BUILDERS`; ``TurnIndex``
    builds itself via :func:`turn_index.build_turn_index`, but all are handed the
    SAME store (so write / search / read_turn provably resolve the same history).
    ``Compaction`` (the no-summary ladder) is appended iff the profile allowlists
    it. A profile may list fewer — e.g. a minimal agent with only
    ``StepPersistence`` — and that is exactly what it gets. Capability names
    without a Python builder are left for the spec-declared (YAML) path.

    **Recall-instruction ownership:** exactly ONE capability emits the composite
    recall ladder. When ``Memory`` is allowlisted it owns it (durable-fact +
    past-turn + last-rung history, in ``_DEFAULT_GUIDANCE``), so ``TurnIndex`` and
    ``ConversationSearch`` get ``add_instructions=False`` here to keep that single
    ladder the ONLY recall guidance the model sees. The current assistant profile
    allowlists ``Memory`` and NOT ``TurnIndex`` (BEAM+gated benchmark: index
    summaries are a ±7/−5 wash vs memory's reliable +0.146 lift), so the
    ``TurnIndex``-owns path below is a fallback only reached by a custom profile
    that re-adds it. When ``Memory`` is absent but ``TurnIndex`` is present,
    ``TurnIndex`` owns it (recognition → search_index → read_turn →
    search_conversation_history), and ``ConversationSearch`` alone is
    suppressed. The suppression gates live here (build time) because capabilities
    cannot see each other; ``add_instructions=False`` never disables tool
    registration.
    """
    # Both builders are keyed by the SAME names a profile allowlists, so the
    # profile is the gatekeeper for each.
    builders = {
        **PERSISTENCE_BUILDERS,
        "TurnIndex": build_turn_index,
        # Memory is its own store (MarkdownStorage), not the shared run store, so
        # its builder ignores the arg — but keeping it here means the profile's
        # `Memory` allowlist entry actually gets built (un-gated, M2 live-test beat).
        "Memory": _build_memory,
    }
    store = _build_store()
    # Exactly ONE capability owns the composite recall ladder:
    #   - Memory present       -> Memory's `_DEFAULT_GUIDANCE` is the ladder (durable
    #     fact + past-turn + last-rung history); TurnIndex and ConversationSearch's
    #     instruction scraps are suppressed so the model sees one ladder, not three.
    #   - Memory absent        -> TurnIndex owns it (recognition -> search_index ->
    #     read_turn -> search_conversation_history); ConversationSearch suppressed.
    # The suppression gates live HERE (build time) because capabilities can't see
    # each other; `add_instructions=False` only toggles instruction TEXT, never the
    # tool registration (search_index / read_turn / search_conversation_history are
    # still attached).
    has_turn_index = "TurnIndex" in profile.capability_names
    has_memory = "Memory" in profile.capability_names
    capabilities = []
    for name, builder in builders.items():
        if name not in profile.capability_names:
            continue
        if name == "ConversationSearch" and (has_turn_index or has_memory):
            capabilities.append(_conversation_search_suppressed(store))
        elif name == "TurnIndex" and has_memory:
            capabilities.append(_turn_index_suppressed(store))
        else:
            capabilities.append(builder(store))
    if _COMPACTION_CAPABILITY in profile.capability_names:
        capabilities.append(build_compaction(compaction_config))
    return capabilities
