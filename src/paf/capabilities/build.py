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

import os
from pathlib import Path

from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits
from pydantic_ai_harness.warn_on_cache_busts import WarnOnCacheBusts

from paf.capabilities.compaction import build_compaction
from paf.capabilities.fs import FileSystem as _FileSystem
from paf.capabilities.git import Git as _Git
from paf.capabilities.memory import Memory
from paf.capabilities.persistence import PERSISTENCE_BUILDERS, _build_store
from paf.capabilities.profiles import AgentProfile
from paf.capabilities.search import ConversationSearch, SnapshotHistorySource
from paf.capabilities.shell import Shell as _Shell
from paf.capabilities.turn_index import build_turn_index
from paf.capabilities.web import Web as _Web
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


def _build_filesystem(store) -> _FileSystem:
    """Scoped filesystem access rooted at the project root (paf-owned copy).

    `store` is accepted for a uniform builder signature but unused — the
    filesystem is self-contained over its own root, not the persistence store.
    Rooting at PROJECT_ROOT lets the model work with repo files (read/write/
    edit/list/search/find) while the harness toolset's traversal prevention +
    protected-pattern defaults (`.git/`, `.env`, keys, secrets read-only) keep it
    confined and safe.
    """
    return _FileSystem(root_dir=PROJECT_ROOT)


def _build_shell(store) -> _Shell:
    """Shell command execution rooted at the project root (paf-owned copy).

    `store` is accepted for a uniform builder signature but unused — the shell
    is self-contained over its own cwd, not the persistence store. Rooting at
    PROJECT_ROOT lets the model run commands against the repo; the harness
    toolset's deny-list (rm/dd/shutdown/...), interactive-command blocking, and
    output truncation keep it bounded. Environment defaults to inheriting the
    parent, so host secrets (incl. LLM API keys) remain present unless a caller
    opts into `denied_env_patterns=LLM_API_KEY_ENV_PATTERNS`.
    """
    return _Shell(cwd=PROJECT_ROOT)


def _build_git(store) -> _Git:
    """Safe git inspection + checkpoint tools rooted at the project root.

    `store` is accepted for a uniform builder signature but unused — git
    operates over the repo at PROJECT_ROOT, not the persistence store. The
    toolset confines every user-supplied path to this root; the two write tools
    (git_checkpoint / git_restore_paths) refuse empty path lists so the agent
    can never broad-`add -A` or wipe more than it named.
    """
    return _Git(root=PROJECT_ROOT)


def _build_web(store) -> _Web:
    """Tavily-backed web search + page extraction tools.

    `store` is accepted for a uniform builder signature but unused — web tool
    state is just the Tavily client, read lazily from `TAVILY_API_KEY` at call
    time (so the capability constructs and registers even before a key exists,
    and each tool reports the missing key/package rather than failing hard).
    """
    return _Web()


def _build_tool_output_limits(store) -> ToolOutputLimits:
    """Framework-level tool-return reduction — a BUILTIN, not a paf icon-capability.

    Intercepts oversized tool returns in `after_tool_execute` and reduces them
    once, so a giant return doesn't dominate (and re-send on every later request
    of) the context window. Complements the eviction/compaction tier: a spilled
    payload is offloaded to the flat-file store instead of being retained in
    history, then paged back on demand via the builtin `read_tool_result`.

    Config (per Landon: "plug-and-play with some configuration"): by default,
    returns over `_DEFAULT_THRESHOLD` (10_000 chars) are LOSS LESSLY SPILLED to a
    `LocalFileStore` with a bounded truncation fallback — zero LLM cost, no silent
    drop. `read_tool_result` is registered by the capability's own toolset.

    `store` (the shared persistence store) is accepted for a uniform builder
    signature but unused — ToolOutputLimits uses its own `LocalFileStore` spill
    backend by default (swap via `store=` for a custom OverflowStore).
    """
    return ToolOutputLimits()


def _build_warn_on_cache_busts(store) -> WarnOnCacheBusts:
    """Observational prompt-cache-collapse monitor — a BUILTIN, not a paf icon-capability.

    Attaches to the agent's model requests; after each response it reads
    ``usage.cache_read_tokens`` and warns (``CacheBustWarning``, stdlib ``warnings``)
    once when the established cacheable prefix collapses mid-run. A collapse means a
    moved prefix (reordered tools, injected timestamp, serialization-level block hop)
    or a provider-side expiry — either way it spikes billing by re-charging tokens a
    stable prefix would have served from cache. Purely observational: no tools,
    instructions, or model-settings, so it composes with every other capability.

    Defaults are conservative (collapse_ratio 0.5, min_prefix_tokens 1024,
    cache_ttl_seconds 300): only a drop below half of an established >=1024-token
    prefix fires, so ordinary rounding or partial misses don't warn. Silent when
    caching is off or unreported (cache_read_tokens stays 0), so it never fires
    spuriously in runs that don't cache. Same drop-in pattern as ToolOutputLimits
    — consumed as a pydantic-ai AbstractCapability, per Landon.

    To surface busts in Langfuse traces, route stdlib warnings into logging once at
    startup: ``logging.captureWarnings(True)`` (Logfire bridges logging, not warnings;
    they reach the trace via the ``py.warnings`` logger).
    """

    return WarnOnCacheBusts()


# Default model for the demo sub-agents. Bare deepseek slug (NOT the `:nitro`
# variant) for determinism — :nitro's `provider.sort: throughput` routing bounces
# across upstream providers (fp8 vs fp4) and gives run-to-run nondeterminism (see
# specs.py `_openrouter_model_settings`). Override via SUBAGENT_MODEL / .env.
_SUBAGENT_MODEL = os.environ.get("SUBAGENT_MODEL", "openrouter:deepseek/deepseek-v4-flash-0731")


def _build_subagents(store) -> SubAgents:
    """Explicit sub-agent delegation (BUILTIN, not a paf port) — demo leaf agents.

    Lets the demo agent delegate a self-contained task to a named child agent via
    one `delegate_task(agent_name, task)` tool. Each child runs in its own fresh,
    isolated run and never sees the parent conversation (the parent must pass
    everything the task needs). Two demo children, both on the demo model:

    - ``researcher`` — gathers/verifies facts and returns a tight evidence-backed
      brief (no tools, prompt-only; uses the same model as the demo subject).
    - ``writer`` — drafts prose/format from material the parent passes in (no
      tools; deterministic same-model leaf).

    This is a pydantic-ai AbstractCapability consumed from the harness per Landon
    (same drop-in pattern as ToolOutputLimits / WarnOnCacheBusts). It is NOT
    spec-serializable (holds live `Agent` instances; get_serialization_name ->
    None), so it is built ONLY here in Python by passing explicit `SubAgent`
    entries — it can never resolve from a YAML spec. `agent_folders=None` disables
    disk-agent loading entirely so the demo is purely explicit and deterministic
    (it does not vary with whatever markdown agent files happen to exist in
    `~/.agents/` or the project). `inherit_tools` is off (children don't get the
    parent's toolset), keeping delegation a clean leaf call. `store` unused
    (uniform builder signature).
    """
    from pydantic_ai import Agent

    researcher = Agent(
        _SUBAGENT_MODEL,
        name="researcher",
        description="Gathers and verifies facts, returns a tight evidence-backed brief",
        system_prompt=(
            "You are a research sub-agent. You do not see the parent conversation. "
            "Given a research task, return a concise, factual, evidence-backed brief. "
            "State what is certain vs. uncertain. Do not fabricate; say 'unknown' when you "
            "do not have evidence for something."
        ),
    )
    writer = Agent(
        _SUBAGENT_MODEL,
        name="writer",
        description="Drafts clear prose from material the delegating agent provides",
        system_prompt=(
            "You are a writing sub-agent. You do not see the parent conversation. "
            "Given source material and a writing task, produce clean, clear prose. "
            "Rewrite for clarity and structure, preserve all facts, add nothing."
        ),
    )
    return SubAgents(
        agents=[SubAgent(researcher), SubAgent(writer)],
        agent_folders=None,  # purely explicit, deterministic demo (no disk loading)
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
        # Scoped filesystem (paf-owned harness copy) rooted at the project root.
        "FileSystem": _build_filesystem,
        # Shell command execution (paf-owned harness copy) rooted at the project root.
        "Shell": _build_shell,
        # Safe git inspection + checkpoint tools rooted at the repo root.
        "Git": _build_git,
        # Tavily-backed web search + page extraction tools.
        "Web": _build_web,
        # Framework-level tool-return reduction (BUILTIN, paf-supplied config).
        "ToolOutputLimits": _build_tool_output_limits,
        # Observational prompt-cache-collapse monitor (BUILTIN, paf-supplied config).
        "WarnOnCacheBusts": _build_warn_on_cache_busts,
        # Sub-agent delegation (BUILTIN, not a paf port) — explicit child leaf agents.
        "Subagents": _build_subagents,
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
