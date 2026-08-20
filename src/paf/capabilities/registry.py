"""Capability catalog — the master registry of capability types an agent can enable.

A profile is a *named allowlist* over this catalog (see ``profiles.py``). M1 ships
all harness capabilities we actually attach to an agent; nothing is scaffolded
ahead of need (YAGNI). Core pydantic-ai capabilities (``WebSearch``, ``Thinking``,
etc.) do NOT need a catalog entry — they are native to ``Agent.from_spec`` and the
spec loader resolves them without ``custom_capability_types``. This catalog exists
only for capabilities the loader cannot resolve by default (harness capabilities
like ``StepPersistence``), keyed by the CamelCase name used in the spec YAML.

Compaction is built in Python (see ``compaction.py``): ``TieredCompaction``'s nested
``tiers`` must be real strategy objects, and no harness compaction class overrides
``from_spec``, so a YAML ``tiers:`` dict list would crash at run time. The tier types
are still cataloged so a profile can reference them and the schema stays complete.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.step_persistence import StepPersistence
from pydantic_ai_harness.subagents import SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits
from pydantic_ai_harness.warn_on_cache_busts import WarnOnCacheBusts

from paf.capabilities.fs import FileSystem
from paf.capabilities.git import Git
from paf.capabilities.memory import Memory
from paf.capabilities.memory._window import MemoryConsolidationWindow
from paf.capabilities.search import ConversationSearch
from paf.capabilities.shell import Shell
from paf.capabilities.turn_index import TurnIndexCapability
from paf.capabilities.web import Web

# name -> capability type. The name is the serialization form used in a spec's
# `capabilities:` entry and the key a profile allowlists against.
CAPABILITY_TYPES: dict[str, type[Any]] = {
    "StepPersistence": StepPersistence,
    # "Compaction" gates the Python-built no-summary ladder (see compaction.py);
    # TieredCompaction is its concrete type for catalog/schema completeness.
    "Compaction": TieredCompaction,
    # Compaction tier types. TieredCompaction is assembled in Python (compaction.py);
    # the tier types are cataloged so the profile allowlist + schema stay complete.
    "TieredCompaction": TieredCompaction,
    "ClampOversizedMessages": ClampOversizedMessages,
    "ClearToolResults": ClearToolResults,
    "SlidingWindowCompaction": SlidingWindowCompaction,
    # Consolidation-on swap for the sliding-window tier (Settings.consolidate /
    # CONSOLIDATE=1): folds the about-to-evict slice into durable memory first.
    "MemoryConsolidationWindow": MemoryConsolidationWindow,
    "SummarizingCompaction": SummarizingCompaction,
    # Lossless recall over the persistence store. Built in Python (build.py) because
    # its `source` needs a real SnapshotHistorySource object; cataloged so the
    # profile allowlist + schema stay complete.
    "ConversationSearch": ConversationSearch,
    # One-line-per-turn index in context + gated `read_turn`. Built in Python (build.py)
    # to share the single persistence store; has `from_spec`, so it can ALSO resolve
    # declaratively from a spec (`- TurnIndex: {index_dir: ...}`).
    "TurnIndex": TurnIndexCapability,
    # Durable-facts notebook (MEMORY.md + topic files). Near-copy of the harness
    # Memory capability; paf-owned so its injected block shares the one marker
    # convention. Built by build.py whenever the profile allowlists it; for the
    # assistant profile it IS the recall-ladder owner (see profiles.py). Has
    # `from_spec`, so it ALSO resolves declaratively (`- Memory: {...}`).
    "Memory": Memory,
    # Scoped filesystem access (read/write/edit/list/search/find) rooted at a
    # directory, copied verbatim from the harness. Has `get_toolset`/`from_spec`
    # so it resolves declaratively from a spec (`- FileSystem: {root_dir: ...}`)
    # AND can be built in Python (build.py) with a paf-anchored root_dir.
    "FileSystem": FileSystem,
    # Shell command execution (run/start/check/stop) rooted at a cwd, copied
    # verbatim from the harness. `from_spec` + `get_toolset` so it resolves
    # declaratively AND via build.py with paf-anchored settings.
    "Shell": Shell,
    # Safe git inspection + checkpoint tools (status/changed/diff/checkpoint/
    # restore) rooted at a workspace dir. Adapted from coding_agent git_tools;
    # `get_toolset` so it resolves via build.py (and declaratively when rooted).
    "Git": Git,
    # Tavily-backed web search + page extraction (search_web_docs /
    # search_official_docs / fetch_web_page). Adapted from coding_agent
    # web_search_tools. Reads TAVILY_API_KEY from the env at call time.
    "Web": Web,
    # Framework-level tool-return reduction (truncate/summarize/spill oversized
    # tool results via after_tool_execute + a `read_tool_result` read-back tool).
    # A pydantic-ai AbstractCapability from the harness (NOT a paf icon-capability
    # / FunctionToolset bundle) — built in Python (build.py) so the profile gate
    # controls it and real Band/Spill config objects are passed; cataloged so the
    # profile allowlist + schema resolve. Kept as a BUILTIN per Landon (plug-and-play
    # with config, not ported into paf.capabilities).
    "ToolOutputLimits": ToolOutputLimits,
    # Observational prompt-cache-collapse monitor (BUILTIN, not a paf port). Reads
    # `usage.cache_read_tokens` after each model request and warns (CacheBustWarning,
    # stdlib `warnings`) when the established cacheable prefix collapses mid-run —
    # a moved prefix or a provider-side expiry spikes billing. Consumed as a
    # pydantic-ai AbstractCapability; adds no tools/instructions/model-settings so it
    # composes with anything. Silent when caching is off/unreported. Cataloged so the
    # profile allowlist + schema resolve; built in Python (build.py) so profile gate
    # controls it and real config objects are passed. Kept as a BUILTIN per Landon —
    # same drop-in pattern as ToolOutputLimits, NOT ported into paf.capabilities.
    "WarnOnCacheBusts": WarnOnCacheBusts,
    # Sub-agent delegation: expose a single `delegate_task(agent_name, task)` tool so
    # the agent can hand a self-contained task to a named child agent that runs in its
    # own fresh, isolated run (never sees this conversation). Consumed as a pydantic-ai
    # AbstractCapability from the harness (NOT a paf icon-capability port) per Landon.
    # NOT spec-serializable — it holds live `Agent` instances (get_serialization_name
    # returns None), so it is ALWAYS built in Python (build.py `_build_subagents`) by
    # passing explicit child `SubAgent` entries; it can never resolve from a YAML spec.
    "Subagents": SubAgents,
}
