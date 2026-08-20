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


# The demo/live-testing profile — the "EVERYTHING" sandbox.
#
# demo_assistant is the agent where we build out and tune the full stack —
# persistence, recall, durable memory (journal), and context compaction — plus
# poke at native spec-declarable capabilities (WebSearch, WebFetch, XSearch,
# Thinking, MCP, ToolSearch, ImageGeneration) via the YAML `capabilities:` list.
# The profile allowlists every *harness* capability that needs a Python build
# step (the ones pydantic-ai can't resolve natively): StepPersistence,
# ConversationSearch, Memory, and Compaction — mirroring the production
# assistant's stack. Once the experiment proves out, demo's tuned values are
# promoted to the clean production `assistant` spec and this sandbox is reset.
# Native capabilities don't belong here (they need no builder / no allowlist
# row); they resolve directly from the demo spec's YAML `capabilities:`.
DEMO_PROFILE = AgentProfile(
    name="demo",
    capability_names=(
        "StepPersistence",
        "ConversationSearch",
        "Memory",
        "Compaction",
        # Scoped filesystem access — the "everything sandbox" gets it so the
        # model can read/write/edit/list/search/find under a root, with
        # expected_hash stale-write guarding. (Copied harness FileSystem.)
        "FileSystem",
        # Shell command execution — run/start/check/stop, rooted at a cwd, with
        # allow/deny lists, interactive-command blocking, and env-masking
        # (incl. LLM-creds preset). (Copied harness Shell.)
        "Shell",
        # Safe git inspection + checkpoint tools — status/changed/diff (chunked,
        # paged), checkpoint (explicit paths only, no broad add), restore
        # (explicit paths only). (Adapted from coding_agent git_tools.)
        "Git",
        # Tavily-backed web search + page extraction — search_web_docs,
        # search_official_docs, fetch_web_page. Reads TAVILY_API_KEY at call
        # time; tools degrade to clear messages when key/package missing.
        # (Adapted from coding_agent web_search_tools.)
        "Web",
        # Framework-level tool-return reduction (BUILTIN, not a paf port) —
        # spills/summarizes/truncates oversized tool results so a giant return
        # doesn't dominate context. Consumed as a pydantic-ai AbstractCapability
        # with paf-supplied config; the builtin `read_tool_result` lets the model
        # page spilled payloads back on demand. Plug-and-play per Landon.
        "ToolOutputLimits",
        # Observational prompt-cache-collapse monitor (BUILTIN, not a paf port) —
        # warns via CacheBustWarning (stdlib `warnings`) when the cacheable prefix
        # collapses mid-run (moved prefix or provider-side expiry → billing spike).
        # Observer-only: no tools/instructions/model-settings. Silent when caching
        # is off/unreported. Same drop-in pattern as ToolOutputLimits per Landon.
        "WarnOnCacheBusts",
        # Sub-agent delegation (BUILTIN, not a paf port) — exposes a single
        # `delegate_task(agent_name, task)` tool so the model can hand a
        # self-contained task to a named child agent running in its own fresh,
        # isolated run. Explicit child agents are defined in code (build.py
        # `_build_subagents`) because the capability is not spec-serializable.
        # Demo: two named sub-agents sharing the demo model.
        "Subagents",
    ),
)


# Name lookup so a spec/CLI can reference a profile by a string that resolves
# to the allowlist. Only used by spec-driven loaders; keep small.
PROFILES: dict[str, AgentProfile] = {
    profile.name: profile for profile in (ASSISTANT_PROFILE, DEMO_PROFILE)
}
