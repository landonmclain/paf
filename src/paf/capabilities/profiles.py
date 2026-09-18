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

        Capabilities that **opt out of spec serialization** (``get_serialization_name()``
        returns ``None`` — e.g. ``Subagents``, which holds live ``Agent`` instances and
        can never be expressed in YAML) are **excluded** from the returned registry.
        ``Agent.from_spec`` insists every type in ``custom_capability_types`` be
        spec-serializable and hard-crashes otherwise; a serialization-opted-out
        capability can only be supplied as a live object via ``build_runtime_capabilities``
        (gated by ``capability_names``, not ``capability_types``), so it must be kept in
        the former but kept OUT of the latter.
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
        "Web",
        "Visual",
        "Browser",
        "Skills",
        "ToolOutputLimits",
        "WarnOnCacheBusts",
        "Subagents",
        "Planning",
        "ThreadSummarizer",
        "Secrets",
        # "Guardrails",
        # "ToolGating",
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
        # Codebase-map: navigation-first codebase browser rooted at the repo root
        # — overview/lookup_file/read_block/search_codebase + run_codebase_map
        # (re-index). Lets the model orient in a codebase by climbing a ladder
        # instead of brute-force keyword scans.
        "CodebaseMap",
        # Vault: navigation-first note ladder over a markdown vault. Roots at
        # the spec `vault:` block (default paf-owned data/vault/ playground) —
        # vault_overview/lookup_note/read_note + note_find/note_search. Lets the
        # model navigate and read an Obsidian-style vault instead of brute-force
        # scans. Tier B (graph) / Tier C (writes) are later phases.
        "Vault",
        # Tavily-backed web search + page extraction — search_web (open) +
        # fetch_web_page (domain-gated extract). Reads TAVILY_API_KEY at call
        # time; tools degrade to clear messages when key/package missing.
        # (Adapted from coding_agent web_search_tools.)
        "Web",
        # Vision-model description of captured screenshots (describe_screenshot)
        # — one tool that sends base64 PNG(s) + a prompt to a vision model and
        # returns its text description. A contained side-call, NOT a sub-agent
        # (no child run / no delegation / no inherited tools). Reads VISION_MODEL
        # at call time (default openrouter:xiaomi/mimo-v2.5).
        "Visual",
        # Mobile-first Chromium browser automation (paf-owned full copy of the
        # harness PlaywrightBrowser) — ~19 browser tools (navigate/snapshot/click/
        # type/press/select/hover/wait_for/screenshot/get_text/scroll/back/forward/
        # execute_js/console_messages/tabs/dialog/network/set_viewport) over a
        # headless Chromium with native SSRF defense (allowed_domains +
        # block_private_addresses) and chromium_sandbox. The context is mobile by
        # default (390×844 touch, retina scale) and `set_viewport` switches the
        # viewport/mobile profile mid-run by recreating the context. Requires the
        # harness[playwright] extra + Chromium (`playwright install chromium`).
        # Headless + private-address-blocked defaults set in build.py `_build_browser`.
        "Browser",
        # Skill library — LIST / LOAD on demand + deterministic POINTER trigger.
        # list_skills/load_skill keep the prompt thin; point_skills pre-arms
        # which skill a message needs (zero-LLM BM25, skill-note-hint re-homed).
        # One paf-native capability for all skill functionality, wired to the
        # repo's data/skills/ library.
        "Skills",
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
        # Model-owned task plan for long turns (paf-owned port — rehomed from
        # the harness BUILTIN) — write_plan/read_plan/add_task/update_task_status/
        # remove_task, surfaced each request as a marker-tagged, cache-safe tail
        # reminder. Persisted per-conversation to data/plans/<cid>/plan.json via
        # a FilePlanStore (flat-file, diffable). Built in Python (build.py
        # `_build_planning`) so the profile gate controls it.
        "Planning",
        # Thread-conversation summarizer — per-conversation episodic background
        # summarizer. Phase 1: captures each eviction block (with per-message
        # {run_id}:{turn_id} refs) to data/summaries/<agent>/evictions.jsonl via
        # the window hook (wired only when BOTH ThreadSummarizer and Compaction
        # are allowlisted). OFF for the production assistant; ON here so the
        # "everything sandbox" exercises the capture waterfall live.
        "ThreadSummarizer",
        # Structural prompt-injection defense (BUILTIN, not a paf port) — screens
        # tool results and withholds high-risk content before the model sees it.
        # Option A from the guardrails plan: adopt the builtin PromptInjectionDefender
        # (semantic_detection=True, Tier 2 MiniLM classifier). paf's own
        # on_detection handler (guardrails/prompt_injection.py) traces each hit.
        # "Guardrails",
        # Per-call execution-time tool gate (BUILTIN shell, paf-owned policy) —
        # refuses the highest-risk tools (shell, git, filesystem writes) before
        # they execute. The profile is only a build-time tool allowlist; this is
        # the missing per-call gate.
        # "ToolGating",
        # Secret-handle resolution + result redaction (BUILTIN shell, paf-owned
        # policy) — the model proposes $SECRET:key, the runtime resolves it at
        # execution time so the real value never enters the window.
        "Secrets",
    ),
)


# The ContextBench benchmark-runner profile — the codebase-map × benchmark A/B
# harness agent.
#
# contextbench is the agent that runs the ContextBench-flavored manual/eventual
# full eval of paf's navigation-first codebase-map. Per Landon (2026-08-30), the
# "product arm" toolset is the default: BOTH codebase-map (the map ladder) AND a
# bare shell (ls/find/rg/cat discovery) are allowlisted in one profile, and the
# cold-start-vs-map arms are branched later by *instruction*, not by separate
# profiles/specs (keeping the toolset the only variable in spirit, one agent kind).
#
# Deliberately NOT the everything-sandbox: no Web/Browser/Vault/Visual/Skills/
# Subagents/Planning — the benchmark agent should not wander off to unrelated
# capability surfaces. It gets the codebrowser + the shell/git/filesystem it
# needs to inspect and patch a snapshot, rooted per-spec at the target snapshot.
CONTEXTBENCH_PROFILE = AgentProfile(
    name="contextbench",
    capability_names=(
        # No-map control on the SAME ansible `10a41eff` question (e14): replicate
        # e10-vs-e12 on the big repo. The model navigates the ~150MB ansible tree
        # with bare discovery only (Shell/FileSystem/Git). The map arm (e13) did it
        # in 13 calls / 0 searches / 3.4M chars; the question is whether bare grep +
        # whole-file reads stay tractable on a repo this size, or spiral.
        # "CodebaseMap",
        "Shell",
        "FileSystem",
        "Git",
    ),
)


# The runtime2 test/profile helper — a LEAN agent for exercising the
# Session stream/event contract (src/paf/runtime.py), NOT a product surface.
#
# runtime2_test gets the smallest toolset that can deterministically produce
# every turn-stream lifecycle edge: Shell's `run` is both the success path
# (`run("echo ok")` -> tool_call -> ToolRunning -> tool_result) and the
# deterministic FALLIBLE path (`run("exit 1")` -> a non-zero ToolReturnPart
# outcome -> ToolError with the same step_id). No Memory/Compaction/Vault/
# Search/Web/Browser/Subagents — those derail a trivial prompt into 8-search
# loops and are out of scope for asserting ToolRunning/ToolError pairing.
RUNTIME2_TEST_PROFILE = AgentProfile(
    name="runtime2_test",
    capability_names=(
        # StepPersistence so turns actually land in the store (data/paf/) — the
        # web UI's history render reads list_runs(cid); without this writer the
        # lane stays empty and nothing renders on reload. Purely additive (writes
        # steps to disk, no extra tools), so the deterministic ToolRunning/
        # ToolError test surface is undisturbed.
        "StepPersistence",
        "Shell",
    ),
)


# Name lookup so a spec/CLI can reference a profile by a string that resolves
# to the allowlist. Only used by spec-driven loaders; keep small.
PROFILES: dict[str, AgentProfile] = {
    profile.name: profile
    for profile in (
        ASSISTANT_PROFILE,
        DEMO_PROFILE,
        CONTEXTBENCH_PROFILE,
        RUNTIME2_TEST_PROFILE,
    )
}
