"""Capability catalog — the master registry of capability types an agent can enable.

A profile is a named allowlist over this catalog (see ``profiles.py``). Core
pydantic-ai capabilities (``WebSearch``, ``Thinking``, etc.) need no catalog entry —
the spec loader resolves them natively. This catalog exists only for capabilities
the loader cannot resolve by default (harness capabilities like ``StepPersistence``),
keyed by the CamelCase name used in the spec YAML.

Compaction is built in Python (see ``compaction.py``): ``TieredCompaction``'s nested
``tiers`` must be real strategy objects, so a YAML ``tiers:`` dict list would crash
at run time. The tier types are still cataloged so a profile can reference them and
the schema stays complete.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.capabilities import MCP
from pydantic_ai_harness import PromptInjectionDefender, ToolGuardrail
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    SlidingWindowCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.step_persistence import StepPersistence
from pydantic_ai_harness.subagents import SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits
from pydantic_ai_harness.warn_on_cache_busts import WarnOnCacheBusts

from paf.capabilities._window import TurnBoundSlidingWindow
from paf.capabilities.browser import Browser
from paf.capabilities.codebase_map import CodebaseMap
from paf.capabilities.fs import FileSystem
from paf.capabilities.git import Git
from paf.capabilities.memory import Memory
from paf.capabilities.planning import Planning
from paf.capabilities.search import ConversationSearch
from paf.capabilities.shell import Shell
from paf.capabilities.skills import Skills
from paf.capabilities.thread_summarizer import ThreadSummarizer
from paf.capabilities.vault import Vault
from paf.capabilities.visual import Visual
from paf.capabilities.web import Web

# name -> capability type. The name is the serialization form used in a spec's
# `capabilities:` entry and the key a profile allowlists against.
CAPABILITY_TYPES: dict[str, type[Any]] = {
    "StepPersistence": StepPersistence,
    # Python-built no-summary compaction ladder; tier types cataloged for schema.
    "Compaction": TieredCompaction,
    "TieredCompaction": TieredCompaction,
    "ClampOversizedMessages": ClampOversizedMessages,
    "ClearToolResults": ClearToolResults,
    "SlidingWindowCompaction": SlidingWindowCompaction,
    # Turn-bounded trim-only sliding-window tier — never evicts the active turn.
    "TurnBoundSlidingWindow": TurnBoundSlidingWindow,
    # Lossless recall over the persistence store; built in Python (needs a real
    # SnapshotHistorySource), cataloged so the allowlist + schema stay complete.
    "ConversationSearch": ConversationSearch,
    # Durable-facts notebook (MEMORY.md + topic files). Has `from_spec`, so it
    # resolves declaratively AND via build.py.
    "Memory": Memory,
    # Per-conversation background thread summarizer. OFF by default — the profile
    # allowlist is the ON/OFF switch.
    "ThreadSummarizer": ThreadSummarizer,
    # Scoped filesystem access rooted at a directory. `from_spec` + `get_toolset`.
    "FileSystem": FileSystem,
    # Shell command execution rooted at a cwd. `from_spec` + `get_toolset`.
    "Shell": Shell,
    # Safe git inspection + checkpoint tools rooted at a workspace dir.
    "Git": Git,
    # Navigation-first codebase browser. Built in Python (needs a real scan_root
    # + storage), not spec-serialized.
    "CodebaseMap": CodebaseMap,
    # Navigation-first note ladder over a markdown vault. Built in Python (needs
    # a real vault_root + allow/deny), not spec-serialized.
    "Vault": Vault,
    # Tavily-backed web search + page extraction. Reads TAVILY_API_KEY at call
    # time; fetch_web_page is domain-gated.
    "Web": Web,
    # Vision-model description of captured screenshots. Reads VISION_MODEL at
    # call time. Built via build.py.
    "Visual": Visual,
    # Mobile-first Chromium browser automation over a headless Chromium.
    # `from_spec` AND `get_toolset`. Requires the harness[playwright] extra +
    # `playwright install chromium`.
    "Browser": Browser,
    # Framework-level tool-return reduction (spill/truncate oversized results).
    # Built in Python so the profile gate controls it.
    "ToolOutputLimits": ToolOutputLimits,
    # Observational prompt-cache-collapse monitor. Silent when caching is off.
    # Built in Python so the profile gate controls it.
    "WarnOnCacheBusts": WarnOnCacheBusts,
    # Skill library — LIST / LOAD on demand + a deterministic POINTER trigger.
    # `get_toolset` so it resolves via build.py (and declaratively with a path).
    "Skills": Skills,
    # Sub-agent delegation — a single `delegate_task` tool. NOT spec-serializable
    # (holds live Agent instances), so always built in Python.
    "Subagents": SubAgents,
    # Model-owned task plan for long turns. Spec-serializable, but wired via the
    # Python seam so the profile gate controls it.
    "Planning": Planning,
    # Structural prompt-injection defense. Built in Python so the profile gate
    # controls it.
    "Guardrails": PromptInjectionDefender,
    # Per-call execution-time tool gate. Built in Python so the profile gate
    # controls it.
    "ToolGating": ToolGuardrail,
    # Secret-handle resolution + result redaction. Built in Python so the profile
    # gate controls it.
    "Secrets": ToolGuardrail,
    # Builtin MCP servers. A core pydantic-ai capability (has `from_spec`), so the
    # loader could resolve it natively; it's cataloged so a profile can allowlist
    # it and the gate stays consistent (build.py wraps loaded toolsets in MCP).
    "MCP": MCP,
}
