"""Subagent roster: leaf specs -> built child agents + their run policy.

Each leaf is a spec under ``specs/sub_agents/`` resolved by bare name (the
recursive ``get_spec_path``), carrying the pydantic-ai fields (model,
model_settings, description, instructions) plus a paf-specific ``subagent:``
policy block that ``AgentSpec`` would drop::

    subagent:
      name: writer              # runtime name (delegate_task key); default:
                                # the spec name minus a trailing "_subagent"
      timeout_seconds: 300
      usage_limits: {request_limit: 30, total_tokens_limit: 400000}
      max_calls_per_run: 5
      background: allowed       # allowed | required | never

Children always run with their OWN usage budget (never the parent's run usage —
a background child outlives its parent run), so the timeout and usage limit
default to bounded values when a spec omits them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

BackgroundPolicy = Literal["allowed", "required", "never"]

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_REQUEST_LIMIT = 30

LEAF_EXCLUDED_CAPABILITIES = frozenset(
    {"Subagents", "StepPersistence", "ConversationSearch", "Memory", "ThreadSummarizer"}
)
"""Never built for a child, whatever its profile allows: a leaf is depth 1 (no
delegation) and persists results, not transcripts (no step store, recall,
memory, or summarizer — nothing that would write into or read from the
parent conversation)."""

TLDR_INSTRUCTION = (
    "Start your response with a 1-3 sentence TL;DR of the result, then the full "
    "deliverable."
)
"""Appended to every leaf's instructions. A convention, not enforced: the
parent's completion notify shows the head of the output, so a TL;DR up front
makes a truncated notify still carry the answer."""


@dataclass(frozen=True)
class LeafPolicy:
    """Per-leaf run policy from the spec's ``subagent:`` block."""

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    usage_limits: UsageLimits = field(
        default_factory=lambda: UsageLimits(request_limit=DEFAULT_REQUEST_LIMIT)
    )
    max_calls_per_run: int | None = None
    background: BackgroundPolicy = "allowed"

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> LeafPolicy:
        """Parse a ``subagent:`` block; absent keys keep the bounded defaults."""
        raw = raw if isinstance(raw, dict) else {}
        limits = raw.get("usage_limits")
        background = raw.get("background", "allowed")
        if background not in ("allowed", "required", "never"):
            raise ValueError(
                f"subagent.background must be allowed|required|never, got {background!r}"
            )
        max_calls = raw.get("max_calls_per_run")
        return cls(
            timeout_seconds=float(raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)),
            usage_limits=(
                UsageLimits(**limits)
                if isinstance(limits, dict)
                else UsageLimits(request_limit=DEFAULT_REQUEST_LIMIT)
            ),
            max_calls_per_run=int(max_calls) if max_calls is not None else None,
            background=cast(BackgroundPolicy, background),
        )


@dataclass(frozen=True)
class Leaf:
    """One delegatable child: runtime name, description, agent, policy."""

    name: str
    description: str
    agent: Agent[Any, Any]
    policy: LeafPolicy = field(default_factory=LeafPolicy)


def load_leaf(
    spec_name: str,
    *,
    store: Any,
    filesystem_config: dict | None,
    agent_name: str | None = None,
    spec_directory: Path | None = None,
) -> Leaf:
    """Build one leaf from its spec.

    Tools come from one of two places:

    - **No ``profile:``** (the writer): just the SAME scoped FileSystem as the
      parent (root from the parent's ``filesystem_config`` / default
      PROJECT_ROOT, unless the leaf spec sets ``filesystem.root_dir``) — the
      parent passes file paths and the child reads them in its own fresh run.
    - **``profile: <name>``**: that profile's capabilities, built through the
      same ``build_runtime_capabilities`` path as the subject agent, from the
      leaf spec's own config blocks (``web:``, ``shell:``, ...). The filesystem
      root still defaults to the parent's.

    Either way a child NEVER gets :data:`LEAF_EXCLUDED_CAPABILITIES` — no step
    persistence, recall, memory, summarizer, or subagents (depth 1).
    ``agent_name`` (the parent paf agent) scopes any durable store a tooled leaf
    uses, e.g. Shell's. ``spec_directory`` overrides the spec root (tests).
    """
    # Local imports: specs.py imports build.py at module level, and build.py
    # imports this package — importing specs here at module load would cycle.
    from paf.capabilities.build import (  # noqa: PLC0415
        _build_filesystem,
        build_runtime_capabilities,
    )
    from paf.capabilities.profiles import (  # noqa: PLC0415
        PROFILES,
        SUBAGENT_PROFILES,
        AgentProfile,
    )
    from paf.specs import (  # noqa: PLC0415
        DEFAULT_SPEC_DIRECTORY,
        _spec_runtime_configs,
        get_spec_path,
        resolve_background_spec,
    )

    spec_dir = spec_directory or DEFAULT_SPEC_DIRECTORY
    spec, raw = resolve_background_spec(spec_name, spec_dir)
    spec_path = get_spec_path(spec_name, spec_dir)
    block_raw = raw.get("subagent")
    block: dict[str, Any] = block_raw if isinstance(block_raw, dict) else {}
    name = str(block.get("name") or spec_name.removesuffix("_subagent"))
    configs = _spec_runtime_configs(spec_path)
    configs["filesystem_config"] = configs["filesystem_config"] or filesystem_config
    configs.pop("subagents_config", None)  # depth 1: a leaf never delegates

    profile_name = raw.get("profile")
    if profile_name:
        known = {**PROFILES, **SUBAGENT_PROFILES}
        if profile_name not in known:
            raise KeyError(
                f"Subagent spec {spec_name!r} names unknown profile {profile_name!r}. "
                f"Known: {sorted(known)}"
            )
        allowed = tuple(
            cap
            for cap in known[profile_name].capability_names
            if cap not in LEAF_EXCLUDED_CAPABILITIES
        )
        capabilities = build_runtime_capabilities(
            AgentProfile(name=f"subagent:{name}", capability_names=allowed),
            **configs,
            agent_name=agent_name,
        )
    else:
        capabilities = [_build_filesystem(store, configs["filesystem_config"])]
    # `instructions=` APPENDS to the spec's own `instructions:` (from_spec merges
    # them), so only the TL;DR line is passed. The previous builder also passed
    # the spec text as `system_prompt=`, which sent the writer its full prompt
    # twice (instructions + a system-prompt part).
    agent = Agent.from_spec(
        spec,
        model=spec.model,
        name=name,
        description=spec.description,
        instructions=TLDR_INSTRUCTION,
        model_settings=cast(Any, spec.model_settings),
        capabilities=capabilities,
    )
    return Leaf(
        name=name,
        description=str(spec.description or ""),
        agent=agent,
        policy=LeafPolicy.from_raw(block),
    )
