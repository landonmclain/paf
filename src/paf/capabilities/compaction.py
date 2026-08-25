"""Context compaction assembly — translate config into real compaction objects.

Spec files (`specs/*.yaml`) go through pydantic-ai's ``AgentSpec``/``Agent.from_spec``,
which resolves each top-level ``capabilities:`` entry via that class's ``from_spec()``
and — crucially — only ONE level deep (see ``_capabilities_from_spec`` in pydantic-ai's
``agent/__init__.py``: it calls ``cap_cls.from_spec(*args, **kwargs)`` once per entry).

Harness compaction classes (``TieredCompaction`` and every tier type) do NOT override
``from_spec``, so a YAML ``tiers:`` list of dicts stays raw dicts and crashes at run time
(``AttributeError ... 'model'``). ``TieredCompaction.tiers`` is typed
``Sequence[CompactionStrategy]`` — real objects. So the tiered compaction cannot be
declared in YAML; it must be assembled here, in Python.

This module owns that assembly. ``build_compaction()`` builds the ``TieredCompaction``
ladder **minus the summarizing tier**: the LLM rewrite (``SummarizingCompaction``) is
deliberately excluded — long-term recall in paf is the **store-backed memory system**
(owner → organizer → curated index, via ``ConversationSearch`` / ``read_turn``), not a
compacted summary. (Historical note: the summary tier was once scoped as mutually
exclusive with the now-retired turn-index one-liner block; turn-index was archived and we
no longer summarize at all.) The non-summary tiers (clamp, clear-tool-results, sliding
window) are pure context management and stay; they only ever *shrink* the live context
without producing a competing summary. ``build_runtime_capabilities()``
(in ``build.py``) selects the compaction builder iff the agent's profile allowlists it.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    SlidingWindowCompaction,
    TieredCompaction,
)

from paf.config import settings

# ---------------------------------------------------------------------------
# Compaction tuning — spec-file driven (per-agent), with benchmark-default fallback.
#
# Prod compaction dials live in the agent's spec YAML under a top-level
# `compact:` block (option B — diffable, per-agent, reviewable). `build_agent`
# reads that block with `CompactionConfig.model_validate(...)`, then threads it
# through `build_runtime_capabilities(profile, compaction_config=...)` ->
# `build_compaction(config=...)`. When NO config is supplied (None), the builder
# falls back to the historical benchmark-tuned defaults so existing call sites,
# tests, and the harness all keep byte-identical behavior.
# ---------------------------------------------------------------------------


@dataclass
class CompactionConfig:
    """Per-assistant context-compaction tuning (from `compact:` in the spec YAML).

    All fields optional — an unset field falls back to the same default
    `build_compaction()` uses when given no config at all. This keeps the spec
    declarative ("only set what this agent wants different") while the builder
    owns the canonical defaults.

    Attributes:
        target_tokens: live-history size (tokens) that *triggers* compaction to
            start escalating. HIGH for live 300k-context (keep conversation in
            context, compact only near the ceiling); the benchmark uses 5000 to
            force-evict evidence out of context for retrieval testing.
        window_max_messages: sliding-window tier — how many recent messages to
            examine before evicting the oldest.
        window_keep_messages: sliding-window tier — how many recent messages to
            KEEP after eviction (the surviving recent window).
        clear_tool_max_tokens: ClearToolResults — above this size, a tool result
            is collapsed to a placeholder. Low = aggressively scavenge bulky tool
            dumps (biggest context saver); high = keep more tool output verbatim.
        clear_tool_keep_pairs: ClearToolResults — how many most-recent
            tool-call/result pairs always survive clearing.
        clamp_max_part_tokens: ClampOversizedMessages — any single message part
            larger than this is trimmed to head+tail.
        clamp_head_chars / clamp_tail_chars: head/tail chars kept when clamping.
        consolidate: when True, replace the plain sliding window with
            MemoryConsolidationWindow (fold the about-to-evict slice into the
            durable journal before dropping). Default False.
    """

    target_tokens: int | None = None
    window_max_messages: int | None = None
    window_keep_messages: int | None = None
    clear_tool_max_tokens: int | None = None
    clear_tool_keep_pairs: int | None = None
    clamp_max_part_tokens: int | None = None
    clamp_head_chars: int | None = None
    clamp_tail_chars: int | None = None
    consolidate: bool | None = None

    @classmethod
    def from_mapping(cls, data: dict | None) -> CompactionConfig:
        """Validate a raw YAML mapping leniently (unknown keys ignored)."""
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def build_compaction(config: CompactionConfig | None = None) -> TieredCompaction:
    """Assemble the assistant's context-compaction capability (no summary tier).

    Cheap tiers first, expensive last — ``TieredCompaction`` runs them in order
    and escalates only while the history still exceeds the budget (it stops once
    the history fits ``target_fraction`` of the model's context window). Ladder
    (least → most destructive / expensive), WITHOUT ``SummarizingCompaction``:

    1. ``ClampOversizedMessages`` — trim any single oversized part to head+tail.
    2. ``ClearToolResults`` — replace tool-call/result pairs with placeholders.
    3. ``SlidingWindowCompaction`` — drop oldest messages, keep the recent window
       (plus the first user message) — the harness's cheap default.

    ``config`` (optional, from the agent spec's ``compact:`` block) overrides the
    historical defaults. When ``None``, the historical defaults are used verbatim
    (benchmark-tuned: ``target_tokens=4000``, window 10/5, ClearToolResults 500/4,
    clamp 50_000/2000) so existing call sites and tests keep byte-identical
    behavior. When ``config`` supplies values, they replace only the fields set.

    When ``config.consolidate`` (or ``Settings.consolidate``) is True, the
    sliding-window tier is replaced by
    :class:`~paf.capabilities.memory._window.MemoryConsolidationWindow`, which
    folds the about-to-be-evicted slice into the durable memory store (via the
    writable owner sub-agent) *before* dropping it. Default (unset) keeps the
    byte-identical plain window.

    The LLM rewrite (4. ``SummarizingCompaction``) is intentionally absent — paf's
    long-term recall is the store-backed memory system (owner → organizer → curated
    index), not a compacted summary, and we no longer summarize at all (the old
    turn-index summary layer that once made it mutually exclusive is retired/archived).
    Context is bounded here by pruning; long-term recall is the store-backed
    ``ConversationSearch`` / ``read_turn`` instead of a compacted summary.

    Historical default values target ~4000 tokens of live history; a spec
    ``compact:`` block overrides for live (e.g. the assistant spec uses
    ``target_tokens`` ~ 1/3 of the 300k context so compaction only fires on
    genuinely long conversations, plus a wide window).
    """
    cfg = config or CompactionConfig()

    target = cfg.target_tokens if cfg.target_tokens is not None else 4000
    clear_max = cfg.clear_tool_max_tokens if cfg.clear_tool_max_tokens is not None else 500
    clear_keep = cfg.clear_tool_keep_pairs if cfg.clear_tool_keep_pairs is not None else 4
    clamp_max = cfg.clamp_max_part_tokens if cfg.clamp_max_part_tokens is not None else 50_000
    clamp_head = cfg.clamp_head_chars if cfg.clamp_head_chars is not None else 2_000
    clamp_tail = cfg.clamp_tail_chars if cfg.clamp_tail_chars is not None else 2_000
    win_max = cfg.window_max_messages if cfg.window_max_messages is not None else 10
    win_keep = cfg.window_keep_messages if cfg.window_keep_messages is not None else 20
    consolidate = cfg.consolidate if cfg.consolidate is not None else bool(settings.consolidate)

    if consolidate:
        from paf.capabilities.memory._window import MemoryConsolidationWindow

        window = MemoryConsolidationWindow(
            max_messages=win_max,
            keep_messages=win_keep,
        )
    else:
        window = SlidingWindowCompaction(
            max_messages=win_max,
            keep_messages=win_keep,
        )
    return TieredCompaction(
        target_tokens=target,
        tiers=[
            # 1. Clamp any single oversized part (rare, but cheap when it fires).
            ClampOversizedMessages(
                max_part_tokens=clamp_max,
                keep_head_chars=clamp_head,
                keep_tail_chars=clamp_tail,
            ),
            # 2. Clear old tool results, keeping the most recent pairs intact.
            #    Fires only when the live history is genuinely bulky; keeps the
            #    `clear_keep` most-recent tool-call/result pairs untouched.
            ClearToolResults(
                max_tokens=clear_max,
                keep_pairs=clear_keep,
            ),
            # 3. Drop the oldest messages (or consolidate-then-drop, when enabled).
            window,
        ],
    )
