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
ladder **minus the summarizing tier**: with turn-index ON (the M2 summary layer), the LLM
rewrite (``SummarizingCompaction``) is mutually exclusive with the `<turn-index>` one-liner
block — both inject a compressed *memory* of past turns. The non-summary tiers (clamp,
clear-tool-results, sliding window) are pure context management and stay; they only ever
*shrink* the live context without producing a competing summary. ``build_runtime_capabilities()``
(in ``build.py``) selects the compaction builder iff the agent's profile allowlists it.
"""

from __future__ import annotations

from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    SlidingWindowCompaction,
    TieredCompaction,
)

from paf.config import settings


def build_compaction() -> TieredCompaction:
    """Assemble the assistant's context-compaction capability (no summary tier).

    Cheap tiers first, expensive last — ``TieredCompaction`` runs them in order
    and escalates only while the history still exceeds the budget (it stops once
    the history fits ``target_fraction`` of the model's context window). Ladder
    (least → most destructive / expensive), WITHOUT ``SummarizingCompaction``:

    1. ``ClampOversizedMessages`` — trim any single oversized part to head+tail.
    2. ``ClearToolResults`` — replace tool-call/result pairs with placeholders.
    3. ``SlidingWindowCompaction`` — drop oldest messages, keep the recent window
       (plus the first user message) — the harness's cheap default.

    When ``Settings.consolidate`` (``CONSOLIDATE`` / `.env`) is set, the
    sliding-window tier is replaced by
    :class:`~paf.capabilities.memory._window.MemoryConsolidationWindow`, which
    folds the about-to-be-evicted slice into the durable memory store (via the
    writable owner sub-agent) *before* dropping it. Default (unset) keeps
    the byte-identical plain window.

    The LLM rewrite (4. ``SummarizingCompaction``) is intentionally absent — it
    is mutually exclusive with the turn-index one-liner block (both are summaries).
    Context is bounded here by pruning; long-term recall is the store-backed
    ``ConversationSearch`` / ``read_turn`` instead of a compacted summary.

    Values are small/observable for M1 so a few turns trip the escalation; raise
    them for a longer-lived context.
    """
    if settings.consolidate:
        from paf.capabilities.memory._window import MemoryConsolidationWindow

        window = MemoryConsolidationWindow(
            max_messages=4,
            keep_messages=2,
        )
    else:
        window = SlidingWindowCompaction(
            max_messages=4,
            keep_messages=2,
        )
    return TieredCompaction(
        target_tokens=1200,
        tiers=[
            # 1. Clamp any single oversized part (rare, but cheap when it fires).
            ClampOversizedMessages(
                max_part_tokens=50_000,
                keep_head_chars=2_000,
                keep_tail_chars=2_000,
            ),
            # 2. Wipe tool-call/result pairs (token-heavy, low value) to placeholders.
            ClearToolResults(
                max_tokens=1,
                keep_pairs=4,
            ),
            # 3. Drop the oldest messages (or consolidate-then-drop, when enabled).
            window,
        ],
    )
