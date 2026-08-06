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

This module owns that assembly. ``build_compaction()`` builds the full ``TieredCompaction``
ladder; ``build_runtime_capabilities()`` (in ``build.py``) hands it to ``build_agent``,
which passes it via the ``capabilities=`` kwarg to ``Agent.from_spec`` (which MERGES it
with any spec-declared capabilities, ``agent/__init__.py``). Spec-declarable capabilities
that DO override ``from_spec`` (``StepPersistence``) stay in the YAML.
"""

from __future__ import annotations

from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
)


def build_compaction() -> TieredCompaction:
    """Assemble the assistant's context-compaction capability.

    Cheap tiers first, expensive last — ``TieredCompaction`` runs them in order
    and escalates only while the history still exceeds the budget (it stops once
    the history fits ``target_fraction`` of the model's context window). Ladder
    (least → most destructive / expensive):

    1. ``ClampOversizedMessages`` — trim any single oversized part to head+tail.
    2. ``ClearToolResults`` — replace tool-call/result pairs with placeholders.
    3. ``SlidingWindowCompaction`` — drop oldest messages, keep the recent window
       (plus the first user message) — the harness's cheap default.
    4. ``SummarizingCompaction`` — the LLM rewrite; only reached if still over
       budget. Inherits the assistant's model when ``model`` is omitted.

    Values are small/observable for M1 so a few turns trip the escalation; raise
    them for a longer-lived context.
    """
    return TieredCompaction(
        target_fraction=0.75,
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
            # 3. Drop the oldest messages, keep the recent window + first user msg.
            SlidingWindowCompaction(
                max_messages=4,
                keep_messages=2,
            ),
            # 4. LLM rewrite — only if still over budget after the cheap tiers.
            SummarizingCompaction(
                max_messages=1,
                keep_messages=20,
            ),
        ],
    )
