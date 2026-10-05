"""LongMemEval seed-time thread-summarizer helper (benchmark harness only).

Mirrors ``tests/_lmem_consolidate.py`` (the extractor/owner seed-time helper):
it folds a *given* conversation into per-conversation thread-beat summaries via
the thread summarizer, so the benchmark subject gets the same ``<thread-summary>``
injection + ``list_turn_summaries`` navigation tool a live conversation would.

The live worker (:class:`paf.capabilities.thread_summarizer._background
.ThreadSummarizerQueue`) is driven by a cadence + per-conversation watermark over
a live conversation; this seed-time variant runs the same fold over the ENTIRE
seeded haystack in one call, exactly like ``consolidate_conversation_chunked``
runs the memory owner over the whole haystack. Keeping it here (not in the
production module) matches the rule that agent model/settings/guidance come from
specs and the harness constants stay hardcoded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

logger = logging.getLogger(__name__)


async def summarize_conversation(
    messages: list[ModelMessage],
    *,
    capture_root: str | Path,
    conversation_id: str,
    agent_name: str = "assistant",
) -> int:
    """Summarize a full given conversation into per-conversation thread beats.

    Seed-time variant of the thread-summarizer worker: slices the ENTIRE
    ``messages`` list into ~10-turn blocks and runs the structured summarizer
    once per block, appending each to a ``ThreadSummaryStore`` at
    ``capture_root/conversations/<hash(conversation_id)>/``. Mirrors
    ``consolidate_conversation_chunked`` (the extractor/owner invocation).

    Unlike the live worker, the trailing partial block (< ``BLOCK_TURNS`` turns)
    IS folded: the benchmark haystack is static (no later turns will accumulate),
    so a 1-9 turn tail would otherwise never be summarized. Returns the number of
    blocks summarized.
    """
    from paf.capabilities.thread_summarizer._background import (
        BLOCK_TURNS,
        _block_anchor_from_refs,
        _distinct_complete_turns,
        _span_range,
        summary_from_beats,
    )
    from paf.capabilities.thread_summarizer._store import ThreadSummaryStore
    from paf.capabilities.thread_summarizer._summarize import _ref_for, summarize_span

    turns, turn_anchors = _distinct_complete_turns(messages)
    if not turns:
        return 0
    # Conversation-absolute per-turn refs across the WHOLE span (each ref's
    # `turn` is its position in conversation history, not within a slice).
    turn_refs = [_ref_for(messages, a) for a in turn_anchors]
    span_runs = [getattr(m, "run_id", None) for m in messages]
    store = ThreadSummaryStore(capture_root, conversation_id)
    n_blocks = (len(turns) + BLOCK_TURNS - 1) // BLOCK_TURNS
    n = 0
    for start in range(0, len(turns), BLOCK_TURNS):
        block = turns[start : start + BLOCK_TURNS]
        # Stamp each turn's conversation-absolute ref onto its messages so the
        # prompt/beats cite the real {run_id}:{turn_id}, not a slice-relative one.
        block_msgs: list = []
        for k, turn in enumerate(block):
            abs_ref = turn_refs[start + k]
            for m in turn:
                m._thread_abs_ref = abs_ref
            block_msgs.extend(turn)
        block_refs = [r for r in turn_refs[start : start + len(block)] if r is not None]
        block_anchor = _block_anchor_from_refs(span_runs, block_refs)
        beats = await summarize_span(block_msgs, conversation_id)
        block_start, block_end = _span_range(block)
        store.append_summary(
            summary_from_beats(
                beats,
                conversation_id,
                block_anchor,
                block_refs,
                started_at=block_start,
                ended_at=block_end,
            )
        )
        n += 1
        logger.info(
            "summarized block %d/%d (%d turns) of %d msgs for %s",
            n,
            n_blocks,
            len(block),
            len(messages),
            conversation_id,
        )
    return n
