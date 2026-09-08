"""Per-conversation thread summarizer — the ``ThreadSummarizer`` capability.

Distinct from ``Memory`` (cross-conversation durable-fact extraction): the thread
summarizer is **intra-conversation chronological beats** over eviction blocks —
\"what this conversation was doing and how it evolved,\" ordered and
`(run, turn)`-anchored, so it does not duplicate memory's semantic axis.

Phase 1 ships eviction **capture**: each ``TurnBoundSlidingWindow`` eviction is
appended as an immutable, ``conversation_id``-tagged ``EvictionBlock`` to the
per-agent WAL (``data/summaries/<agent>/evictions.jsonl``). The background
``beats`` worker and any in-context presentation land in Phases 2-3.
"""

from __future__ import annotations

from ._capability import ThreadSummarizer, capture_directory_for

__all__ = ["ThreadSummarizer", "capture_directory_for"]
