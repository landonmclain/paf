"""Per-conversation thread summarizer — the ``ThreadSummarizer`` capability.

Distinct from ``Memory`` (cross-conversation durable-fact extraction): the thread
summarizer is **intra-conversation chronological beats** — *what this
conversation was doing and how it evolved*, ordered and `(run, turn)`-anchored,
so it does not duplicate memory's semantic axis.

The summarizer is an independent, separately-enabled capability driven by the
same cadence + watermark machinery as Memory's owner, but keyed by
**conversation** (``{profile}:{cid}``), not agent. The runtime trigger resolves
the unsummarized turn-span directly from the persisted step store
(``resolve_unseen_span(conversation_id=...)``), the worker summarizes the
curated user-prompt + final-answer span, and advances a per-conversation
``MemoryWatermark`` on success. There is no eviction WAL.
"""

from __future__ import annotations

from ._capability import ThreadSummarizer, capture_directory_for

__all__ = ["ThreadSummarizer", "capture_directory_for"]
