"""Per-conversation thread summarizer — the ``ThreadSummarizer`` capability.

Distinct from ``Memory`` (cross-conversation durable-fact extraction): the thread
summarizer is **intra-conversation chronological beats** over eviction blocks —
\"what this conversation was doing and how it evolved,\" ordered and
`(run, turn)`-anchored, so it does not duplicate memory's semantic axis.

Phase 0 ships the capability shell (registration + spec serializability + OFF by
default via the profile allowlist). Capture (evictions WAL), the background
``beats`` worker, and any in-context presentation land in later phases.
"""

from __future__ import annotations

from ._capability import ThreadSummarizer

__all__ = ["ThreadSummarizer"]
