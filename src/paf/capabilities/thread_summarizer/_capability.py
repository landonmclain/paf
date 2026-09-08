"""ThreadSummarizer capability: background thread-conversation summarizer.

A per-conversation background summarizer distinct from ``Memory``: while
``Memory`` is *cross*-conversation durable-fact extraction, the thread
summarizer is *intra*-conversation chronological beats (the eviction-blocks
record). ``Memory`` extracts and dedupes semantic facts; this capability
records *what this conversation was doing across evicted turns and how it
evolved* (episodic, ordered, `(run, turn)`-anchored ``beats``) — the axis
memory does not own.

Phase 0 — this module is the capability SHELL: registration + spec
serializability + ON/OFF via the profile allowlist. Capture (evictions WAL),
the background ``beats`` worker, and any in-context presentation land in
later phases (see the spec in the vault).
"""

from __future__ import annotations

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext


class ThreadSummarizer(AbstractCapability[AgentDepsT]):
    """Background per-conversation thread summarizer (eviction-blocks beats).

    Phase 0 shell only. No capture, no worker, no injection yet.

    ``before_model_request`` is currently a no-op: the in-context presentation
    surface (a second Memory-style injected block) is deliberately DEFERRED to
    Phase 2/3 pending real ``beats`` output on real evictions. Keep the hook
    present so the capability has a place to inject, but it must not mutate or
    accumulate until then.
    """

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """No-op in Phase 0 — presentation is deferred (Phase 2/3 decision)."""
        return request_context

    @classmethod
    def from_spec(cls, **kwargs: object) -> ThreadSummarizer[AgentDepsT]:
        """Construct a thread-summarizer capability from serializable options.

        Phase 0 accepts no options; the signature is reserved for the spec
        block (``- ThreadSummarizer: {...}``) once capture/worker config
        exists.
        """
        return cls()

    @classmethod
    def get_serialization_name(cls) -> str:
        """The name used by custom capability specs (the allowlist key)."""
        return "ThreadSummarizer"
