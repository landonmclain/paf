"""ThreadSummarizer capability: background thread-conversation summarizer.

A per-conversation background summarizer distinct from ``Memory``: while
``Memory`` is *cross*-conversation durable-fact extraction, the thread
summarizer is *intra*-conversation chronological beats (the eviction-blocks
record). ``Memory`` extracts and dedupes semantic facts; this capability
records *what this conversation was doing across evicted turns and how it
evolved* (episodic, ordered, `(run, turn)`-anchored ``beats``) — the axis
memory does not own.

Phase 1 — this module owns the **capture** side: it learns the active thread
from ``ctx.conversation_id`` each turn and exposes a capture hook that appends
an immutable ``EvictionBlock`` per eviction to
``data/summaries/<agent>/evictions.jsonl`` (see :mod:`._capture`). The
background ``beats`` worker (Phase 2) and any in-context presentation (Phase 3)
build on this WAL.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext

from paf.capabilities.thread_summarizer._capture import (
    CapturedMessage,
    EvictionBlock,
    append_eviction_block,
    next_sequence,
    now_iso,
)

#: Default capture root (name matches memory's per-agent convention).
_DEFAULT_CAPTURE_ROOT = "data/summaries"


def capture_directory_for(agent_name: str | None) -> Path:
    """Per-agent capture dir: ``<CWD>/data/summaries/<agent_name>/``.

    Mirrors ``memory_dir_for``/``DEFAULT_MEMORY_DIR`` (``data/memory/<agent>/``):
    the capture WAL is scoped per agent, with the thread disambiguated by each
    block's explicit ``conversation_id``.
    """
    return Path(_DEFAULT_CAPTURE_ROOT) / (agent_name or "agent")


class ThreadSummarizer(AbstractCapability[AgentDepsT]):
    """Background per-conversation thread summarizer (eviction-blocks beats).

    Phase 1: captures each eviction into the append-only evictions WAL, tagged
    with the active thread's ``conversation_id`` (learned from ``ctx`` each
    turn). Presentation/injection is still deferred — ``before_model_request``
    does no context mutation, it only records the thread id.
    """

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        capture_directory: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._capture_directory = (
            Path(capture_directory) if capture_directory is not None else capture_directory_for(agent_name)
        )
        self._conversation_id: str | None = None

    def attach_to_window(self, window: object) -> None:
        """Bind this capability's capture hook to a ``TurnBoundSlidingWindow``.

        Enables capture (OFF by default: only called when the profile
        allowlists ThreadSummarizer AND Compaction).
        """
        from paf.capabilities._window import (
            TurnBoundSlidingWindow,  # local: no hard dep
        )

        if not isinstance(window, TurnBoundSlidingWindow):
            return  # not our window — nothing to attach in Phase 1
        window.capture_hook = self._capture

    @property
    def capture_directory(self) -> Path:
        return self._capture_directory

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Record the active thread; no context mutation (presentation is Phase 3)."""
        self._conversation_id = ctx.conversation_id
        return request_context

    def _capture(self, rows: list[dict], keep_messages: int) -> None:
        """Append one immutable EvictionBlock for an eviction (deterministic)."""
        block = EvictionBlock(
            seq=next_sequence(self._capture_directory),
            conversation_id=self._conversation_id,
            evicted_at=now_iso(),
            keep_messages=keep_messages,
            messages=[CapturedMessage(**r) for r in rows],
        )
        append_eviction_block(self._capture_directory, block)

    @classmethod
    def from_spec(cls, **kwargs: object) -> ThreadSummarizer[AgentDepsT]:
        """Construct from serializable options (``agent_name`` / ``capture_directory``)."""
        agent_name = kwargs.get("agent_name")
        capture_directory = kwargs.get("capture_directory")
        return cls(
            agent_name=str(agent_name) if agent_name is not None else None,
            capture_directory=str(capture_directory) if capture_directory is not None else None,
        )

    @classmethod
    def get_serialization_name(cls) -> str:
        """The name used by custom capability specs (the allowlist key)."""
        return "ThreadSummarizer"
