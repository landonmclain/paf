"""ThreadSummarizer capability: int*-conversation* thread-beat summarizer.

Distinct from ``Memory``: ``Memory`` is *cross*-conversation durable-fact
extraction, deduped into a shared journal; the thread summarizer is
*intra*-conversation chronological beats — *what this conversation was doing
across turns and how it evolved*, ``(run, turn)``-anchored and per-conversation.
It is an independent, separately-enabled capability so a profile can run thread
beats without Memory (or vice-versa) — it is its own thing with its own
per-conversation watermarks under ``data/summaries/<agent>/conversations/``.

The summarizer is driven by the SAME cadence + watermark machinery as Memory's
owner, but keyed by conversation. There is no eviction WAL here: the trigger
resolves the unsummarized turn-span directly from the persisted step store
(``resolve_unseen_span(conversation_id=...)``), the worker summarizes the
curated user-prompt + final-answer span, and advances a per-conversation
``MemoryWatermark`` on success. See the spec's re-architecture note (2026-09-08).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext

#: Default capture root (name matches memory's per-agent convention).
_DEFAULT_CAPTURE_ROOT = "data/summaries"


def capture_directory_for(agent_name: str | None) -> Path:
    """Per-agent capture dir: ``<CWD>/data/summaries/<agent_name>/``.

    Mirrors ``memory_dir_for``/``DEFAULT_MEMORY_DIR`` (``data/memory/<agent>``).
    The thread summarizer keeps its derived summaries + per-conversation
    watermarks below this root, under ``conversations/<hash(cid)>/``.
    """
    return Path(_DEFAULT_CAPTURE_ROOT) / (agent_name or "agent")


@dataclass(init=False)
class ThreadSummarizer(AbstractCapability[AgentDepsT]):
    """Background per-conversation thread summarizer (on/off surface).

    ``@dataclass(init=False)`` is REQUIRED, not cosmetic: pydantic-ai's spec
    loader validates that every custom capability class carries
    ``__dataclass_fields__`` (see ``get_capability_registry``'s
    ``_validate_capability``), and ``init=False`` keeps this type's hand-written
    ``__init__`` intact.

    Phase 1/2 rework (2026-09-08): the capture WAL is gone. This capability is
    the allowlist/on-off surface; the runtime trigger + worker own the cadence
    and per-conversation watermarks.
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
