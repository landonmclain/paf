"""Scheduled-tasks capability (Phase 4): the agent-facing toolset.

Wraps the per-conversation ``ScheduledTaskStore`` behind a five-tool toolset
(``schedule_task`` / ``list_scheduled_tasks`` / ``update_scheduled_task`` /
``remove_scheduled_task`` / ``run_scheduled_task``). ``root`` is the
scheduled-tasks data root, ``agent`` scopes it, and ``queue`` is the run-layer
queue (wired in paf-sse) that ``run_scheduled_task`` enqueues into. When
``queue`` is None the ``run`` tool reports the queue as unwired.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from paf.background import BackgroundQueueService

from ._store import ScheduledTaskStore
from ._toolset import ScheduledTasksToolset
from ._types import ScheduledJob


@dataclass
class ScheduledTasks(AbstractCapability[AgentDepsT]):
    """Scheduled tasks for an agent: CRUD + manual run over the conversation store."""

    root: str | Path = "data/scheduled_tasks"
    agent: str = ""
    queue: BackgroundQueueService[ScheduledJob] | None = None

    def _store_factory(self) -> Callable[[str], ScheduledTaskStore]:
        def factory(cid: str) -> ScheduledTaskStore:
            del cid  # the store is per-agent; conversation is a column on the job
            return ScheduledTaskStore(self.root, agent=self.agent)

        return factory

    async def for_run(self, ctx) -> ScheduledTasksToolset[AgentDepsT]:
        """Return a fresh toolset per run, scoped to the run's conversation."""
        return ScheduledTasksToolset[AgentDepsT](
            store_factory=self._store_factory(),
            queue=self.queue,
            agent=self.agent,
            conversation_id=getattr(ctx, "conversation_id", None),
        )

    @classmethod
    def get_serialization_name(cls) -> str:
        """The name used by custom capability specs (the allowlist key)."""
        return "ScheduledTasks"
