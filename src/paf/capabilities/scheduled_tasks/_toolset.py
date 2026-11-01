"""Scheduled-tasks toolset — the agent-facing surface.

Five tools over the per-conversation ``ScheduledTaskStore``: ``schedule_task``
(create), ``list_scheduled_tasks``, ``update_scheduled_task``,
``remove_scheduled_task`` (soft-delete/archive), and ``run_scheduled_task``
(manually enqueue a job now). The ``run`` tool needs the run-layer queue, which
is only wired in paf-sse; until then it reports the queue as unwired.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from paf.background import BackgroundQueueService

from ._store import ScheduledTaskStore
from ._types import NotifyMode, ScheduledJob


class ScheduledTasksToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent CRUD + manual-run over its conversation's scheduled tasks."""

    def __init__(
        self,
        *,
        store_factory: Callable[[str], ScheduledTaskStore],
        queue: BackgroundQueueService[ScheduledJob] | None,
        agent: str,
        conversation_id: str | None = None,
    ) -> None:
        super().__init__()
        self._store_factory = store_factory
        self._queue = queue
        self._agent = agent
        self._conversation_id = conversation_id
        self.add_function(self.schedule_task, name="schedule_task")
        self.add_function(self.list_scheduled_tasks, name="list_scheduled_tasks")
        self.add_function(self.update_scheduled_task, name="update_scheduled_task")
        self.add_function(self.remove_scheduled_task, name="remove_scheduled_task")
        self.add_function(self.run_scheduled_task, name="run_scheduled_task")

    def _store(self) -> ScheduledTaskStore:
        return self._store_factory(self._conversation_id or "global")

    async def schedule_task(
        self,
        schedule: str,
        name: str,
        description: str,
        instructions: str | None = None,
        script: str | None = None,
        notify: str = "always",
        delivery: str = "file",
    ) -> str:
        """Schedule a recurring task in this conversation.

        Provide exactly one of `instructions` (agent-executed: the agent follows
        the instruction block and does the work) or `script` (script-executed:
        a script runs code-only, skipping the agent unless X). `schedule` is a
        cron expression (e.g. "0 9 * * *" for 9am daily).

        Args:
            schedule: Cron expression for when the task fires.
            name: Human-readable name for the task (shown in the UI).
            description: What the task does / when it runs (shown in the UI).
            instructions: The instruction block (agent-executed task).
            script: A script to run code-only (script-executed task).
            notify: For script tasks, "always" (agent reviews every run) or
                "on_trigger" (agent only when the script signals).
            delivery: Delivery target (default "file").

        Returns:
            The new task's id and a summary.
        """
        if (instructions is None) == (script is None):
            raise ModelRetry(
                "Provide exactly one of `instructions` or `script` for the task."
            )
        try:
            mode = NotifyMode(notify)
        except ValueError:
            raise ModelRetry(
                f"`notify` must be 'always' or 'on_trigger', got {notify!r}."
            ) from None
        job = ScheduledJob(
            schedule=schedule,
            name=name,
            description=description,
            instructions=instructions,
            script=script,
            notify=mode,
            delivery=delivery,
            agent=self._agent,
            conversation_id=self._conversation_id or "global",
        )
        created = self._store().create_job(job)
        return (
            f"Scheduled task {created.id} created: {created.name!r} — "
            f"schedule={created.schedule!r}, "
            f"mode={'agent' if created.instructions else 'script'}, "
            f"notify={created.notify.value}"
        )

    async def list_scheduled_tasks(self) -> str:
        """List this conversation's scheduled tasks (active ones)."""
        jobs = self._store().list_jobs()
        if not jobs:
            return "No scheduled tasks in this conversation."
        lines = []
        for j in jobs:
            mode = "agent" if j.instructions else "script"
            state = "enabled" if j.enabled else "paused"
            lines.append(
                f"- {j.id}: {j.name!r} — {j.description!r} "
                f"({j.schedule!r}, {mode}, {state}, notify={j.notify.value})"
            )
        return "\n".join(lines)

    async def update_scheduled_task(
        self,
        job_id: str,
        schedule: str | None = None,
        name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        script: str | None = None,
        notify: str | None = None,
        delivery: str | None = None,
        enabled: bool | None = None,
    ) -> str:
        """Update a scheduled task's fields. Only the provided fields change.

        Setting `instructions` switches the task to agent-executed (clearing any
        script); setting `script` switches it to script-executed (clearing any
        instructions).

        Args:
            job_id: The task's id.
            schedule: New cron expression.
            name: New human-readable name.
            description: New description of what the task does / when it runs.
            instructions: New instruction block (agent-executed).
            script: New script (script-executed).
            notify: "always" or "on_trigger".
            delivery: New delivery target.
            enabled: Whether the task is active.

        Returns:
            A summary of the updated task.
        """
        store = self._store()
        if store.get_job(job_id) is None:
            raise ModelRetry(f"No scheduled task with id {job_id!r}.")
        if instructions is not None and script is not None:
            raise ModelRetry("Provide `instructions` or `script`, not both.")
        mode = None
        if notify is not None:
            try:
                mode = NotifyMode(notify)
            except ValueError:
                raise ModelRetry(
                    f"`notify` must be 'always' or 'on_trigger', got {notify!r}."
                ) from None
        if instructions is not None or script is not None:
            store.set_definition(job_id, instructions=instructions, script=script)
        updated = store.update_job(
            job_id,
            schedule=schedule,
            name=name,
            description=description,
            notify=mode,
            delivery=delivery,
            enabled=enabled,
        )
        if updated is None:
            raise ModelRetry(f"No scheduled task with id {job_id!r}.")
        return (
            f"Updated scheduled task {job_id}: {updated.name!r} — "
            f"schedule={updated.schedule!r}, "
            f"mode={'agent' if updated.instructions else 'script'}, "
            f"enabled={updated.enabled}"
        )

    async def remove_scheduled_task(self, job_id: str) -> str:
        """Remove (archive) a scheduled task. It stops firing but its history stays."""
        if not self._store().remove_job(job_id):
            raise ModelRetry(f"No scheduled task with id {job_id!r}.")
        return f"Removed scheduled task {job_id}."

    async def run_scheduled_task(self, job_id: str) -> str:
        """Run a scheduled task now, outside its schedule."""
        store = self._store()
        job = store.get_job(job_id)
        if job is None:
            raise ModelRetry(f"No scheduled task with id {job_id!r}.")
        if self._queue is None:
            raise ModelRetry(
                "The scheduled-tasks run queue is not wired in this deployment; "
                "cannot run the task now."
            )
        self._queue.enqueue(self._conversation_id or "global", job)
        return f"Queued scheduled task {job_id} to run now."
