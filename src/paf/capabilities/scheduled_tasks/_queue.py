"""Scheduled-task run layer — the queue that executes due jobs (Phase 3).

A :class:`BackgroundQueueService` subclass (mirroring ``ShellBackgroundQueue``)
that consumes the jobs the scheduler (Phase 2) enqueues. ``process_job`` runs
each job — an agent turn (``instructions``) or a script (``script``) — appends
a minimal ``RunRecord`` to the run history, and feeds ``ProcessEvent``s for
Processes-panel visibility.

The agent turn runs via the ``run_turn`` seam (wired in paf-sse to
``Session.notify``, the steer gate — follows shell exactly: it steers into a
live turn or starts a new one). The verdict / cost / finished_at fields are
deferred to Phase 5 (delivery); Phase 3 records only what's known at fire time.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import uuid4

from paf.background import BackgroundQueueService, QueueState
from paf.background_processes._events import (
    ProcessEvent,
    ProcessEventType,
    get_agent_emitter,
)
from paf.capabilities._sandbox import SandboxedExecutor
from paf.clock import now_local

from ._schedule import next_fire
from ._store import ScheduledTaskStore
from ._types import NotifyMode, RunRecord, ScheduledJob

log = logging.getLogger(__name__)

#: Default script timeout (seconds).
_SCRIPT_TIMEOUT_S = 60.0


class ScheduledTaskQueue(BackgroundQueueService[ScheduledJob]):
    """Runs scheduled jobs: agent turn (``instructions``) or script (``script``).

    A single queue shared across agents. ``store_factory(agent)`` returns the
    agent's ``ScheduledTaskStore`` (to append run records); ``run_turn(agent,
    cid, content)`` runs an agent turn in that conversation (wired to
    ``Session.notify`` in paf-sse); ``executor`` runs script-executed jobs. The
    job carries its ``agent`` + ``conversation_id``, so the queue routes by them.
    """

    name = "scheduled-tasks"

    def __init__(
        self,
        *,
        store_factory: Callable[[str], ScheduledTaskStore],
        run_turn: Callable[[str, str, str, str], Awaitable[None]],
        executor: SandboxedExecutor,
        cwd: str | Path | None = None,
        on_complete=None,
    ) -> None:
        super().__init__()
        self.store_factory = store_factory
        self.run_turn = run_turn
        self.executor = executor
        self.cwd = Path(cwd).resolve() if cwd else None
        self.on_complete = on_complete
        self._job_cid: dict[str, str | None] = {}

    async def _emit(
        self, agent: str, job_id: str, status: str, label: str, detail: str = ""
    ) -> None:
        """Push one live process event to the job's agent emitter.

        Scheduled tasks are conversation-scoped by origin, so the event carries
        ``lane="conversation"`` and the job's conversation id as ``scope``.
        """
        await get_agent_emitter(agent).emit_process(
            ProcessEvent(
                id=job_id,
                label=label,
                status=ProcessEventType(status),
                detail=detail,
                lane="conversation",
                scope=self._job_cid.get(job_id),
            )
        )

    async def process_job(
        self,
        session_id: str,
        job: ScheduledJob,
        state: QueueState[ScheduledJob],
    ) -> None:
        del state
        self._job_cid[job.id] = session_id
        store = self.store_factory(job.agent)
        run_id = uuid4().hex[:8]
        store.append_run(
            RunRecord(
                run_id=run_id,
                job_id=job.id,
                agent=job.agent,
                conversation_id=session_id,
                status="running",
            )
        )
        try:
            if job.script is not None:
                await self._run_script(session_id, job, job.script, run_id)
            else:
                await self._run_agent(session_id, job, run_id)
            store.append_run(
                RunRecord(
                    run_id=run_id,
                    job_id=job.id,
                    agent=job.agent,
                    conversation_id=session_id,
                    status="done",
                )
            )
        except Exception as exc:
            store.append_run(
                RunRecord(
                    run_id=run_id,
                    job_id=job.id,
                    agent=job.agent,
                    conversation_id=session_id,
                    status="error",
                    detail=str(exc),
                )
            )
            await self._emit(
                job.agent, job.id, "error", job.name, str(exc)
            )
            raise

    def _done_detail(self, job: ScheduledJob) -> str:
        """Human detail for a finished run: when it ran, when it runs next, and what it does.

        Uses wall-clock times (not relative "ago") so the row reads as a status
        line; the description is the job's own ``description`` field.
        """
        last = job.last_run
        last_str = last.strftime("%-I:%M%p").lower() if last else "never"
        try:
            nxt = next_fire(job.schedule, now_local())
            next_str = nxt.strftime("%-I:%M%p").lower()
        except ValueError:
            next_str = "?"
        return f"Last ran {last_str}. Next run {next_str}. {job.description}"

    async def _run_agent(self, session_id: str, job: ScheduledJob, run_id: str) -> None:
        """Run an agent-executed job: a full turn via the ``run_turn`` seam."""
        await self._emit(
            job.agent, job.id, "running", job.name, "agent turn"
        )
        await self.run_turn(job.agent, session_id, job.instructions or "", run_id)
        await self._emit(
            job.agent, job.id, "done", job.name, self._done_detail(job)
        )

    async def _run_script(
        self, session_id: str, job: ScheduledJob, script: str, run_id: str
    ) -> None:
        """Run a script-executed job; notify per ``notify`` mode."""
        await self._emit(job.agent, job.id, "running", job.name, "script")
        result = await self.executor.run(
            script,
            cwd=self.cwd or Path("."),
            env=None,
            timeout=_SCRIPT_TIMEOUT_S,
        )
        ok = result.exit_code == 0
        if job.notify == NotifyMode.always or (
            job.notify == NotifyMode.on_trigger and not ok
        ):
            await self._notify(session_id, job, result, run_id)
        await self._emit(
            job.agent,
            job.id,
            "done" if ok else "error",
            job.name,
            self._done_detail(job),
        )

    async def _notify(
        self, session_id: str, job: ScheduledJob, result, run_id: str
    ) -> None:
        """Deliver the script result to the conversation (best-effort)."""
        try:
            detail = (result.stdout or "").strip() or (result.stderr or "").strip()
            await self.run_turn(
                job.agent,
                session_id,
                f"[system] Scheduled task {job.id} finished with exit code "
                f"{result.exit_code}.\n\n{detail}",
                run_id,
            )
        except Exception:  # noqa: BLE001 - notify is best-effort
            log.exception("scheduled notify failed for %s", job.id)
