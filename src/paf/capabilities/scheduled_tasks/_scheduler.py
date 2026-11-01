"""Scheduled-task scheduler — the generic time-triggered producer.

A per-agent background loop that fires due scheduled jobs by enqueuing them
into a :class:`BackgroundQueueService` (the run layer, which consumes them).
It mirrors the shell background queue's shape: one instance per agent, with
per-conversation state handled internally.

Design (see the vault spec ``Scheduled-Tasks-Cron-Spec.md`` §4.7):

- **Sleep-until-next-due**, not a fixed-interval poll: the loop computes the
  earliest next fire across every enabled, non-archived job in every
  conversation, sleeps until it (bounded by ``cap`` as a backstop), and fires
  the jobs that come due.
- **Wake-on-change**: each conversation store's ``on_change`` callback sets a
  shared :class:`asyncio.Event`, so a job created/edited/removed mid-sleep wakes
  the loop immediately instead of waiting for the next fire.
- **Single-flight**: ``next_fire`` is always strictly after ``now`` (croniter),
  so a job is never fired twice in one pass; after firing, its next fire is in
  the future.
- **Restart safety**: ``last_run`` is the only persisted watermark, updated on
  each fire. On restart, next-fire is recomputed from ``now``, so a job never
  fires twice; a fire missed during downtime is skipped, not caught up.

The scheduler is **synchronous over the sync stores** (like the shell toolset):
it calls the store methods directly from the async loop. The async layer is the
loop itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from paf.background import BackgroundQueueService
from paf.clock import now_local

from ._schedule import next_fire
from ._store import ScheduledTaskStore
from ._types import ScheduledJob

#: Default backstop on the sleep: the loop re-checks at least this often even
#: if no change event fires, so a missed change is never waited out forever.
DEFAULT_CAP_S = 60.0


class ScheduledTaskScheduler:
    """Per-agent loop that fires due scheduled jobs into a background queue.

    ``root`` is the scheduled-tasks data root (``data/scheduled_tasks``) and
    ``agent`` scopes it to
    one agent's conversations (``<root>/<agent>/conversations/<cid>/``). ``queue``
    is the :class:`BackgroundQueueService` the run layer consumes;
    the scheduler only ever calls ``queue.enqueue(cid, job)``.
    """

    def __init__(
        self,
        root: str | Path = "data/scheduled_tasks",
        *,
        agent: str | None = None,
        queue: BackgroundQueueService[ScheduledJob],
        clock: Callable[[], datetime] = now_local,
        cap: float = DEFAULT_CAP_S,
    ) -> None:
        self._root = Path(root)
        self._agent = agent
        self._queue = queue
        self._clock = clock
        self._cap = cap
        self._stores: dict[str, ScheduledTaskStore] = {}
        self._change = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Spawn the run loop as a background task (idempotent)."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        """Cancel the run loop and await its exit."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # --- change signal -----------------------------------------------------

    def _wake(self) -> None:
        """Store ``on_change`` bridge: any mutation wakes the loop."""
        self._change.set()

    def _agents(self) -> list[str]:
        """The agents whose master lists this loop scans.

        ``agent`` scopes it to one agent; ``None`` scans every agent's master
        list under ``root`` (the single-loop, database-table shape).
        """
        if self._agent:
            return [self._agent]
        if not self._root.is_dir():
            return []
        return sorted(d.name for d in self._root.iterdir() if d.is_dir())

    def _store_for(self, agent: str) -> ScheduledTaskStore:
        store = self._stores.get(agent)
        if store is None:
            store = ScheduledTaskStore(
                self._root,
                agent=agent,
                on_change=self._wake,
            )
            self._stores[agent] = store
        return store

    # --- schedule computation ---------------------------------------------

    def _next_fires(self, now: datetime) -> list[tuple[str, ScheduledJob, datetime]]:
        """``(agent, job, next_fire)`` for every enabled, non-archived job."""
        fires: list[tuple[str, ScheduledJob, datetime]] = []
        for agent in self._agents():
            store = self._store_for(agent)
            for job in store.list_jobs():
                if not job.enabled:
                    continue
                try:
                    fire_dt = next_fire(job.schedule, now)
                except ValueError:
                    continue  # invalid schedule: skip, don't crash the loop
                fires.append((agent, job, fire_dt))
        return fires

    def _fire(self, agent: str, job: ScheduledJob) -> None:
        """Enqueue `job` into its origin conversation's queue and stamp ``last_run``.

        Re-reads the job first so a job disabled/archived while the loop slept
        is not fired. The queue routes by the job's ``agent`` + ``conversation_id``.
        """
        store = self._store_for(agent)
        current = store.get_job(job.id)
        if current is None or not current.enabled or current.archived:
            return
        # Stamp last_run BEFORE enqueueing so the job the queue runs carries the
        # CURRENT fire time (the run's own "last ran"), not the previous fire's.
        updated = store.update_job(job.id, last_run=self._clock())
        if updated is None:
            return
        self._queue.enqueue(updated.conversation_id, updated)

    # --- the loop ----------------------------------------------------------

    async def run(self) -> None:
        """Sleep-until-next-due loop; runs until cancelled."""
        while True:
            self._change.clear()
            now = self._clock()
            fires = self._next_fires(now)
            if fires:
                target = min(fire_dt for _, _, fire_dt in fires)
                delay = min(max((target - now).total_seconds(), 0.0), self._cap)
            else:
                delay = self._cap
            await self._sleep(delay)
            now2 = self._clock()
            for agent, job, fire_dt in fires:
                if fire_dt <= now2:
                    self._fire(agent, job)

    async def _sleep(self, delay: float) -> None:
        """Wait up to `delay` seconds, returning early when a store changes.

        The change event is the primary wake (a job created/edited/removed
        mid-sleep); the timeout is the ``cap`` backstop so a missed change is
        never waited out forever.
        """
        try:
            await asyncio.wait_for(self._change.wait(), timeout=delay)
        except TimeoutError:
            pass
