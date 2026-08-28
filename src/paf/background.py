"""paf background work-queue engine — out-of-band jobs, one queue per type.

A framework-free engine that runs jobs *outside* a model turn: a caller like the
memory consolidation capability enqueues work and returns immediately; an
asyncio worker drains the queue in the background. The model never calls this
directly and nothing is injected by it — it is **not** a pydantic-ai
capability and intentionally has no ``get_toolset()`` / injection seam. It
lives at the framework level (next to ``config.py`` / ``runtime.py``), not
under ``capabilities/``, because it is shared infrastructure that multiple
consumer capabilities call *into*.

Only the engine lives here. coding_agent's consumer job types
(``WorklogJob`` / ``CompactionMemoryJob``) and their prompts are deliberately
NOT ported — paf consumers define their own job dataclasses and subclass
:class:`BackgroundQueueService`, implementing :meth:`process_job`.

Parallelism model: one :class:`BackgroundQueueService` instance == one queue
== one worker per session, FIFO. Within a queue jobs are ordered (correct for
an append-only writer like memory — you don't want two consolidations
stomping the journal concurrently). Independent reasons run *at the same time*
by having **separate instances** (e.g. a memory queue and a future
summarization queue) whose workers interleave across the event loop. There is
no worker pool and no dependency scheduler — "job B waits on job A" is
expressed as ordered steps inside a single :meth:`process_job`.

Async-first: paf's entry points are async, so the daemon-thread
``BackgroundServiceManager`` coding_agent used is unnecessary here. A worker
idle-exits after :attr:`IDLE_TIMEOUT_S` and is transparently re-spawned by the
next :meth:`enqueue`; callers must NOT assume a warm worker.

Ported from coding_agent ``app/services/background/`` (``base.py`` +
``queues.py``); kept line-faithful. Two deliberate deltas: ``QueueState``
carries ``active_jobs`` (the honest ``running`` status) and ``QueueStatus`` is
included (both were in coding_agent's ``queues.py``), and the sync-thread
bridge is dropped as unnecessary in an all-async process.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class QueueState[JobT]:
    """Runtime state for one session-specific background queue."""

    queue: asyncio.Queue[JobT] = field(default_factory=asyncio.Queue)
    worker_task: asyncio.Task | None = None
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_jobs: int = 0


@dataclass
class QueueStatus:
    """A point-in-time snapshot of one session's queue."""

    name: str
    queued: int
    running: bool
    idle: bool


class BackgroundQueueService[JobT](ABC):
    """Base class for one background queue type.

    Example paf subclasses:
    - ``MemoryConsolidationQueue`` (consumer #1; enqueued on compaction /
      trigger events so the owner runs out-of-band even when eviction doesn't).
    - a future summarization queue (summarizer fires on eviction).

    One instance == one queue type == one worker per session. The worker drains
    FIFO; it idle-exits after :attr:`IDLE_TIMEOUT_S` and :meth:`enqueue`
    transparently re-spawns it.
    """

    name: str = "background"

    #: Seconds a worker waits for the next job before idle-exiting. Callers
    #: must not assume a warm worker — enqueue() spawns one as needed.
    IDLE_TIMEOUT_S: float = 10.0

    def __init__(self) -> None:
        self._states: dict[str, QueueState[JobT]] = {}

    def get_state(self, session_id: str) -> QueueState[JobT]:
        if session_id not in self._states:
            self._states[session_id] = QueueState()
        return self._states[session_id]

    def enqueue(self, session_id: str, job: JobT) -> None:
        state = self.get_state(session_id)
        state.queue.put_nowait(job)

        if state.worker_task is None or state.worker_task.done():
            state.worker_task = asyncio.create_task(
                self._worker(session_id),
                name=f"{self.name}:{session_id}",
            )

    async def drain(self, session_id: str) -> None:
        state = self.get_state(session_id)
        await state.queue.join()

    async def drain_with_timeout(self, session_id: str, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self.drain(session_id), timeout=timeout)
            return True
        except TimeoutError:
            return False

    def status(self, session_id: str) -> QueueStatus:
        state = self.get_state(session_id)
        running = state.worker_task is not None and not state.worker_task.done()
        return QueueStatus(
            name=self.name,
            queued=state.queue.qsize(),
            running=running,
            idle=state.active_jobs == 0 and not running,
        )

    async def _worker(self, session_id: str) -> None:
        state = self.get_state(session_id)

        while True:
            try:
                job = await asyncio.wait_for(
                    state.queue.get(), timeout=self.IDLE_TIMEOUT_S
                )
            except TimeoutError:
                return

            state.active_jobs += 1
            try:
                await self.process_job(session_id, job, state)
            except Exception:
                log.exception(
                    "%s_job_failed session=%s",
                    self.name,
                    session_id,
                )
            finally:
                state.active_jobs -= 1
                state.queue.task_done()

    @abstractmethod
    async def process_job(
        self,
        session_id: str,
        job: JobT,
        state: QueueState[JobT],
    ) -> None:
        raise NotImplementedError
