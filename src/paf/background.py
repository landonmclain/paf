"""paf background work-queue engine — out-of-band jobs, one queue per type.

A framework-free engine that runs jobs *outside* a model turn: a caller like the
memory consolidation capability enqueues work and returns immediately; an
asyncio worker drains the queue in the background. The model never calls this
directly and nothing is injected by it — it is **not** a pydantic-ai
capability and intentionally has no ``get_toolset()`` / injection seam. It
lives at the framework level (next to ``config.py`` / ``runtime.py``), not
under ``capabilities/``, because it is shared infrastructure that multiple
consumer capabilities call *into*.

Only the engine lives here. Consumers define their own job dataclasses and
subclass :class:`BackgroundQueueService`, implementing :meth:`process_job`.

Parallelism model: one :class:`BackgroundQueueService` instance == one queue
== one worker per session, FIFO. Within a queue jobs are ordered (correct for
an append-only writer like memory — you don't want two consolidations
stomping the journal concurrently). Independent reasons run *at the same time*
by having **separate instances** (e.g. a memory queue and a future
summarization queue) whose workers interleave across the event loop. There is
no worker pool and no dependency scheduler — "job B waits on job A" is
expressed as ordered steps inside a single :meth:`process_job`.

Async-first: paf's entry points are async, so no daemon-thread bridge is
needed. A worker idle-exits after :attr:`IDLE_TIMEOUT_S` and is transparently
re-spawned by the next :meth:`enqueue`; callers must NOT assume a warm worker.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

log = logging.getLogger(__name__)


def _log_subscriber_failure(task: asyncio.Task) -> None:
    """Log an exception raised by a fire-and-forget subscriber coroutine."""
    try:
        task.result()
    except Exception:
        log.exception("background-task subscriber coroutine failed")


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


# ---------------------------------------------------------------------------
# BackgroundCoordinator — the session-level background orchestrator.
#
# Wires the concrete queue types (memory consolidation, memory review, thread
# summarizer) to a session's cadence + watermark gating and exposes their
# lifecycle as ``BackgroundTask`` rows -- the "what the UI sees" surface.
# ``runtime.Session`` holds a ``background`` slot (a ``BackgroundCoordinator |
# None``) and calls the single ``after_turn()`` seam at the end of ``stream()``.
#
# "Never break the turn": every trigger method wraps its body in try/except and
# marks the task row ERROR rather than re-raising, and a failing subscriber
# callback is logged and skipped -- a background failure never propagates into
# the turn.
# ---------------------------------------------------------------------------

# Step lifecycle statuses.
PROCESSING = "running"
DONE = "done"
ERROR = "error"
QUEUED = "queued"
SKIPPED = "skipped"

StepStatus = Literal["running", "done", "error", "queued", "skipped"]

# How long a finished background job's last-run outcome (done/skipped) stays
# visible in the Processes panel before the row reverts to plain idle. Chosen to
# be comfortably larger than the SSE idle heartbeat so the carry-forward is
# re-published at least once, then naturally expires.
_PROCESS_STATUS_HOLD_S = 60


@dataclass
class BackgroundTask:
    """Registry entry for a background task (consumer #1 = memory consolidation).

    One row per background task with a stable ``id`` the UI can mutate
    ``queued -> running -> done|error|skipped`` by. ``progress`` is 0..1 when
    known; ``detail`` is branch-specific free text (e.g. ``"5 turns -> 1
    summary"``). Lives in the coordinator's task registry and is REPLAYED to
    late-joining ``subscribe_tasks`` subscribers, so a web tab that connects
    mid-fold sees ``running`` instead of nothing.
    """

    id: str
    label: str
    status: StepStatus = QUEUED
    progress: float | None = None
    detail: str = ""
    updated_at: str = ""
    # Monotonic per-coordinator publish revision, stamped by _publish_task. A
    # strictly-increasing counter, so "which publish is newest" is unambiguous
    # even when two publishes carry identical microsecond-stripped timestamps
    # (see _reconcile_status). Not serialized to subscribers via as_event.
    publish_seq: int = 0

    def as_event(self) -> BackgroundEvent:
        """The additive event form subscribers receive on each transition."""
        return BackgroundEvent(
            kind=f"task:{self.label}",
            status=self.status,
            detail=self.detail,
            metadata={"id": self.id, "progress": self.progress},
        )


@dataclass(frozen=True)
class BackgroundEvent:
    """Background / long-running work item (consolidation, subagent, ...).

    Kept generic so the first consumer (memory consolidation) and later ones
    (sub-agents, long-running services) share one surface. ``kind`` names the
    background task type; ``status`` mirrors :data:`StepStatus`.
    """

    kind: str
    status: StepStatus
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobIdleStatus:
    """Read-only "idle" status for one background job, for the Processes panel.

    Produced by ``BackgroundCoordinator.job_status_summary()`` so a UI can show
    the three background processes as persistent rows even when nothing is
    running — with a count and a human reason ("2 turns since last fold ·
    next ~24 min") instead of an empty panel. ``id`` is the stable row key the
    frontend coalesces transitions against; when the real task fires later it
    overlays this idle row (same id, queued/running/done).
    """

    id: str
    label: str
    status: Literal["idle", "done", "skipped", "running", "queued", "error"] = "idle"
    progress: float | None = None
    detail: str = ""


class BackgroundCoordinator:
    """Owns the session's background machinery: task registry + 3 queues + triggers.

    Constructed by the app/CLI and attached to a session via ``session.background``;
    a bare ``Session`` (memory-less profile, e.g. ``contextbench``) is ``None``
    here.

    Owns:
    - the task registry (``_background_tasks``, ``subscribe_tasks`` replay,
      ``_publish_task``, ``task_history``),
    - the three lazy queues (memory consolidation, memory review, thread
      summarizer),
    - the three ``after_turn`` triggers (``_maybe_trigger_*``) scoped by
      conversation + the owner-writes-watermark contract,
    - the ``_complete`` / ``on_complete`` seams that mutate task rows on
      worker completion.

    All cadence/watermark/identity state is self-contained: the coordinator
    derives its memory dir, thread capture root, and cadences from ``agent_name``
    / ``profile``, so attaching one is a single call and the Session stays lean.
    """

    def __init__(
        self,
        store,
        *,
        agent_name: str = "assistant",
        profile=None,
        memory_directory: str | None = None,
        consolidate_span=None,
        consolidate_every: int | None = None,
        review_every: int | None = None,
        consolidate_turns: int | None = None,
        review_turns: int | None = None,
        thread_summarize_block=None,
        thread_summarize_span=None,
    ) -> None:
        from paf.capabilities.memory._window import memory_dir_for

        self._store = store
        self.agent_name = agent_name
        self.profile = profile

        # Turn-count cadence (the active gate). Resolved from the spec's
        # `memory:` block; falls back to the same block's minute values reading
        # as counts when an old spec only set `consolidate_every`/`review_every`.
        from paf.specs import resolve_memory_config

        _mem_cfg = resolve_memory_config(agent_name)
        self.consolidate_turns = (
            consolidate_turns
            if consolidate_turns is not None
            else _mem_cfg["consolidate_turns"]
        )
        self.review_turns = (
            review_turns if review_turns is not None else _mem_cfg["review_turns"]
        )

        # Per-agent scope dirs.
        self._memory_directory = memory_directory or memory_dir_for(self.agent_name)

        from paf.capabilities.thread_summarizer._capability import capture_directory_for

        self._thread_capture_root = str(capture_directory_for(self.agent_name))
        self._thread_summarizer_enabled = bool(
            profile is not None
            and "ThreadSummarizer" in getattr(profile, "capability_names", ())
        )
        self._thread_summarize_span = (
            thread_summarize_span
            if thread_summarize_span is not None
            else thread_summarize_block
        )

        # Consolidation seams (injected only by tests; production defaults the
        # queue's real fold/reviewer).
        self._consolidate_span = consolidate_span

        # Lazily-built queues (one per type) + their rows.
        self._memory_queue: Any | None = None
        self._review_queue: Any | None = None
        self._thread_queue: Any | None = None

        # Task registry + subscriber list (+ seq counters for stable ids).
        self._background_tasks: dict[str, BackgroundTask] = {}
        self._task_subscribers: list[Callable[[BackgroundEvent], Any]] = []
        self._consolidation_seq = 0
        self._review_seq = 0
        self._thread_summarizer_seq = 0

    @classmethod
    def for_session(cls, session) -> BackgroundCoordinator | None:
        """Build (or skip) the coordinator for a session from its own state.

        The app/CLI attaches the result to ``session.background``. Returns
        ``None`` for a profile that needs no background machinery (no Memory,
        no ThreadSummarizer) -- e.g. ``contextbench`` -- so those agents run
        truly bare. Reads only public/duck-typed session attributes (no import
        of ``paf.runtime``), so there is no circular dependency.
        """
        agent_name = session.agent_name
        profile = getattr(session, "profile", None)
        caps = getattr(profile, "capability_names", ()) if profile is not None else ()
        has_memory = "Memory" in caps
        has_thread = "ThreadSummarizer" in caps
        if not has_memory and not has_thread:
            return None
        return cls(
            store=session._store,
            agent_name=agent_name,
            profile=profile,
        )

    # -- task registry / subscriber fan-out -------------------------------

    @property
    def memory_queue(self):
        """The lazily-built background consolidation queue (None until first use)."""
        return self._memory_queue

    @property
    def review_queue(self):
        """The lazily-built background reviewer queue (None until first use)."""
        return self._review_queue

    @property
    def thread_summarizer_queue(self):
        """The lazy thread queue (None when the profile leaves it OFF)."""
        return self._thread_queue

    def subscribe_tasks(
        self, cb: Callable[[BackgroundEvent], Any]
    ) -> Callable[[BackgroundEvent], Any]:
        """Register a consumer for background-task lifecycle.

        The callback is invoked on every ``BackgroundEvent`` transition
        (queued/running/done/error/skipped) for every registered task. On
        subscribe, ALREADY-ACTIVE tasks (status queued|running) are REPLAYED
        synchronously first, so a late-joining connection (e.g. a web tab that
        opens mid-fold) sees ``running`` instead of missing the transition.

        Returns the callback unchanged (convenient: ``cb = s.subscribe_tasks(cb)``).
        """
        if cb not in self._task_subscribers:
            self._task_subscribers.append(cb)
        for task in self._background_tasks.values():
            if task.status in (QUEUED, PROCESSING):
                cb(task.as_event())
        return cb

    def unsubscribe_tasks(self, cb: Callable[[BackgroundEvent], Any]) -> None:
        """Deregister a consumer previously registered via :meth:`subscribe_tasks`.

        Idempotent and best-effort: a callback that isn't registered (or was
        already removed) is a no-op. Used by long-lived server-side feeds (the
        dedicated ``/api/processes/stream``) so a disconnecting client doesn't
        leak a subscriber that keeps pushing into a dead queue.
        """
        try:
            self._task_subscribers.remove(cb)
        except ValueError:
            pass

    def task_history(self, conversation_id: str) -> list[BackgroundTask]:
        """Background-task rows scoped to a conversation, newest first.

        Includes terminal ``error`` rows (e.g. a failed thread summarizer) so a
        UI can surface "a background task failed for this conversation" even
        though the completion lands AFTER the turn's SSE stream closes. Task
        ids are prefixed ``{kind}.{conversation_id}.{seq}``; we match on the
        conversation segment (bounds-checked so a cid prefix can't over-match).
        """
        needle = f".{conversation_id}."
        rows = [t for t in self._background_tasks.values() if needle in f".{t.id}"]
        rows.sort(key=lambda t: t.updated_at, reverse=True)
        return rows

    def _reconcile_status(
        self, row: JobIdleStatus, prefix: str, conversation_id: str
    ) -> JobIdleStatus:
        """Merge the most-recent last-run outcome into an idle row.

        The Processes panel wants the background jobs to LOOK alive: when a job
        just finished (done/skipped) within the hold window, carry its status +
        detail ("N records folded", "only 6 unsettled turns; need >= 10")
        instead of a bland "idle". After ``_PROCESS_STATUS_HOLD_S`` passes, the
        row reverts to the freshly-computed idle state (the caller rebuilds it
        every heartbeat anyway), so a stale outcome never lingers.

        ``prefix`` is the task-id namespace (``consolidation``, ``review``,
        ``thread_summarize``); we only match tasks scoped to THIS conversation
        via the ``.{cid}.`` segment so one agent's run never leaks onto another.
        Read-only: never enqueues, mutates the registry, or advances state.
        """
        needle = f".{conversation_id}."
        recent: BackgroundTask | None = None

        # Select the *authoritative* task row. Task ids end in ``.{seq}``, a
        # per-(kind,conversation) monotonic counter, so a HIGHER seq is always
        # the newer trigger — even when two publishes land in the same
        # microsecond-stripped second (the norm: a skip and the fold that
        # follows it fire within moments, and _publish_task stamps both with
        # the same `now`). Comparing only `updated_at` then picks the FIRST
        # dict-iteration task on a tie — which can resurrect a STALE skipped
        # row ("only 4 unsettled turn(s); need >= 5") over the genuinely newer
        # running fold. Compare (seq, updated_at): a newer seq wins; fall back
        # to timestamp only within the same seq.
        def _key(t: BackgroundTask) -> tuple[int, str]:
            seg = t.id.rsplit(".", 1)[-1]
            try:
                return (int(seg), t.updated_at)
            except ValueError:
                return (-1, t.updated_at)  # unparsable -> oldest bucket

        for task in self._background_tasks.values():
            if (
                task.id.startswith(f"{prefix}.")
                and needle in task.id
                and task.status in (QUEUED, PROCESSING, DONE, SKIPPED, ERROR)
            ):
                if recent is None or _key(task) > _key(recent):
                    recent = task
        if recent is None:
            return row
        try:
            updated = datetime.fromisoformat(recent.updated_at)
        except (TypeError, ValueError):
            return row
        # An ACTIVE task (queued/running) is live: its last publish timestamp
        # only records when the job was handed off, not its progress, so a long
        # fold (minutes on an LLM) must NEVER decay to "idle" just because the
        # heartbeat lands past the hold window. Surface it as running regardless
        # of age. Only TERMINAL outcomes obey the hold window, so a finished
        # run still reverts its row to idle once the outcome is stale.
        is_active = recent.status in (QUEUED, PROCESSING)
        if (
            not is_active
            and (datetime.now(UTC) - updated).total_seconds() > _PROCESS_STATUS_HOLD_S
        ):
            return row
        if is_active:
            # An in-flight fold carries NO idle cadence text: the
            # "n/N turns to a fold/review" countdown describes how far the
            # NEXT cadence has progressed since the last done watermark — it is
            # unrelated to the job currently running and would mislead under a
            # `running` status. Use the task's own detail (empty while the
            # worker hasn't reported progress); never fall back to the idle
            # row's detail.
            return JobIdleStatus(
                id=row.id,
                label=row.label,
                status=recent.status,
                progress=recent.progress,
                detail=recent.detail,
            )
        return JobIdleStatus(
            id=row.id,
            label=row.label,
            status=recent.status,
            progress=recent.progress,
            detail=recent.detail or row.detail,
        )

    async def job_status_summary(self, conversation_id: str) -> list[JobIdleStatus]:
        """READ-ONLY idle status for the 3 background processes (Processes panel).

        Lets a UI render the background jobs as persistent rows — "idle, N
        turns, next ~M min" — even when nothing is running, instead of an
        empty panel. Derives each job's count/reason from the SAME durable
        state its trigger reads (watermarks + persisted runs), but NEVER
        enqueues, writes, or advances anything: a read failure yields an
        "unavailable" idle row rather than a raised error, so a bad watermark
        can't break (or block) the feed.

        Each row's ``id`` is the STABLE key the real task uses (``consolidation.*``
        etc.), so when that job fires later the live queued/running/done frames
        coalesce OVER the idle row instead of duplicating it.
        """
        from paf.capabilities._turns import user_anchor_indices
        from paf.capabilities.memory._background import resolve_unseen_span
        from paf.capabilities.memory._watermark import MemoryWatermark
        from paf.capabilities.persistence import build_history_source
        from paf.capabilities.thread_summarizer._background import BLOCK_TURNS
        from paf.capabilities.thread_summarizer._store import THREAD_WATERMARK_FILENAME

        rows: list[JobIdleStatus] = []
        source = None
        tip_run = None
        tip_turn = 0

        try:
            source = build_history_source(self._store)
            # A single pass over persisted runs/history feeds all three counts.
            runs = await self._store.list_runs(conversation_id=conversation_id)
            tip_run = runs[-1].run_id if runs else None
            tip_history = await source.run_history(run_id=tip_run) if tip_run else []
            anchors = user_anchor_indices(tip_history)
            tip_turn = len(anchors) - 1 if anchors else 0
        except Exception:
            # The shared store/history seam is down (or this store can't build
            # a history source): report all 3 jobs as "unavailable" rather than
            # raising — the feed must never tear the SSE socket on a status read.
            for _id, _label in (
                (f"consolidation.{conversation_id}", "Consolidating memory"),
                (f"review.{conversation_id}", "Reviewing memory"),
                (f"thread_summarize.{conversation_id}", "Summarizing thread"),
            ):
                rows.append(
                    JobIdleStatus(
                        id=_id, label=_label, detail="idle · status unavailable"
                    )
                )
            return rows

        async def _span_count(wm: MemoryWatermark) -> int:
            """Number of distinct complete user turns newer than the watermark.

            Mirrors the summarizer trigger's own gate (``distinct_user_turn_texts``),
            so the idle ``n/BLOCK_TURNS`` displayed always means the same thing the
            trigger tests. A fresh/empty watermark resolves the whole completed
            history (``resolve_unseen_span`` handles it), so it reports the real
            accumulated count instead of short-circuiting to 0 forever.
            """
            from paf.capabilities._turns import distinct_user_turn_texts

            try:
                span = await resolve_unseen_span(
                    source,
                    wm,
                    tip_run=tip_run or "",
                    tip_turn=tip_turn,
                    conversation_id=conversation_id,
                )
            except Exception:
                return 0
            if not span.messages:
                return 0
            return len(distinct_user_turn_texts(span.messages))

        # 1. Memory consolidation (agent-scoped, TURN-count cadence).
        #    Display mirrors the thread-summarizer gate: n accrued turns vs the
        #    block threshold, never a wall-clock countdown.
        try:
            wm = MemoryWatermark.load(self._memory_directory)
            n = await _span_count(wm)
            reason = (
                f"{n}/{self.consolidate_turns} turns to a fold"
                if n < self.consolidate_turns
                else "due for a fold"
            )
            rows.append(
                self._reconcile_status(
                    JobIdleStatus(
                        id=f"consolidation.{conversation_id}",
                        label="Consolidating memory",
                        detail=f"idle · {reason}",
                    ),
                    "consolidation",
                    conversation_id,
                )
            )
        except Exception:
            rows.append(
                JobIdleStatus(
                    id=f"consolidation.{conversation_id}",
                    label="Consolidating memory",
                    detail="idle · status unavailable",
                )
            )

        # 2. Memory review (agent-scoped, TURN-count cadence, slower block).
        try:
            wm = MemoryWatermark.load(
                self._memory_directory, filename="memory-review-watermark.json"
            )
            n = await _span_count(wm)
            reason = (
                f"{n}/{self.review_turns} turns to a review"
                if n < self.review_turns
                else "due for a review"
            )
            rows.append(
                self._reconcile_status(
                    JobIdleStatus(
                        id=f"review.{conversation_id}",
                        label="Reviewing memory",
                        detail=f"idle · {reason}",
                    ),
                    "review",
                    conversation_id,
                )
            )
        except Exception:
            rows.append(
                JobIdleStatus(
                    id=f"review.{conversation_id}",
                    label="Reviewing memory",
                    detail="idle · status unavailable",
                )
            )

        # 3. Thread summarizer (per-conversation, turn-count cadence).
        if self._thread_summarizer_enabled:
            try:
                from paf.capabilities.thread_summarizer._store import ThreadSummaryStore

                # Derive the SAME conversation-scoped dir + run-anchored watermark
                # the trigger/worker use (conversation_key hashed dir, run_id), or
                # the displayed n/BLOCK_TURNS diverges from the real gate.
                store = ThreadSummaryStore(self._thread_capture_root, conversation_id)
                wm = MemoryWatermark.load(
                    store.directory,
                    run_id=tip_run or "",
                    filename=THREAD_WATERMARK_FILENAME,
                )
                n = await _span_count(wm)
                reason = (
                    f"{n}/{BLOCK_TURNS} turns to a summary"
                    if n < BLOCK_TURNS
                    else "due for a summary"
                )
                rows.append(
                    self._reconcile_status(
                        JobIdleStatus(
                            id=f"thread_summarize.{conversation_id}",
                            label="Summarizing thread",
                            detail=f"idle · {reason}",
                        ),
                        "thread_summarize",
                        conversation_id,
                    )
                )
            except Exception:
                rows.append(
                    JobIdleStatus(
                        id=f"thread_summarize.{conversation_id}",
                        label="Summarizing thread",
                        detail="idle · status unavailable",
                    )
                )

        return rows

    def _publish_task(self, task: BackgroundTask) -> None:
        """Store a task in the registry and notify all subscribers.

        A failing subscriber must never break the session/coordinator: the
        error is logged and the callback skipped (fire-and-forget awaits).
        """
        task.updated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        self._background_tasks[task.id] = task
        ev = task.as_event()
        for cb in self._task_subscribers:
            try:
                result = cb(ev)
                if inspect.isawaitable(result):
                    t = asyncio.ensure_future(result)  # fire-and-forget, never block
                    t.add_done_callback(_log_subscriber_failure)
            except Exception:  # a subscriber must never break the session
                log.exception("background-task subscriber failed task=%s", task.id)

    # -- queue builders (lazy) ---------------------------------------------

    def _ensure_memory_queue(self):
        """Build the self-contained worker queue, bound to ``self._store``."""
        if self._memory_queue is None:
            from paf.capabilities.memory._background import (
                MemoryConsolidationQueue,
                resolve_unseen_span,
            )
            from paf.capabilities.memory._watermark import MemoryWatermark
            from paf.capabilities.persistence import build_history_source

            def _complete(kind: StepStatus, detail: str = "") -> None:
                for task_id, task in list(self._background_tasks.items()):
                    if task_id.startswith("consolidation.") and task.status in (
                        QUEUED,
                        PROCESSING,
                    ):
                        task.status = kind
                        task.detail = detail
                        task.progress = 1.0 if kind == DONE else None
                        self._publish_task(task)

            source = build_history_source(self._store)

            async def span_source(run_id: str, up_to_turn_id: int):
                wm = MemoryWatermark.load(self._memory_directory, run_id=run_id)
                return await resolve_unseen_span(
                    source, wm, tip_run=run_id, tip_turn=up_to_turn_id
                )

            real_fold = self._consolidate_span
            queue_ref: MemoryConsolidationQueue | None = None

            async def publish_fold(messages, directory, watermark):
                try:
                    if real_fold is not None:
                        n = await real_fold(messages, directory, watermark)
                    else:
                        # Production coordinators pass no `consolidate_span`; fall
                        # back to the queue's real fold so the owner actually
                        # writes.
                        assert queue_ref is not None
                        n = await queue_ref._default_consolidate_span(
                            messages, directory, watermark
                        )
                    _complete(DONE, f"{n} records folded")
                    return n
                except Exception as exc:
                    _complete(ERROR, str(exc))
                    raise

            queue = MemoryConsolidationQueue(
                memory_directory=self._memory_directory,
                agent_name=self.agent_name,
                span_source=span_source,
                consolidate_span=publish_fold,
            )
            queue_ref = queue
            self._memory_queue = queue
        return self._memory_queue

    def _ensure_review_queue(self):
        """Build the self-contained background reviewer queue."""
        if self._review_queue is None:
            from paf.capabilities.memory._background import MemoryReviewQueue
            from paf.capabilities.memory._review import review as _default_review

            def _complete(kind: StepStatus, detail: str = "") -> None:
                for task_id, task in list(self._background_tasks.items()):
                    if task_id.startswith("review.") and task.status in (
                        QUEUED,
                        PROCESSING,
                    ):
                        task.status = kind
                        task.detail = detail
                        task.progress = 1.0 if kind == DONE else None
                        self._publish_task(task)

            async def review_span(directory: str) -> int:
                from paf.capabilities.memory._overrides import OverrideStore

                try:
                    n = await _default_review(
                        directory,
                        overrides=OverrideStore(directory),
                        agent_name=self.agent_name,
                    )
                    _complete(DONE, f"{n} verdict(s) applied")
                    return n
                except Exception as exc:
                    _complete(ERROR, str(exc))
                    raise

            self._review_queue = MemoryReviewQueue(
                memory_directory=self._memory_directory,
                agent_name=self.agent_name,
                review_span=review_span,
            )
        return self._review_queue

    def _ensure_thread_summarizer_queue(self):
        """Build the conversation-scoped thread worker lazily."""
        if self._thread_queue is None:
            from paf.capabilities.thread_summarizer._background import (
                ThreadSummarizerQueue,
            )

            self._thread_queue = ThreadSummarizerQueue(
                capture_root=self._thread_capture_root,
                summarize_span_fn=self._thread_summarize_span,
                on_complete=self._complete_thread_summarize,
            )
        return self._thread_queue

    def _complete_thread_summarize(
        self, conversation_id: str, success: bool, detail: str
    ) -> None:
        """Publish completion for active thread-summarizer task rows."""
        for task in self._background_tasks.values():
            if task.id.startswith(
                f"thread_summarize.{conversation_id}."
            ) and task.status in (QUEUED, PROCESSING):
                task.status = DONE if success else ERROR
                task.detail = detail
                task.progress = 1.0 if success else None
                self._publish_task(task)

    # -- post-turn triggers (the single seam) ------------------------------

    async def after_turn(self, cid: str | None) -> None:
        """Fire the post-turn background triggers for one conversation.

        The single seam ``runtime.Session.stream()`` calls at the end of a turn.
        Schedules each background job whose cadence/watermark says it's due; the
        workers are the sole watermark writers. Never raises -- a failure in any
        trigger is captured on that task's row.
        """
        await self._maybe_trigger_consolidation(cid)
        await self._maybe_trigger_review(cid)
        await self._maybe_trigger_thread_summarize(cid)

    async def _maybe_trigger_thread_summarize(self, cid: str | None) -> None:
        """Fire conversation-scoped thread summarization on its own cadence.

        Mirror of ``_maybe_trigger_consolidation``, but scoped by CONVERSATION,
        not agent: resolve this conversation's unseen turn-span (the portion
        strictly newer than its per-conversation ``MemoryWatermark``), gate on
        a TURN COUNT (>= ``BLOCK_TURNS`` complete turns), single-flight, and
        hand the PRE-RESOLVED span to the worker. The trigger SCHEDULES only;
        the worker is the sole per-conversation watermark writer and advances
        only on a completed (non-error) summarize pass -- a failed pass leaves
        the watermark behind so the span is retried next cadence, never skipped.

        Unlike Memory's TIME cadence, the thread summarizer fires on a TURN
        COUNT: it only enqueues once at least ``BLOCK_TURNS`` (10) COMPLETE
        turns have accumulated since the watermark, so a 1-turn trailing thread
        never triggers a useless 1-turn block. Two conversations never share a
        watermark or cross-summarize. ``cid`` is already profile-scoped.
        """
        if not self._thread_summarizer_enabled or cid is None:
            return
        from paf.capabilities._turns import (
            distinct_user_turn_texts,
            user_anchor_indices,
        )
        from paf.capabilities.memory._background import resolve_unseen_span
        from paf.capabilities.memory._watermark import MemoryWatermark
        from paf.capabilities.persistence import build_history_source
        from paf.capabilities.thread_summarizer._background import (
            BLOCK_TURNS,
            ThreadSummarizeJob,
        )
        from paf.capabilities.thread_summarizer._store import (
            THREAD_WATERMARK_FILENAME,
            ThreadSummaryStore,
        )

        source = build_history_source(self._store)

        self._thread_summarizer_seq += 1
        task_id = f"thread_summarize.{cid}.{self._thread_summarizer_seq}"
        task = BackgroundTask(id=task_id, label=f"Summarizing thread ({cid})")
        self._publish_task(task)  # queued
        try:
            runs = await self._store.list_runs(conversation_id=cid)
            if not runs:
                task.status = SKIPPED
                task.detail = "no persisted runs yet"
                self._publish_task(task)
                return
            tip_run = runs[-1].run_id
            tip_history = await source.run_history(run_id=tip_run)
            anchors = user_anchor_indices(tip_history)
            if not anchors:
                task.status = SKIPPED
                task.detail = "no persisted turns"
                self._publish_task(task)
                return
            tip_turn = len(anchors) - 1
            store = ThreadSummaryStore(self._thread_capture_root, cid)
            wm = MemoryWatermark.load(
                store.directory, run_id=tip_run, filename=THREAD_WATERMARK_FILENAME
            )
            # Resolve the unseen span for THIS conversation now; hand the worker
            # precomputed material so it summarizes exactly what was scheduled.
            span = await resolve_unseen_span(
                source,
                wm,
                tip_run=tip_run,
                tip_turn=tip_turn,
                conversation_id=cid,
            )
            # TURN-COUNT gate (the thread-specific cadence). A block is only
            # worth producing once >= BLOCK_TURNS *distinct real* turns
            # accumulated; fewer and the narration is noise, not a thread.
            span_turns = (
                len(distinct_user_turn_texts(span.messages)) if span.messages else 0
            )
            if span_turns < BLOCK_TURNS:
                task.status = SKIPPED
                task.detail = (
                    f"only {span_turns} unsettled turn(s); need >= {BLOCK_TURNS}"
                )
                self._publish_task(task)
                return
            # Single-flight: skip if a thread summarize is already queued/running.
            queue = self._ensure_thread_summarizer_queue()
            cst = queue.status(cid)
            if cst.queued > 0 or cst.running:
                task.status = SKIPPED
                task.detail = "thread summarization already in flight"
                self._publish_task(task)
                return
            queue.enqueue(
                cid,
                ThreadSummarizeJob(
                    conversation_id=cid,
                    messages=span.messages,
                    watermark=span.watermark,
                ),
            )
            task.status = PROCESSING  # job handed to the worker
            self._publish_task(task)
        except (
            Exception
        ) as exc:  # trigger path itself failed (rare); never break the turn
            task.status = ERROR
            task.detail = str(exc)
            self._publish_task(task)

    async def _maybe_trigger_review(self, cid: str | None) -> None:
        """Fire the background reviewer hook, publishing task lifecycle.

        Mirrors ``_maybe_trigger_consolidation``: publish a ``queued`` task row
        first, then mark skipped / processing. Completion (done/error) is
        published by the review span seam in ``_ensure_review_queue`` -- the
        reviewer is now visible on the background-task timeline / Processes
        panel just like consolidation and thread summarization.

        Like consolidation: the trigger SCHEDULES only. It enqueues a ReviewJob
        carrying the resolved tip position; the review worker is the SOLE
        review-watermark writer and advances it only on a COMPLETED pass. A
        failed/never-run review leaves the watermark behind so it retries next
        cadence instead of being silently skipped.
        """
        if cid is None:
            return
        from paf.capabilities._turns import (
            distinct_user_turn_texts,
            user_anchor_indices,
        )
        from paf.capabilities.memory._background import ReviewJob, resolve_unseen_span
        from paf.capabilities.memory._watermark import MemoryWatermark
        from paf.capabilities.persistence import build_history_source

        source = build_history_source(self._store)

        self._review_seq += 1
        task_id = f"review.{cid}.{self._review_seq}"
        task = BackgroundTask(id=task_id, label=f"Reviewing memory ({cid})")
        self._publish_task(task)  # queued
        try:
            runs = await self._store.list_runs(conversation_id=cid)
            if not runs:
                task.status = SKIPPED
                task.detail = "no persisted runs yet"
                self._publish_task(task)
                return
            tip_run = runs[-1].run_id
            tip_history = await source.run_history(run_id=tip_run)
            anchors = user_anchor_indices(tip_history)
            if not anchors:
                task.status = SKIPPED
                task.detail = "no persisted turns"
                self._publish_task(task)
                return
            tip_turn = len(anchors) - 1
            wm = MemoryWatermark.load(
                self._memory_directory,
                run_id=tip_run,
                filename="memory-review-watermark.json",
            )
            # TURN-COUNT gate: fire review only once >= `review_turns` complete
            # turns have accrued since the last review, mirroring the
            # summarizer's block gate. The reviewer is SLOW (~10 min
            # archive/merge); 10 turns batches ~2 consolidation folds, and
            # review never mid-races consolidation.
            span = await resolve_unseen_span(
                source,
                wm,
                tip_run=tip_run,
                tip_turn=tip_turn,
            )
            span_turns = (
                len(distinct_user_turn_texts(span.messages)) if span.messages else 0
            )
            if span_turns < self.review_turns:
                task.status = SKIPPED
                task.detail = (
                    f"only {span_turns} unsettled turn(s); need >= {self.review_turns}"
                )
                self._publish_task(task)
                return
            # Mirror the owner's worker-owns-watermark contract: the trigger
            # SCHEDULES only. It enqueues a ReviewJob carrying the resolved tip
            # position; the review worker is the SOLE review-watermark writer
            # and advances it only on a COMPLETED (non-error) curation pass. A
            # failed/never-run review leaves the watermark behind so it retries
            # next cadence instead of being silently skipped.
            advance = MemoryWatermark(
                run_id=tip_run, turn_id=tip_turn, consumed_ts=runs[-1].started_at
            )
            rq = self._ensure_review_queue()
            # Single-flight: skip if a review is already queued or running so a
            # slow worker behind the cadence gate can't pile up duplicate passes.
            rst = rq.status(self.agent_name)
            if rst.queued > 0 or rst.running:
                task.status = SKIPPED
                task.detail = "memory review already in flight"
                self._publish_task(task)
                return
            rq.enqueue(
                self.agent_name,
                ReviewJob(run_id=tip_run, turn_id=tip_turn, watermark=advance),
            )
            task.status = PROCESSING  # job handed to the worker
            self._publish_task(task)
        except Exception as exc:  # the review hook must not break the turn
            task.status = ERROR
            task.detail = str(exc)
            self._publish_task(task)

    async def _maybe_trigger_consolidation(self, cid: str | None) -> None:
        """Fire the background-consolidation hook, publishing task lifecycle.

        Consumer #1 of the ``BackgroundEvent`` surface (mirrors coding_agent
        ``MaintenanceStep``). First publish a ``queued`` task row, then fire the
        trigger. The trigger builds the queue (and enqueues) ONLY when the
        watermark says enough turns have accumulated; if the queue is still None,
        no job is due this turn -- mark skipped, don't leave a phantom "running"
        row.

        ``cid`` here is ALREADY profile-scoped by the caller (``_scoped_cid``).
        """
        if cid is None:
            return
        from paf.capabilities._turns import (
            distinct_user_turn_texts,
            user_anchor_indices,
        )
        from paf.capabilities.memory._background import (
            ConsolidationJob,
            resolve_unseen_span,
        )
        from paf.capabilities.memory._watermark import MemoryWatermark
        from paf.capabilities.persistence import build_history_source

        source = build_history_source(self._store)

        self._consolidation_seq += 1
        task_id = f"consolidation.{cid}.{self._consolidation_seq}"
        task = BackgroundTask(id=task_id, label=f"Consolidating memory ({cid})")
        self._publish_task(task)  # queued
        try:
            runs = await self._store.list_runs(conversation_id=cid)
            if not runs:
                task.status = SKIPPED
                task.detail = "no persisted runs yet"
                self._publish_task(task)
                return
            tip_run = runs[-1].run_id
            tip_history = await source.run_history(run_id=tip_run)
            anchors = user_anchor_indices(tip_history)
            if not anchors:
                task.status = SKIPPED
                task.detail = "no persisted turns"
                self._publish_task(task)
                return
            tip_turn = len(anchors) - 1
            wm = MemoryWatermark.load(self._memory_directory, run_id=tip_run)
            # Resolve the unseen span NOW and hand it to the worker as
            # precomputed material so it folds exactly what was scheduled
            # (re-resolving inside the worker against the not-yet-advanced
            # watermark would otherwise see a moving cursor race). The trigger
            # SCHEDULES only -- it does NOT advance the watermark here. The
            # worker is the sole owner-watermark writer and advances only on a
            # completed (non-error) fold; a failed fold leaves the watermark
            # behind so this span is retried next cadence, never skipped.
            span = await resolve_unseen_span(
                source,
                wm,
                tip_run=tip_run,
                tip_turn=tip_turn,
            )
            # TURN-COUNT gate: fire consolidation only once >=
            # `consolidate_turns` complete turns have accrued since the last
            # fold. Memory is cheap extract+append, so 5 turns keeps it warm
            # without a per-turn model call.
            span_turns = (
                len(distinct_user_turn_texts(span.messages)) if span.messages else 0
            )
            if span_turns < self.consolidate_turns:
                task.status = SKIPPED
                task.detail = f"only {span_turns} unsettled turn(s); need >= {self.consolidate_turns}"
                self._publish_task(task)
                return
            # Single-flight: skip if a consolidation is already queued or running
            # so a slow worker behind the cadence gate can't pile up duplicate folds.
            mq = self._ensure_memory_queue()
            mst = mq.status(self.agent_name)
            if mst.queued > 0 or mst.running:
                task.status = SKIPPED
                task.detail = "consolidation already in flight"
                self._publish_task(task)
                return
            mq.enqueue(
                self.agent_name,
                ConsolidationJob(
                    run_id=tip_run,
                    turn_id=tip_turn,
                    messages=span.messages,
                    watermark=span.watermark,
                ),
            )
            task.status = PROCESSING  # job handed to the worker
            self._publish_task(task)
        except Exception as exc:  # trigger path itself failed (rare)
            task.status = ERROR
            task.detail = str(exc)
            self._publish_task(task)
