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
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Coroutine
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, cast

from pydantic_ai_harness.step_persistence._context import (
    current_run_id,
    live_run_history,
)

from paf.background_processes._events import (
    ProcessEvent,
    ProcessEventType,
    get_agent_emitter,
)
from paf.observability.instrumentation import root_span

log = logging.getLogger(__name__)

#: Minimum seconds between keyword-source scans (the trigger fires every turn).
_KEYWORD_CHECK_INTERVAL_S = 60.0


def _log_subscriber_failure(task: asyncio.Task) -> None:
    """Log an exception raised by a fire-and-forget subscriber coroutine."""
    try:
        task.result()
    except Exception:
        log.exception("background-task subscriber coroutine failed")


#: Harness contextvars scoped to ONE agent run. Reset for detached background
#: work so it never inherits the spawning run's identity or live history.
_RUN_SCOPED_VARS: tuple[ContextVar[Any], ...] = (current_run_id, live_run_history)


def _detached_context() -> Context:
    """Copy the current context with the run-scoped harness vars cleared."""
    ctx = copy_context()
    for var in _RUN_SCOPED_VARS:
        ctx.run(var.set, None)
    return ctx


class BackgroundRunner:
    """Process-level task group for background workers that outlive sessions.

    Every :class:`BackgroundQueueService` spawns its worker here instead of on
    the session's event loop, so a worker keeps running after the turn that
    enqueued its job ends. That is what lets a detached background process be
    supervised to completion (truthful terminal status) even though the session
    that started it is gone. The app/CLI starts it once at boot and shuts it
    down on exit; it is never tied to a single session's lifecycle.
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()

    def spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task:
        """Run ``coro`` as a tracked background task on the current loop.

        The task is created on whatever loop is current (the server's loop in
        the app), so it survives session teardown; the runner only cancels it
        on :meth:`shutdown`.

        The task runs in a COPY of the caller's context with the harness
        step-persistence run scope reset (see :func:`_detached_context`): a
        worker outlives the run that spawned it, so it must not carry that
        run's ``current_run_id`` (which made every notify turn record the
        spawning turn as its ``parent_run_id``) or its live message list.
        Everything else — notably the OpenTelemetry span context, which keeps
        background spans nested under the triggering turn's trace — is kept.
        """
        task = asyncio.create_task(coro, context=_detached_context())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def shutdown(self) -> None:
        """Cancel and await every tracked task (called on server exit)."""
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()


# Process-level singleton: every queue's workers run here, detached from any
# single session's event loop.
_background_runner = BackgroundRunner()


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
            state.worker_task = _background_runner.spawn(self._worker(session_id))

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
                await self._run_job(session_id, job, state)
            except Exception:
                log.exception(
                    "%s_job_failed session=%s",
                    self.name,
                    session_id,
                )
            finally:
                state.active_jobs -= 1
                state.queue.task_done()

    async def _run_job(
        self, session_id: str, job: JobT, state: QueueState[JobT]
    ) -> None:
        """Run one job, in its own trace when :meth:`job_trace` names one."""
        spec = self.job_trace(session_id, job)
        if spec is None:
            await self.process_job(session_id, job, state)
            return
        name, attributes = spec
        with root_span(name, new_trace=True, **attributes):
            await self.process_job(session_id, job, state)

    def job_trace(
        self, session_id: str, job: JobT
    ) -> tuple[str, dict[str, Any]] | None:
        """``(span name, attributes)`` to run ``job`` as the root of a NEW trace.

        The worker task inherits whatever span was current when it was spawned
        (``asyncio.create_task`` copies it) — typically an unrelated conversation
        turn, or a notify turn that happened to trigger it, and for a warm worker
        possibly a turn from minutes earlier. A queue whose jobs are not part of
        that turn overrides this so each job gets its own trace (relate traces
        with ``langfuse.session.id`` in the attributes, not by nesting).
        ``None`` (default) runs the job under the inherited span.
        """
        del session_id, job
        return None

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
# Wires the concrete queue types (memory consolidation, thread summarizer) to a session's cadence + watermark gating and exposes their
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
    # Itemized result lines (e.g. each fact the reviewer archived), shown
    # expandable under the row.
    lines: list[str] = field(default_factory=list)
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
    the background processes as persistent rows even when nothing is
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
    # Itemized result of the run this row reports (see BackgroundTask.lines).
    lines: tuple[str, ...] = ()
    # The last completed run, kept on an idle row so its result outlives the
    # status hold window: its detail and when it finished (ISO 8601).
    last: str = ""
    last_at: str | None = None


@dataclass
class _IdleContext:
    """Shared history material for idle-row computation (one pass, every row).

    ``job_status_summary`` resolves the store/history seam ONCE and hands the
    result to every registered process's ``idle_row``, so the idle rows
    share a single pass over persisted runs instead of each re-reading history.
    """

    source: Any
    tip_run: str | None
    tip_turn: int


@dataclass
class ProcessDefinition:
    """Self-describing definition of one background process.

    The coordinator is a thin host over a registry of these: it dispatches
    post-turn triggers and idle-row computation to registered definitions
    instead of hardcoding each job type. Adding a process = registering a
    definition (id namespace, label, idle policy, enabled gate, trigger, and
    idle-row computation).
    """

    id_namespace: str  # task-id prefix, e.g. "consolidation"
    label: str  # "Consolidating memory"
    idle_policy: Literal["persistent", "transient"] = "persistent"
    # Whether this process is active for a given coordinator (profile gating).
    enabled: Callable[[BackgroundCoordinator], bool] | None = None
    # Post-turn trigger: schedule the job if its cadence/watermark says due.
    trigger: Callable[[str | None], Awaitable[None]] | None = None
    # Idle-row computation for the Processes panel (shared history ctx).
    idle_row: Callable[[str, _IdleContext], Awaitable[JobIdleStatus | None]] | None = (
        None
    )


class BackgroundCoordinator:
    """Owns the session's background machinery: task registry + 3 queues + triggers.

    Constructed by the app/CLI and attached to a session via ``session.background``;
    a bare ``Session`` (memory-less profile, e.g. ``contextbench``) is ``None``
    here.

    Owns:
    - the task registry (``_background_tasks``, ``subscribe_tasks`` replay,
      ``_publish_task``, ``task_history``),
    - the lazy queues (memory consolidation, thread summarizer),
    - the ``after_turn`` triggers (``_maybe_trigger_*``) scoped by
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
        agent_name: str = "demo_assistant",
        profile=None,
        lane: Literal["agent", "conversation"] | None = None,
        conversation_id: str | None = None,
        memory_directory: str | None = None,
        processes_root: str | None = None,
        consolidate_span=None,
        consolidate_every: int | None = None,
        consolidate_turns: int | None = None,
        thread_summarize_block=None,
        thread_summarize_span=None,
        keyword_pass=None,
    ) -> None:
        from paf.capabilities.memory._window import memory_dir_for

        self._store = store
        self.agent_name = agent_name
        self.profile = profile
        # Which lane this coordinator hosts: "agent" (memory consolidation,
        # shared per-agent) or "conversation" (thread summarizer +
        # future conversation-scoped processes, per-conversation). None = a
        # legacy direct construction that registers every process.
        self.lane = lane
        self.conversation_id = conversation_id
        # The shared per-agent live-event bus. Both lanes of this agent and any
        # other process (shell, future curator) emit here; the feed subscribes
        # once per agent and filters by lane/scope.
        self._emitter = get_agent_emitter(self.agent_name)

        # Dumb append-only run-event store: every task status
        # transition is recorded losslessly, scoped by agent. Root is
        # injectable so tests point at a tmp dir.
        from paf.background_processes._store import RunEventStore

        if processes_root is None:
            from paf.config import PROCESSES_DIR

            processes_root = str(PROCESSES_DIR)
        self._run_events = RunEventStore.open(
            processes_root, agent_name, conversation_id=conversation_id
        )

        # Turn-count cadence (the active gate). Resolved from the spec's
        # `memory:` block; falls back to the same block's minute values reading
        # as counts when an old spec only set `consolidate_every`.
        from paf.specs import resolve_memory_config

        _mem_cfg = resolve_memory_config(agent_name)
        self.consolidate_turns = (
            consolidate_turns
            if consolidate_turns is not None
            else _mem_cfg["consolidate_turns"]
        )
        # The pairwise reviewer runs after each fold unless the spec sets
        # `memory.pairwise_review: false`.
        self.pairwise_review = bool(_mem_cfg.get("pairwise_review", True))

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
        # Keyword curator: the profile allowlist makes the process available, but
        # it only RUNS when the spec opts in (`keywords: {curate: true}`) — a pass
        # sends skill/note digests to the model provider.
        self._keyword_curator_enabled = bool(
            profile is not None
            and "KeywordCurator" in getattr(profile, "capability_names", ())
        )
        self._keyword_pass = keyword_pass  # test seam; None = the real curator pass
        self._keyword_settings_cache: dict[str, Any] | None = None
        self._keyword_last_check = float("-inf")
        self._keyword_last_enqueue = float("-inf")

        # Consolidation seams (injected only by tests; production defaults the
        # queue's real fold).
        self._consolidate_span = consolidate_span

        # Lazily-built queues (one per type) + their rows.
        self._memory_queue: Any | None = None
        self._thread_queue: Any | None = None
        self._keyword_queue: Any | None = None

        # Task registry + subscriber list (+ seq counters for stable ids).
        self._background_tasks: dict[str, BackgroundTask] = {}
        self._consolidation_seq = 0
        self._thread_summarizer_seq = 0
        self._keyword_seq = 0

        # Capability names (profile gating for process registration).
        self._capability_names = tuple(
            getattr(profile, "capability_names", ()) if profile is not None else ()
        )
        # Registry of self-describing background processes (the thin-host seam).
        self._processes = self._build_processes()

    def _build_processes(self) -> list[ProcessDefinition]:
        """Register the coordinator's background processes (the thin-host seam).

        Each process is a self-describing :class:`ProcessDefinition`; the
        coordinator's public entry points (``for_session``, ``after_turn``,
        ``job_status_summary``) iterate this registry instead of hardcoding job
        types. Adding a process = registering a definition here.
        """
        has_memory = "Memory" in self._capability_names
        processes: list[ProcessDefinition] = []
        # Agent lane: memory consolidation, with the pairwise reviewer run
        # inside each fold job (per-agent).
        if self.lane != "conversation":
            processes += [
                ProcessDefinition(
                    id_namespace="consolidation",
                    label="Consolidating memory",
                    enabled=lambda c: has_memory,
                    trigger=self._maybe_trigger_consolidation,
                    idle_row=self._consolidation_idle_row,
                ),
            ]
            # Keyword curator: registered only when the profile allowlists it. No
            # idle row — it is quiet unless something changed, and a permanent
            # "idle" row would be noise in the Processes panel.
            if self._keyword_curator_enabled:
                processes.append(
                    ProcessDefinition(
                        id_namespace="keyword_curate",
                        label="Curating keywords",
                        enabled=lambda c: c._keyword_curator_enabled,
                        trigger=self._maybe_trigger_keyword_curate,
                    )
                )
        # Conversation lane: thread summarizer (per-conversation).
        if self.lane != "agent":
            processes.append(
                ProcessDefinition(
                    id_namespace="thread_summarize",
                    label="Summarizing thread",
                    enabled=lambda c: self._thread_summarizer_enabled,
                    trigger=self._maybe_trigger_thread_summarize,
                    idle_row=self._thread_idle_row,
                ),
            )
        return processes

    def has_enabled(self) -> bool:
        """True if any registered process is enabled for this coordinator."""
        return any(p.enabled is not None and p.enabled(self) for p in self._processes)

    @classmethod
    def for_session(cls, session) -> BackgroundLanes | None:
        """Build (or skip) the two-lane coordinator for a session.

        The app/CLI attaches the result to ``session.background``. Returns a
        :class:`BackgroundLanes` facade holding the shared per-agent lane
        (memory consolidation) and the per-session conversation lane
        (thread summarizer), or ``None`` for a profile that needs no background
        machinery (no Memory, no ThreadSummarizer) -- e.g. ``contextbench`` --
        so those agents run truly bare. Reads only public/duck-typed session
        attributes (no import of ``paf.runtime``), so there is no circular
        dependency.
        """
        agent_name = session.agent_name
        profile = getattr(session, "profile", None)

        # Conversation lane: per-session (one session == one conversation).
        conv = cls(
            store=session._store,
            agent_name=agent_name,
            profile=profile,
            lane="conversation",
            conversation_id=getattr(session, "conversation_id", None),
        )
        conv = conv if conv.has_enabled() else None

        # Agent lane: shared per-agent (one instance per agent_name).
        agent = _agent_coordinators.get(agent_name)
        if agent is None:
            candidate = cls(
                store=session._store,
                agent_name=agent_name,
                profile=profile,
                lane="agent",
            )
            if candidate.has_enabled():
                agent = candidate
                _agent_coordinators[agent_name] = agent

        lanes = BackgroundLanes(agent=agent, conversation=conv)
        if not lanes.enabled():
            return None
        return lanes

    # -- task registry / subscriber fan-out -------------------------------

    @property
    def memory_queue(self):
        """The lazily-built background consolidation queue (None until first use)."""
        return self._memory_queue

    @property
    def thread_summarizer_queue(self):
        """The lazy thread queue (None when the profile leaves it OFF)."""
        return self._thread_queue

    def subscribe_tasks(
        self, cb: Callable[[ProcessEvent], Any]
    ) -> Callable[[ProcessEvent], Any]:
        """Register a consumer for background-task lifecycle.

        The callback is invoked on every ``ProcessEvent`` transition
        (queued/running/done/error/skipped) for every registered task, via the
        shared per-agent emitter. On subscribe, ALREADY-ACTIVE tasks (status
        queued|running) are REPLAYED synchronously first, so a late-joining
        connection (e.g. a web tab that opens mid-fold) sees ``running`` instead
        of missing the transition.

        Returns the callback unchanged (convenient: ``cb = s.subscribe_tasks(cb)``).
        """
        self._emitter.on_any(cb)
        for task in self._background_tasks.values():
            if task.status in (QUEUED, PROCESSING):
                cb(self._to_process_event(task))
        return cb

    def unsubscribe_tasks(self, cb: Callable[[ProcessEvent], Any]) -> None:
        """Deregister a consumer previously registered via :meth:`subscribe_tasks`.

        Idempotent and best-effort: a callback that isn't registered (or was
        already removed) is a no-op. Used by long-lived server-side feeds (the
        dedicated ``/api/processes/stream``) so a disconnecting client doesn't
        leak a subscriber that keeps pushing into a dead queue.
        """
        self._emitter.off_any(cb)

    def task_history(self, conversation_id: str | None = None) -> list[BackgroundTask]:
        """Background-task rows, newest first.

        Includes terminal ``error`` rows (e.g. a failed thread summarizer) so a
        UI can surface "a background task failed for this conversation" even
        though the completion lands AFTER the turn's SSE stream closes. Task
        ids are prefixed ``{kind}.{conversation_id}.{seq}``; we match on the
        conversation segment (bounds-checked so a cid prefix can't over-match).
        ``conversation_id=None`` (the agent lane) returns ALL rows — the agent
        lane's ids are agent-scoped, so there is no conversation to filter by.
        """
        if conversation_id is None:
            rows = list(self._background_tasks.values())
        else:
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

        The recent-outcome source is the DURABLE run journal (``_run_events``,
        ``runs.jsonl``), not the in-memory task registry: the last line
        written for ``{prefix}.{conversation_id}`` is authoritative (file line
        order is truth, so no (seq, timestamp) tie-break is needed). ``prefix``
        is the task-id namespace (``consolidation``, ``thread_summarize``). Read-only: never enqueues, mutates the
        registry, or advances state.
        """
        if self._run_events is None:
            return row
        rec = self._run_events.recent(prefix, conversation_id)
        idle = self._with_last_run(row, prefix, conversation_id)
        if rec is None:
            return idle
        try:
            updated = datetime.fromisoformat(rec["timestamp"])
        except (TypeError, ValueError, KeyError):
            return idle
        # The journal always writes a real StepStatus, but a corrupt/older
        # line could omit it — in that case fall back to plain idle rather
        # than passing None into JobIdleStatus.
        raw_status = rec.get("status")
        if raw_status is None:
            return idle
        status = cast(StepStatus, raw_status)
        # An ACTIVE task (queued/running) is live: its last publish timestamp
        # only records when the job was handed off, not its progress, so a long
        # fold (minutes on an LLM) must NEVER decay to "idle" just because the
        # heartbeat lands past the hold window. Surface it as running regardless
        # of age. Only TERMINAL outcomes obey the hold window, so a finished
        # run still reverts its row to idle once the outcome is stale.
        is_active = status in (QUEUED, PROCESSING)
        if (
            not is_active
            and (datetime.now(UTC) - updated).total_seconds() > _PROCESS_STATUS_HOLD_S
        ):
            return idle
        if is_active:
            # An in-flight fold carries NO idle cadence text: the
            # "n/N turns to a fold/summary" countdown describes how far the
            # NEXT cadence has progressed since the last done watermark — it is
            # unrelated to the job currently running and would mislead under a
            # `running` status. Use the record's own detail (empty while the
            # worker hasn't reported progress); never fall back to the idle
            # row's detail.
            return JobIdleStatus(
                id=row.id,
                label=row.label,
                status=status,
                progress=rec.get("progress"),
                detail=rec.get("detail") or "",
            )
        if status == DONE:
            # The finished run itself: its detail and lines are the result.
            return JobIdleStatus(
                id=row.id,
                label=row.label,
                status=status,
                progress=rec.get("progress"),
                detail=rec.get("detail") or row.detail,
                lines=tuple(rec.get("lines") or ()),
            )
        # A skip/error says nothing about what the job last did: keep that.
        return replace(
            idle,
            status=status,
            progress=rec.get("progress"),
            detail=rec.get("detail") or row.detail,
        )

    def _with_last_run(
        self, row: JobIdleStatus, prefix: str, conversation_id: str
    ) -> JobIdleStatus:
        """``row`` plus the job's last completed run, from the durable run journal.

        The finished outcome only holds the row for ``_PROCESS_STATUS_HOLD_S``,
        and the next turn's "skipped" record replaces it sooner than that. An
        idle row that still says what the job last did -- and what it changed
        -- keeps that result visible until the next run completes.
        """
        assert self._run_events is not None
        done = self._run_events.recent(prefix, conversation_id, status=DONE)
        if done is None:
            return row
        return replace(
            row,
            last=done.get("detail") or "",
            last_at=done.get("timestamp"),
            lines=tuple(done.get("lines") or ()),
        )

    async def job_status_summary(
        self, conversation_id: str | None = None
    ) -> list[JobIdleStatus]:
        """READ-ONLY idle status for the registered background processes (Processes panel).

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

        ``conversation_id=None`` (the agent lane) scopes rows by agent name
        (``consolidation.<agent>``), matching the agent lane's agent-scoped
        task ids, and reads runs across all conversations.

        The store/history seam is resolved ONCE and shared across every enabled
        process's ``idle_row`` (a single pass over persisted runs), then each
        registered process computes its own row.
        """
        from paf.capabilities._turns import user_anchor_indices
        from paf.capabilities.persistence import build_history_source

        # For the agent lane (None), scope rows by agent name so ids are
        # agent-scoped (consolidation.<agent>), matching agent-scoped task ids.
        scope = conversation_id if conversation_id is not None else self.agent_name

        rows: list[JobIdleStatus] = []
        source = None
        tip_run = None
        tip_turn = 0

        try:
            source = build_history_source(self._store)
            # A single pass over persisted runs/history feeds all idle rows.
            runs = await self._store.list_runs(conversation_id=conversation_id)
            tip_run = runs[-1].run_id if runs else None
            tip_history = await source.run_history(run_id=tip_run) if tip_run else []
            anchors = user_anchor_indices(tip_history)
            tip_turn = len(anchors) - 1 if anchors else 0
        except Exception:
            # The shared store/history seam is down (or this store can't build
            # a history source): report every ENABLED process as "unavailable"
            # rather than raising — the feed must never tear the SSE socket on a
            # status read.
            for p in self._processes:
                # Only processes that have an idle row are shown in the panel.
                if p.enabled is not None and p.enabled(self) and p.idle_row is not None:
                    rows.append(
                        JobIdleStatus(
                            id=f"{p.id_namespace}.{scope}",
                            label=p.label,
                            detail="idle · status unavailable",
                        )
                    )
            return rows

        ctx = _IdleContext(source=source, tip_run=tip_run, tip_turn=tip_turn)
        for p in self._processes:
            if p.enabled is not None and p.enabled(self) and p.idle_row is not None:
                row = await p.idle_row(scope, ctx)
                if row is not None:
                    rows.append(row)
        return rows

    async def _span_count(self, source, wm, tip_run, tip_turn, conversation_id) -> int:
        """Number of distinct complete user turns newer than the watermark.

        Mirrors the summarizer trigger's own gate (``distinct_user_turn_texts``),
        so the idle ``n/BLOCK_TURNS`` displayed always means the same thing the
        trigger tests. A fresh/empty watermark resolves the whole completed
        history (``resolve_unseen_span`` handles it), so it reports the real
        accumulated count instead of short-circuiting to 0 forever.
        """
        from paf.capabilities._turns import distinct_user_turn_texts
        from paf.capabilities.memory._background import resolve_unseen_span

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

    async def _consolidation_idle_row(
        self, conversation_id: str, ctx: _IdleContext
    ) -> JobIdleStatus:
        """Idle row for memory consolidation (agent-scoped, TURN-count cadence)."""
        from paf.capabilities.memory._watermark import MemoryWatermark

        try:
            wm = MemoryWatermark.load(self._memory_directory)
            # Memory is per-agent: count turns across ALL conversations (None =
            # all runs), matching the cross-conversation trigger gate — not this
            # conversation's turns alone.
            n = await self._span_count(ctx.source, wm, ctx.tip_run, ctx.tip_turn, None)
            reason = (
                f"{n}/{self.consolidate_turns} turns to a fold"
                if n < self.consolidate_turns
                else "due for a fold"
            )
            return self._reconcile_status(
                JobIdleStatus(
                    id=f"consolidation.{conversation_id}",
                    label="Consolidating memory",
                    detail=f"idle · {reason}",
                ),
                "consolidation",
                conversation_id,
            )
        except Exception:
            return JobIdleStatus(
                id=f"consolidation.{conversation_id}",
                label="Consolidating memory",
                detail="idle · status unavailable",
            )

    async def _thread_idle_row(
        self, conversation_id: str, ctx: _IdleContext
    ) -> JobIdleStatus:
        """Idle row for the thread summarizer (per-conversation, turn-count cadence)."""
        from paf.capabilities.memory._watermark import MemoryWatermark
        from paf.capabilities.thread_summarizer._background import BLOCK_TURNS
        from paf.capabilities.thread_summarizer._store import (
            THREAD_WATERMARK_FILENAME,
            ThreadSummaryStore,
        )

        try:
            # Derive the SAME conversation-scoped dir + run-anchored watermark
            # the trigger/worker use (conversation_key hashed dir, run_id), or
            # the displayed n/BLOCK_TURNS diverges from the real gate.
            store = ThreadSummaryStore(self._thread_capture_root, conversation_id)
            wm = MemoryWatermark.load(
                store.directory,
                run_id=ctx.tip_run or "",
                filename=THREAD_WATERMARK_FILENAME,
            )
            n = await self._span_count(
                ctx.source, wm, ctx.tip_run, ctx.tip_turn, conversation_id
            )
            reason = (
                f"{n}/{BLOCK_TURNS} turns to a summary"
                if n < BLOCK_TURNS
                else "due for a summary"
            )
            return self._reconcile_status(
                JobIdleStatus(
                    id=f"thread_summarize.{conversation_id}",
                    label="Summarizing thread",
                    detail=f"idle · {reason}",
                ),
                "thread_summarize",
                conversation_id,
            )
        except Exception:
            return JobIdleStatus(
                id=f"thread_summarize.{conversation_id}",
                label="Summarizing thread",
                detail="idle · status unavailable",
            )

    async def _publish_task(self, task: BackgroundTask) -> None:
        """Store a task in the registry and emit its transition on the shared bus.

        A failing subscriber must never break the session/coordinator: the
        emitter logs and skips a raising listener.
        """
        task.updated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        self._background_tasks[task.id] = task
        if self._run_events is not None:
            self._run_events.append(self._journal_event(task))
        await self._emitter.emit_process(self._to_process_event(task))

    def _to_process_event(self, task: BackgroundTask) -> ProcessEvent:
        """Map a registry task to the shared live-event payload (lane + scope).

        ``lane`` is this coordinator's lane; ``scope`` is the conversation id
        for a conversation-lane coordinator, or None for the agent lane.
        """
        return ProcessEvent(
            id=task.id,
            label=task.label,
            status=ProcessEventType(task.status),
            progress=task.progress,
            detail=task.detail,
            lane=cast(Literal["agent", "conversation"], self.lane or "agent"),
            scope=self.conversation_id,
            lines=list(task.lines),
        )

    def _journal_event(self, task: BackgroundTask) -> dict:
        """The lossless event record for one task status transition.

        Storage is dumb: it records whatever status the task has at this
        publish, with no interpretation. ``run_id`` is the task id (the
        ``{namespace}.{cid}.{seq}`` key); ``process_id`` is its namespace;
        ``conversation_id`` is the middle segment (None for a global task).
        """
        parts = task.id.split(".")
        return {
            "run_id": task.id,
            "process_id": parts[0],
            "status": task.status,
            "detail": task.detail,
            "progress": task.progress,
            "lines": list(task.lines),
            "agent": self.agent_name,
            "conversation_id": parts[1] if len(parts) >= 3 else None,
            "timestamp": task.updated_at,
        }

    # -- queue builders (lazy) ---------------------------------------------

    def _ensure_memory_queue(self):
        """Build the self-contained worker queue, bound to ``self._store``."""
        if self._memory_queue is None:
            from paf.capabilities.memory._background import (
                FoldOutcome,
                MemoryConsolidationQueue,
                resolve_unseen_span,
            )
            from paf.capabilities.memory._watermark import MemoryWatermark
            from paf.capabilities.persistence import build_history_source

            async def _complete(
                kind: StepStatus, detail: str = "", lines: list[str] | None = None
            ) -> None:
                for task_id, task in list(self._background_tasks.items()):
                    if task_id.startswith("consolidation.") and task.status in (
                        QUEUED,
                        PROCESSING,
                    ):
                        task.status = kind
                        task.detail = detail
                        task.lines = list(lines or [])
                        task.progress = 1.0 if kind == DONE else None
                        await self._publish_task(task)

            async def _progress(detail: str) -> None:
                """Update the running row's detail without finishing it."""
                for task_id, task in list(self._background_tasks.items()):
                    if task_id.startswith("consolidation.") and task.status in (
                        QUEUED,
                        PROCESSING,
                    ):
                        task.detail = detail
                        await self._publish_task(task)

            # The fold's count, for the review-phase and finished details.
            folded = {"n": 0}

            async def on_reviewing() -> None:
                await _progress(f"{folded['n']} records folded · reviewing new facts")

            async def on_finished(outcome: FoldOutcome) -> None:
                """Finish the row once the whole job -- fold AND review -- is over."""
                detail = f"{outcome.records} records folded"
                lines: list[str] = []
                if outcome.error is not None:
                    await _complete(ERROR, f"{detail}, then failed: {outcome.error}")
                    return
                if outcome.review is not None:
                    from paf.capabilities.memory._pairwise import describe_actions

                    detail += f" · {outcome.review.outcome()}"
                    lines = describe_actions(outcome.review, self._memory_directory)
                elif outcome.review_error is not None:
                    detail += f" · review failed: {outcome.review_error}"
                await _complete(DONE, detail, lines)

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
                            messages,
                            directory,
                            watermark,
                            project=queue_ref.current_project,
                        )
                    # Not done yet: the pairwise reviewer may still run in this
                    # job. `on_finished` closes the row once the job is over.
                    folded["n"] = n
                    await _progress(f"{n} records folded")
                    return n
                except Exception as exc:
                    await _complete(ERROR, str(exc))
                    raise

            queue = MemoryConsolidationQueue(
                memory_directory=self._memory_directory,
                agent_name=self.agent_name,
                span_source=span_source,
                consolidate_span=publish_fold,
                pairwise_review=self.pairwise_review,
                on_reviewing=on_reviewing,
                on_finished=on_finished,
            )
            queue.history_source = source
            queue_ref = queue
            self._memory_queue = queue
        return self._memory_queue

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

    def _ensure_keyword_queue(self):
        """Build the per-agent keyword-curation worker lazily."""
        if self._keyword_queue is None:
            from paf.capabilities.keyword_curator import KeywordCuratorQueue

            self._keyword_queue = KeywordCuratorQueue(
                pass_fn=self._keyword_pass,
                on_complete=self._complete_keyword_curate,
            )
        return self._keyword_queue

    async def _complete_keyword_curate(
        self, agent_name: str, success: bool, detail: str
    ) -> None:
        """Publish completion for active keyword-curation task rows."""
        for task in self._background_tasks.values():
            if task.id.startswith(f"keyword_curate.{agent_name}.") and task.status in (
                QUEUED,
                PROCESSING,
            ):
                task.status = DONE if success else ERROR
                task.detail = detail
                task.progress = 1.0 if success else None
                await self._publish_task(task)

    def _keyword_settings(self) -> dict[str, Any]:
        if self._keyword_settings_cache is None:
            from paf.capabilities.keyword_curator import curate_settings

            self._keyword_settings_cache = curate_settings(self.agent_name)
        return self._keyword_settings_cache

    async def _maybe_trigger_keyword_curate(self, cid: str | None) -> None:
        """Enqueue a keyword-curation pass when sources changed (opt-in, throttled).

        Runs after every turn, so it is built to be silent and cheap in the common
        case: off unless the spec sets ``keywords.curate``; the source scan is
        throttled; a pass already in flight or inside the cooldown is skipped with
        no task row. A row is published only when a pass is actually enqueued. The
        worker is the sole writer of the keyword store, and a bounded
        ``curate_limit`` per pass means a large backlog is worked off over several
        passes rather than one expensive one.
        """
        del cid  # the curator is per-agent, not per-conversation
        if not self._keyword_curator_enabled:
            return
        settings = self._keyword_settings()
        if not settings["curate"]:
            return
        now = time.monotonic()
        if now - self._keyword_last_check < _KEYWORD_CHECK_INTERVAL_S:
            return
        self._keyword_last_check = now
        if now - self._keyword_last_enqueue < settings["curate_cooldown_s"]:
            return
        queue = self._ensure_keyword_queue()
        # Single-flight on actual work. `status().running` only says the worker
        # task is alive, and a worker idles for seconds after a job finishes —
        # that must not block the next pass.
        state = queue.get_state(self.agent_name)
        if state.active_jobs > 0 or not state.queue.empty():
            return
        from paf.capabilities.keyword_curator import (
            KeywordJob,
            keyword_dir_for,
            pending_changes,
            scan_sources,
            sources_config_for,
        )

        try:
            cfg = sources_config_for(self.agent_name)
            directory = keyword_dir_for(self.agent_name)
            sources = await asyncio.to_thread(scan_sources, **cfg)
            changed, vanished = pending_changes(directory, sources)
        except Exception:  # noqa: BLE001 - a scan failure must never break the turn
            log.exception("keyword curator scan failed agent=%s", self.agent_name)
            return
        if not changed and not vanished:
            return
        self._keyword_seq += 1
        task = BackgroundTask(
            id=f"keyword_curate.{self.agent_name}.{self._keyword_seq}",
            label="Curating keywords",
        )
        await self._publish_task(task)  # queued
        try:
            queue.enqueue(
                self.agent_name,
                KeywordJob(
                    agent_name=self.agent_name,
                    directory=str(directory),
                    sources_config=cfg,
                    limit=int(settings["curate_limit"]),
                ),
            )
            self._keyword_last_enqueue = now
            task.status = PROCESSING
            task.detail = f"{len(changed)} changed, {len(vanished)} gone"
            await self._publish_task(task)
        except Exception as exc:  # noqa: BLE001 - never break the turn
            task.status = ERROR
            task.detail = str(exc)
            await self._publish_task(task)

    async def _complete_thread_summarize(
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
                await self._publish_task(task)

    # -- post-turn triggers (the single seam) ------------------------------

    async def after_turn(self, cid: str | None) -> None:
        """Fire the post-turn background triggers for one conversation.

        The single seam ``runtime.Session.stream()`` calls at the end of a turn.
        Dispatches to every ENABLED registered process's trigger; each schedules
        its job when its cadence/watermark says it's due. The workers are the
        sole watermark writers. Never raises -- a failure in any trigger is
        captured on that task's row.
        """
        for p in self._processes:
            if p.enabled is not None and p.enabled(self) and p.trigger is not None:
                await p.trigger(cid)

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
        await self._publish_task(task)  # queued
        try:
            runs = await self._store.list_runs(conversation_id=cid)
            if not runs:
                task.status = SKIPPED
                task.detail = "no persisted runs yet"
                await self._publish_task(task)
                return
            tip_run = runs[-1].run_id
            tip_history = await source.run_history(run_id=tip_run)
            anchors = user_anchor_indices(tip_history)
            if not anchors:
                task.status = SKIPPED
                task.detail = "no persisted turns"
                await self._publish_task(task)
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
                await self._publish_task(task)
                return
            # Single-flight: skip if a thread summarize is already queued/running.
            queue = self._ensure_thread_summarizer_queue()
            cst = queue.status(cid)
            if cst.queued > 0 or cst.running:
                task.status = SKIPPED
                task.detail = "thread summarization already in flight"
                await self._publish_task(task)
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
            await self._publish_task(task)
        except (
            Exception
        ) as exc:  # trigger path itself failed (rare); never break the turn
            task.status = ERROR
            task.detail = str(exc)
            await self._publish_task(task)

    async def _maybe_trigger_consolidation(self, cid: str | None) -> None:
        """Fire the background-consolidation hook, publishing task lifecycle.

        Consumer #1 of the ``BackgroundEvent`` surface (mirrors coding_agent
        ``MaintenanceStep``). First publish a ``queued`` task row, then fire the
        trigger. The trigger builds the queue (and enqueues) ONLY when the
        watermark says enough turns have accumulated; if the queue is still None,
        no job is due this turn -- mark skipped, don't leave a phantom "running"
        row.

        ``cid`` here is ALREADY profile-scoped by the caller (``scoped_id``).
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
        task_id = f"consolidation.{self.agent_name}.{self._consolidation_seq}"
        task = BackgroundTask(id=task_id, label="Consolidating memory")
        await self._publish_task(task)  # queued
        try:
            runs = await self._store.list_runs(conversation_id=cid)
            if not runs:
                task.status = SKIPPED
                task.detail = "no persisted runs yet"
                await self._publish_task(task)
                return
            tip_run = runs[-1].run_id
            tip_history = await source.run_history(run_id=tip_run)
            anchors = user_anchor_indices(tip_history)
            if not anchors:
                task.status = SKIPPED
                task.detail = "no persisted turns"
                await self._publish_task(task)
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
                await self._publish_task(task)
                return
            # Single-flight: skip if a consolidation is already queued or running
            # so a slow worker behind the cadence gate can't pile up duplicate folds.
            mq = self._ensure_memory_queue()
            mst = mq.status(self.agent_name)
            if mst.queued > 0 or mst.running:
                task.status = SKIPPED
                task.detail = "consolidation already in flight"
                await self._publish_task(task)
                return
            from paf.conversations import project_for_conversation

            mq.enqueue(
                self.agent_name,
                ConsolidationJob(
                    run_id=tip_run,
                    turn_id=tip_turn,
                    messages=span.messages,
                    watermark=span.watermark,
                    # The span is one conversation's, so it has one project;
                    # the extractor stamps it on project-scoped facts.
                    project=project_for_conversation(
                        self.agent_name,
                        cid,
                        profile=getattr(self.profile, "name", None),
                    ),
                ),
            )
            task.status = PROCESSING  # job handed to the worker
            await self._publish_task(task)
        except Exception as exc:  # trigger path itself failed (rare)
            task.status = ERROR
            task.detail = str(exc)
            await self._publish_task(task)


# ---------------------------------------------------------------------------
# BackgroundLanes — the two-lane facade a Session holds as ``session.background``.
#
# Wraps the agent lane (shared per-agent: memory consolidation) and the
# conversation lane (per-session: thread summarizer + future conversation-scoped
# processes) behind one surface so the runtime and UI call sites are unchanged.
# Each lane is a BackgroundCoordinator; the facade merges their task history,
# idle summary, and subscriber fan-out, and dispatches after_turn to both.
# ---------------------------------------------------------------------------

# Shared per-agent coordinators (one agent lane per agent_name, never per-session).
_agent_coordinators: dict[str, BackgroundCoordinator] = {}


class BackgroundLanes:
    """Two-lane background container: agent lane + conversation lane.

    ``session.background`` is this container (or ``None``). It holds the two
    independent coordinators (``.agent`` and ``.conversation``) and fans out
    the post-turn trigger to both. It does NOT merge their read surfaces —
    ``task_history``/``job_status_summary``/``subscribe_tasks`` stay per-lane,
    and the feed reads each lane directly. The agent lane is shared per-agent;
    the conversation lane is per-session.
    """

    def __init__(
        self,
        agent: BackgroundCoordinator | None,
        conversation: BackgroundCoordinator | None,
    ) -> None:
        self.agent = agent
        self.conversation = conversation

    def enabled(self) -> bool:
        """True if either lane hosts an enabled process."""
        return self.agent is not None or self.conversation is not None

    async def after_turn(self, cid: str | None) -> None:
        """Fire both lanes' post-turn triggers for one conversation."""
        if self.agent is not None:
            await self.agent.after_turn(cid)
        if self.conversation is not None:
            await self.conversation.after_turn(cid)
