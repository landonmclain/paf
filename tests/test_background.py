"""Engine tests for paf.background — concurrency semantics with a synthetic job type.

These tests exercise the generic engine only (no memory import) so they stay
valid even as memory consumers evolve. A simple ``EchoQueue`` subclass records
the order jobs were processed so tests can assert FIFO and concurrency
behaviour.

NOTE: the repo has no pytest-asyncio plugin (and runs the default suite with
``-m 'not live'``). The established convention is sync ``def test_*`` functions
that drive async scenarios via ``asyncio.run(...)`` (see ``tests/test_runtime.py``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC

from paf.background import BackgroundQueueService, QueueState, QueueStatus


@dataclass
class EchoJob:
    value: str


class EchoQueue(BackgroundQueueService[EchoJob]):
    """Synthetic queue: records processing order.

    Plain class (not a dataclass) — a consumer that subclasses
    :class:`BackgroundQueueService` must call ``super().__init__()`` so the
    engine's ``_states`` dict is created. Declaring this a ``@dataclass`` with
    its own fields would generate an ``__init__`` that skips the base.
    """

    def __init__(self, name: str = "echo") -> None:
        super().__init__()
        self.name = name
        self.processed: list[str] = []

    async def process_job(
        self,
        session_id: str,
        job: EchoJob,
        state: QueueState[EchoJob],
    ) -> None:
        self.processed.append(job.value)


def test_enqueue_processes_job() -> None:
    processed: list[str] = []

    class Recorder(EchoQueue):
        async def process_job(self, session_id, job, state) -> None:
            processed.append(job.value)

    async def scenario() -> None:
        q = Recorder()
        q.enqueue("s1", EchoJob("a"))
        await q.drain("s1")

    asyncio.run(scenario())
    assert processed == ["a"]


def test_fifo_order_within_session() -> None:
    processed: list[str] = []

    class Recorder(EchoQueue):
        async def process_job(self, session_id, job, state) -> None:
            processed.append(job.value)

    async def scenario() -> None:
        q = Recorder()
        for v in ["a", "b", "c"]:
            q.enqueue("s1", EchoJob(v))
        await q.drain("s1")

    asyncio.run(scenario())
    assert processed == ["a", "b", "c"]


def test_sessions_are_isolated() -> None:
    processed: list[str] = []

    class Recorder(EchoQueue):
        async def process_job(self, session_id, job, state) -> None:
            processed.append(job.value)

    async def scenario() -> None:
        q = Recorder()
        q.enqueue("s1", EchoJob("a"))
        q.enqueue("s2", EchoJob("b"))
        await asyncio.gather(q.drain("s1"), q.drain("s2"))

    asyncio.run(scenario())
    assert processed == ["a", "b"]


def test_status_reports_queued_and_not_running() -> None:
    async def scenario() -> tuple[EchoQueue, QueueStatus]:
        q = EchoQueue()
        return q, q.status("s1")

    _q, status = asyncio.run(scenario())
    assert status.name == "echo"
    assert status.queued == 0
    assert status.running is False
    assert status.idle is True


def test_drain_with_timeout_returns_false_for_long_job() -> None:
    class NeverQueue(BackgroundQueueService[str]):
        name = "never"

        async def process_job(self, session_id, job, state) -> None:
            await asyncio.sleep(30)

    async def scenario() -> bool:
        q = NeverQueue()
        q.enqueue("s1", "job")
        result = await q.drain_with_timeout("s1", 0.05)
        # cancel the pending worker so the loop closes cleanly
        state = q.get_state("s1")
        if state.worker_task:
            state.worker_task.cancel()
            try:
                await state.worker_task
            except (asyncio.CancelledError, Exception):
                pass
        return result

    assert asyncio.run(scenario()) is False


def test_active_jobs_reflects_running_processing() -> None:
    seen: list[str] = []
    release = asyncio.Event()

    class SlowQueue(BackgroundQueueService[str]):
        name = "slow"

        async def process_job(self, session_id, job, state) -> None:
            seen.append(job)
            await release.wait()

    async def scenario() -> tuple[QueueState[str], list[str]]:
        q = SlowQueue()
        q.enqueue("s1", "j1")
        q.enqueue("s1", "j2")
        await asyncio.sleep(0)  # let the worker start j1
        state = q.get_state("s1")

        attempts = 0
        while state.active_jobs == 0 and attempts < 10:
            await asyncio.sleep(0.01)
            attempts += 1
        assert state.active_jobs == 1
        assert state.queue.qsize() == 1
        status = q.status("s1")
        assert status.queued == 1
        assert status.running is True
        assert status.idle is False

        release.set()  # let j1 finish; worker advances to j2
        await q.drain("s1")
        return state, seen

    state, seen = asyncio.run(scenario())
    assert seen == ["j1", "j2"]
    assert state.active_jobs == 0


def test_worker_idles_out_after_timeout() -> None:
    class FastIdleQueue(BackgroundQueueService[str]):
        name = "fastidle"
        IDLE_TIMEOUT_S = 0.05

        async def process_job(self, session_id, job, state) -> None:
            pass  # nothing to do; keep the engine warm for the timeout check

    async def scenario() -> asyncio.Task | None:
        q = FastIdleQueue()
        q.enqueue("s1", "j")  # starts the worker
        await q.drain("s1")
        # after draining, the worker returns to queue.get() and idle-exits
        await asyncio.sleep(0.2)
        state = q.get_state("s1")
        return state.worker_task

    worker = asyncio.run(scenario())
    assert worker is not None
    assert worker.done() is True


def test_two_independent_queues_run_concurrently() -> None:
    """Separate instances == concurrent workers (the composition model)."""

    class TaggedQueue(EchoQueue):
        pass

    async def drive() -> tuple[list[str], list[str]]:
        a = TaggedQueue(name="a")
        b = TaggedQueue(name="b")

        async def run(q: TaggedQueue, tag: str) -> list[str]:
            for i in range(5):
                q.enqueue("s1", EchoJob(f"{tag}{i}"))
            await q.drain("s1")
            return q.processed

        return await asyncio.gather(run(a, "A"), run(b, "B"))

    ra, rb = asyncio.run(drive())
    assert ra == ["A0", "A1", "A2", "A3", "A4"]
    assert rb == ["B0", "B1", "B2", "B3", "B4"]


class _NoRunsStore:
    """Store duck-type returning no runs (fresh conversation)."""

    async def list_runs(self, conversation_id=None):
        return []

    async def run_history(self, run_id):
        return []


def test_job_status_summary_returns_idle_rows_when_idle() -> None:
    """The Processes-panel snapshot exposes all 3 jobs as idle with a reason.

    A coordinator over a fresh/conversationless store (no runs, no watermarks)
    must still yield consolidation / review / thread summarizer idle rows with
    the stable ids the frontend coalesces transitions against — never raise.
    """
    from tempfile import mkdtemp

    from pydantic_ai_harness.step_persistence import FileStepStore

    from paf.background import BackgroundCoordinator
    from paf.capabilities import PROFILES

    bg = BackgroundCoordinator(
        FileStepStore(directory=mkdtemp()),  # real store, no runs yet
        agent_name="demo_assistant",
        profile=PROFILES.get("demo"),
        memory_directory="/nonexistent-memory-dir",  # loads a fresh watermark
    )

    rows = asyncio.run(bg.job_status_summary("cid123"))
    by_id = {r.id: r for r in rows}

    assert set(by_id) == {
        "consolidation.cid123",
        "review.cid123",
        "thread_summarize.cid123",
    }
    for r in rows:
        assert r.status == "idle"
        assert r.detail  # a human reason is always present


def test_job_status_summary_thread_row_reads_hashed_watermark_dir(monkeypatch) -> None:
    """The thread idle row must load the SAME hashed watermark the worker uses.

    Bug (2026-09-14, PAF-UI issue): the idle block built the capture dir from the
    raw ``conversation_id`` (``conversations/<cid>/``), which never exists --
    the worker writes under ``conversations/<sha256(cid)[:24]>/`` via
    ``ThreadSummaryStore``. So the idle watermark was always fresh -> every
    completed turn counted -> n >= BLOCK_TURNS -> 'due for a summary' forever,
    while a SKIPPED row showed the real count (why skipped rendered correctly).
    Pin the fix: job_status_summary must load the thread watermark from
    ThreadSummaryStore(...).directory AND with run_id anchoring.
    """
    from tempfile import mkdtemp

    from pydantic_ai_harness.step_persistence import FileStepStore

    from paf.background import BackgroundCoordinator
    from paf.capabilities import PROFILES
    from paf.capabilities.memory._watermark import MemoryWatermark
    from paf.capabilities.thread_summarizer._store import (
        THREAD_WATERMARK_FILENAME,
        ThreadSummaryStore,
    )

    calls: dict = {}

    def _spy_load(cls, directory, *, run_id="", filename=THREAD_WATERMARK_FILENAME):
        calls[(str(directory), run_id)] = True
        return MemoryWatermark()  # fresh -> n=0; we only assert WHERE it loads from

    monkeypatch.setattr(MemoryWatermark, "load", classmethod(_spy_load))

    # Temp thread capture root so we never touch real data/summaries.
    import tempfile

    temp_root = tempfile.mkdtemp(prefix="paf-summaries-")
    bg = BackgroundCoordinator(
        FileStepStore(directory=mkdtemp()),
        agent_name="demo_assistant",
        profile=PROFILES.get("demo"),
        memory_directory="/nonexistent-memory-dir",
    )
    bg._thread_capture_root = temp_root

    asyncio.run(bg.job_status_summary("cid123"))

    # The correct dir is ThreadSummaryStore(temp_root, cid).directory (hashed).
    correct = ThreadSummaryStore(temp_root, "cid123").directory
    wrong = f"{temp_root}/conversations/cid123"
    assert str(correct) in {d for (d, _) in calls}, (
        f"thread idle row did not load from ThreadSummaryStore dir {correct}; "
        f"loaded dirs: {sorted(d for (d, _) in calls)}"
    )
    assert not any(wrong in d for (d, _) in calls), (
        f"thread idle row still loads raw-cid dir {wrong!r}"
    )


def test_job_status_summary_degrades_when_store_fails() -> None:
    """A broken store must degrade to 'unavailable' idle rows, not raise.

    The feed calls this on connect; a failure must never tear the SSE socket.
    """
    from paf.background import BackgroundCoordinator
    from paf.capabilities import PROFILES

    class _BrokenStore(_NoRunsStore):
        async def list_runs(self, conversation_id=None):
            raise RuntimeError("store exploded")

    bg = BackgroundCoordinator(
        _BrokenStore(),
        agent_name="demo_assistant",
        profile=PROFILES.get("demo"),
        memory_directory="/nonexistent-memory-dir",
    )

    rows = asyncio.run(bg.job_status_summary("cid123"))  # must not raise
    assert len(rows) == 3
    assert all(r.status == "idle" for r in rows)
    assert all("unavailable" in r.detail for r in rows)


def test_reconcile_keeps_active_task_running_past_hold() -> None:
    """An in-flight (running) task must never decay to idle on the heartbeat.

    Regression for issue #3: the consolidation worker's fold takes minutes, but
    BackgroundTask.updated_at is only stamped once (at enqueue). The SSE idle
    heartbeat re-emits job_status_summary every 30s, and the old code decayed
    any task older than the 60s hold window to plain "idle" — so a long-running
    fold visibly flipped RUNNING -> IDLE while still working, then later DONE.

    _reconcile_status is the coalescing guard: an ACTIVE task (queued/running)
    must surface as running regardless of how old its last publish is. Only
    TERMINAL outcomes obey the hold window.
    """
    from datetime import datetime, timedelta
    from tempfile import mkdtemp

    from pydantic_ai_harness.step_persistence import FileStepStore

    from paf.background import (
        PROCESSING,
        BackgroundCoordinator,
        JobIdleStatus,
    )
    from paf.capabilities import PROFILES

    bg = BackgroundCoordinator(
        FileStepStore(directory=mkdtemp()),
        agent_name="demo_assistant",
        profile=PROFILES.get("demo"),
        memory_directory="/nonexistent-memory-dir",
        processes_root=mkdtemp(),
    )

    # The task was handed to the worker minutes ago and is STILL running (a long
    # LLM fold). updated_at is stamped once at enqueue -> long past the hold.
    # The durable journal is now the reconciliation source.
    bg._run_events.append(
        {
            "run_id": "consolidation.cid123.1",
            "process_id": "consolidation",
            "status": PROCESSING,
            "detail": "folding",
            "progress": None,
            "agent": "demo_assistant",
            "conversation_id": "cid123",
            "timestamp": (datetime.now(UTC) - timedelta(minutes=5)).isoformat(),
        }
    )

    idle_row = JobIdleStatus(
        id="consolidation.cid123",
        label="Consolidating memory",
        detail="idle · next ~3 min",
    )
    out = bg._reconcile_status(idle_row, "consolidation", "cid123")

    # Must carry the live running state + detail, NOT decay to idle.
    assert out.status == PROCESSING
    assert out.detail == "folding"


def test_reconcile_running_task_never_shows_idle_cadence_detail() -> None:
    """A running fold's row must NOT leak the idle turn-countdown as its detail.

    Regression found live (2026-09-14): mid-fold, the Processes panel showed
    ``running`` alongside ``idle · 3/5 turns to a fold``. The 3/5 countdown
    describes how far the NEXT cadence has progressed since the last fold
    watermark — it is unrelated to the fold currently running. The old
    ``detail = recent.detail or row.detail`` fallback let the empty detail of an
    in-flight task (worker hasn't reported progress yet) inherit the idle
    row's cadence text. An ACTIVE task must surface its OWN detail (may be
    empty) and never the idle cadence line.
    """
    from datetime import datetime
    from tempfile import mkdtemp

    from pydantic_ai_harness.step_persistence import FileStepStore

    from paf.background import (
        PROCESSING,
        BackgroundCoordinator,
        JobIdleStatus,
    )
    from paf.capabilities import PROFILES

    bg = BackgroundCoordinator(
        FileStepStore(directory=mkdtemp()),
        agent_name="demo_assistant",
        profile=PROFILES.get("demo"),
        memory_directory="/nonexistent-memory-dir",
        processes_root=mkdtemp(),
    )

    # A fold is running mid-work; the worker has NOT reported progress, so the
    # run carries NO detail of its own yet. Durable journal is the source.
    bg._run_events.append(
        {
            "run_id": "consolidation.cid123.3",
            "process_id": "consolidation",
            "status": PROCESSING,
            "detail": "",
            "progress": None,
            "agent": "demo_assistant",
            "conversation_id": "cid123",
            "timestamp": datetime.now(UTC).isoformat(),
        }
    )

    idle_row = JobIdleStatus(
        id="consolidation.cid123",
        label="Consolidating memory",
        status="idle",
        detail="idle · 3/5 turns to a fold",
    )
    out = bg._reconcile_status(idle_row, "consolidation", "cid123")

    # Status stays running; detail is the task's own (empty), NEVER the idle
    # cadence countdown.
    assert out.status == PROCESSING
    assert out.detail == ""
    assert "turns to a fold" not in out.detail


def test_reconcile_prefers_newer_seq_over_stale_skip_on_timestamp_tie() -> None:
    """A running fold (higher seq) beats a stale skipped row even at EQUAL ts.

    Regression found live (2026-09-14): the panel showed ``running`` +
    ``only 4 unsettled turn(s); need >= 5`` — the stale SKIPPED detail from
    the trigger that gated below threshold. _publish_task stamps ``updated_at``
    at microsecond resolution, so the skip and the fold that follows it within
    moments often share the SAME timestamp. The old selector compared only
    ``updated_at`` and on a tie kept the FIRST dict-iteration task — which was
    the stale skipped one. ``_reconcile_status`` must treat a later ``.seq``
    (monotonic per conversation) as authoritative.
    """
    from datetime import datetime
    from tempfile import mkdtemp

    from pydantic_ai_harness.step_persistence import FileStepStore

    from paf.background import (
        PROCESSING,
        SKIPPED,
        BackgroundCoordinator,
        JobIdleStatus,
    )
    from paf.capabilities import PROFILES

    bg = BackgroundCoordinator(
        FileStepStore(directory=mkdtemp()),
        agent_name="demo_assistant",
        profile=PROFILES.get("demo"),
        memory_directory="/nonexistent-memory-dir",
        processes_root=mkdtemp(),
    )
    now = datetime.now(UTC).replace(microsecond=0).isoformat()

    # seq 1: gated below 5 -> SKIPPED with the familiar message.
    # seq 2: threshold reached -> fold now running, no progress detail yet.
    # Both published in the same second (the norm: skip then fold in moments).
    # The durable journal resolves the tie by FILE LINE ORDER (append order):
    # the running fold is appended LAST, so it is the newest line and wins —
    # no (seq, timestamp) tie-break scan is needed anymore.
    bg._run_events.append(
        {
            "run_id": "consolidation.cid123.1",
            "process_id": "consolidation",
            "status": SKIPPED,
            "detail": "only 4 unsettled turn(s); need >= 5",
            "progress": None,
            "agent": "demo_assistant",
            "conversation_id": "cid123",
            "timestamp": now,
        }
    )
    bg._run_events.append(
        {
            "run_id": "consolidation.cid123.2",
            "process_id": "consolidation",
            "status": PROCESSING,
            "detail": "",
            "progress": None,
            "agent": "demo_assistant",
            "conversation_id": "cid123",
            "timestamp": now,
        }
    )

    idle_row = JobIdleStatus(
        id="consolidation.cid123", label="Consolidating memory", status="idle"
    )
    out = bg._reconcile_status(idle_row, "consolidation", "cid123")

    # The running fold (last line) wins: no stale skip text.
    assert out.status == PROCESSING
    assert out.detail == ""
