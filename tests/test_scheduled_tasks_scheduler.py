"""Tests for the scheduled-task scheduler (Phase 2).

Covers the ``next_fire`` cron helper and the ``ScheduledTaskScheduler`` loop:
fires due jobs, sleeps until next due, wakes on change, cap backstop, and the
single-flight / restart-safety properties.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from datetime import UTC, datetime, timedelta

import pytest

from paf.capabilities.scheduled_tasks import ScheduledJob, ScheduledTaskStore
from paf.capabilities.scheduled_tasks._schedule import next_fire
from paf.capabilities.scheduled_tasks._scheduler import ScheduledTaskScheduler

T0 = datetime(2026, 10, 8, 8, 59, tzinfo=UTC)


class FakeClock:
    """A mutable clock the scheduler reads; tests advance it manually."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakeQueue:
    """Records ``enqueue(session_id, job)`` calls instead of running them."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, ScheduledJob]] = []

    def enqueue(self, session_id: str, job: ScheduledJob) -> None:
        self.enqueued.append((session_id, job))


class AdvancingScheduler(ScheduledTaskScheduler):
    """Test scheduler whose ``_sleep`` advances the fake clock by `delay`.

    Simulates the timeout branch of ``_sleep`` deterministically, so the loop
    can be driven without real-time waits.
    """

    def __init__(self, root, *, clock: FakeClock, **kw) -> None:
        super().__init__(root, clock=clock, **kw)
        self.sleeps: list[float] = []

    async def _sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self._clock.advance(delay)
        await asyncio.sleep(0)


async def _run_loop_async(sched: ScheduledTaskScheduler, yields: int = 5) -> None:
    task = asyncio.create_task(sched.run())
    for _ in range(yields):
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def _run_loop(sched: ScheduledTaskScheduler, yields: int = 5) -> None:
    asyncio.run(_run_loop_async(sched, yields))


# --- next_fire (p4a) ----------------------------------------------------------
def test_next_fire_is_strictly_after_base() -> None:
    nxt = next_fire("0 9 * * *", T0)
    assert nxt > T0
    assert nxt.tzinfo is not None


def test_next_fire_daily_cron() -> None:
    # 8:59 -> next 9:00 is today
    assert next_fire("0 9 * * *", T0) == datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
    # 9:01 -> next 9:00 is tomorrow
    after = datetime(2026, 10, 8, 9, 1, tzinfo=UTC)
    assert next_fire("0 9 * * *", after) == datetime(2026, 10, 9, 9, 0, tzinfo=UTC)


def test_next_fire_invalid_cron_raises() -> None:
    with pytest.raises(ValueError):
        next_fire("not a cron", T0)


def test_next_fire_defaults_to_now() -> None:
    nxt = next_fire("0 9 * * *")
    assert nxt.tzinfo is not None


# --- loop: fires due jobs (p4b) ----------------------------------------------
def test_fires_due_job_and_stamps_last_run() -> None:
    clock = FakeClock(T0)
    queue = FakeQueue()
    with tempfile.TemporaryDirectory() as tmp:
        store = ScheduledTaskStore(tmp, agent="a")
        job = ScheduledJob(
            schedule="0 9 * * *", instructions="post", conversation_id="c1"
        )
        store.create_job(job)
        sched = AdvancingScheduler(tmp, agent="a", queue=queue, clock=clock, cap=3600)
        _run_loop(sched)
        assert len(queue.enqueued) == 1
        cid, fired = queue.enqueued[0]
        assert cid == "c1"
        assert fired.id == job.id
        assert store.get_job(job.id).last_run == datetime(2026, 10, 8, 9, 0, tzinfo=UTC)


def test_sleeps_until_next_due_not_fixed_interval() -> None:
    clock = FakeClock(T0)
    queue = FakeQueue()
    with tempfile.TemporaryDirectory() as tmp:
        store = ScheduledTaskStore(tmp, agent="a")
        store.create_job(
            ScheduledJob(
                schedule="0 9 * * *", instructions="post", conversation_id="c1"
            )
        )
        sched = AdvancingScheduler(tmp, agent="a", queue=queue, clock=clock, cap=3600)
        _run_loop(sched)
        # next due is 60s away (8:59 -> 9:00); the first sleep is 60, not the cap
        assert sched.sleeps[0] == 60


def test_skips_disabled_and_archived_jobs() -> None:
    clock = FakeClock(T0)
    queue = FakeQueue()
    with tempfile.TemporaryDirectory() as tmp:
        store = ScheduledTaskStore(tmp, agent="a")
        store.create_job(
            ScheduledJob(
                schedule="0 9 * * *",
                instructions="post",
                enabled=False,
                conversation_id="c1",
            )
        )
        store.create_job(
            ScheduledJob(
                schedule="0 9 * * *",
                instructions="archived",
                archived=True,
                conversation_id="c1",
            )
        )
        sched = AdvancingScheduler(tmp, agent="a", queue=queue, clock=clock, cap=3600)
        _run_loop(sched)
        assert queue.enqueued == []


# --- loop: wake on change + cap backstop (p4b) -------------------------------
def test_sleep_wakes_on_change() -> None:
    sched = AdvancingScheduler(
        "data/scheduled_tasks", agent="a", queue=FakeQueue(), clock=FakeClock(T0), cap=3600
    )
    sched._change.set()
    start = time.monotonic()
    asyncio.run(sched._sleep(5.0))
    assert time.monotonic() - start < 1.0  # returned early, not after 5s


def test_sleep_times_out_after_cap() -> None:
    sched = ScheduledTaskScheduler(
        "data/scheduled_tasks", agent="a", queue=FakeQueue(), clock=FakeClock(T0), cap=3600
    )
    start = time.monotonic()
    asyncio.run(sched._sleep(0.05))
    assert time.monotonic() - start >= 0.05  # waited the full cap


def test_loop_wakes_on_new_job() -> None:
    queue = FakeQueue()
    with tempfile.TemporaryDirectory() as tmp:
        sched = ScheduledTaskScheduler(tmp, agent="a", queue=queue, cap=3600)

        async def drive() -> None:
            task = asyncio.create_task(sched.run())
            await asyncio.sleep(0.01)
            # loop is sleeping (no jobs, cap=3600); create a job -> change wakes it
            store = sched._store_for("a")
            store.create_job(
                ScheduledJob(
                    schedule="* * * * * *",
                    instructions="post",
                    conversation_id="c1",
                )
            )
            for _ in range(200):
                if queue.enqueued:
                    break
                await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        asyncio.run(drive())
        assert queue.enqueued  # fired within ~1s, not after the 3600s cap


# --- single-flight / restart safety (p4c) ------------------------------------
def test_single_flight_fires_once() -> None:
    clock = FakeClock(T0)
    queue = FakeQueue()
    with tempfile.TemporaryDirectory() as tmp:
        store = ScheduledTaskStore(tmp, agent="a")
        store.create_job(
            ScheduledJob(
                schedule="0 9 * * *", instructions="post", conversation_id="c1"
            )
        )
        sched = AdvancingScheduler(tmp, agent="a", queue=queue, clock=clock, cap=3600)
        _run_loop(sched, yields=10)
        # daily job fires exactly once across many loop iterations
        assert len(queue.enqueued) == 1


def test_restart_skips_missed_fire() -> None:
    # Job fires daily at 9:00; last ran yesterday. Now is 10:00 (past today's
    # 9:00 fire). On restart the scheduler must NOT catch up today's missed
    # fire — next fire is tomorrow 9:00.
    clock = FakeClock(datetime(2026, 10, 8, 10, 0, tzinfo=UTC))
    queue = FakeQueue()
    with tempfile.TemporaryDirectory() as tmp:
        store = ScheduledTaskStore(tmp, agent="a")
        job = ScheduledJob(
            schedule="0 9 * * *", instructions="post", conversation_id="c1"
        )
        store.create_job(job)
        store.update_job(job.id, last_run=datetime(2026, 10, 7, 9, 0, tzinfo=UTC))
        sched = AdvancingScheduler(tmp, agent="a", queue=queue, clock=clock, cap=3600)
        _run_loop(sched, yields=5)  # advances to 15:00, still before tomorrow 9:00
        assert queue.enqueued == []


# --- single loop scans all agents (per-agent master lists) -------------------
def test_single_loop_scans_all_agents() -> None:
    # The scheduler is ONE loop, not per-agent: with agent=None it scans every
    # agent's master list and fires each due job into its own conversation.
    clock = FakeClock(T0)
    queue = FakeQueue()
    with tempfile.TemporaryDirectory() as tmp:
        store_a = ScheduledTaskStore(tmp, agent="a")
        store_b = ScheduledTaskStore(tmp, agent="b")
        job_a = ScheduledJob(
            schedule="0 9 * * *", instructions="a", conversation_id="c1"
        )
        job_b = ScheduledJob(
            schedule="0 9 * * *", instructions="b", conversation_id="c2"
        )
        store_a.create_job(job_a)
        store_b.create_job(job_b)
        sched = AdvancingScheduler(tmp, agent=None, queue=queue, clock=clock, cap=3600)
        _run_loop(sched)
        fired = {cid: j.id for cid, j in queue.enqueued}
        assert fired == {"c1": job_a.id, "c2": job_b.id}


def test_agent_scoped_loop_ignores_other_agents() -> None:
    # agent="a" scopes the loop to just that agent's master list.
    clock = FakeClock(T0)
    queue = FakeQueue()
    with tempfile.TemporaryDirectory() as tmp:
        store_a = ScheduledTaskStore(tmp, agent="a")
        store_b = ScheduledTaskStore(tmp, agent="b")
        store_a.create_job(
            ScheduledJob(
                schedule="0 9 * * *", instructions="a", conversation_id="c1"
            )
        )
        store_b.create_job(
            ScheduledJob(
                schedule="0 9 * * *", instructions="b", conversation_id="c2"
            )
        )
        sched = AdvancingScheduler(tmp, agent="a", queue=queue, clock=clock, cap=3600)
        _run_loop(sched)
        assert [cid for cid, _ in queue.enqueued] == ["c1"]
