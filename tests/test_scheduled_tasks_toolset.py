"""Tests for the scheduled-tasks toolset + capability (Phase 4).

Covers the five agent-facing tools (``schedule_task`` / ``list_scheduled_tasks``
/ ``update_scheduled_task`` / ``remove_scheduled_task`` / ``run_scheduled_task``)
against a real store with a fake queue, plus the capability-wiring test
(cataloged + built via ``build_runtime_capabilities``).
"""

from __future__ import annotations

import asyncio
import tempfile

import pytest
from pydantic_ai.exceptions import ModelRetry

from paf.capabilities.scheduled_tasks import (
    ScheduledTasksToolset,
    ScheduledTaskStore,
)
from paf.capabilities.scheduled_tasks._types import NotifyMode


class FakeQueue:
    """BackgroundQueueService stand-in that records enqueues."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, object]] = []

    def enqueue(self, session_id: str, job) -> None:
        self.enqueued.append((session_id, job))


def _factory(tmp: str):
    stores: dict[str, ScheduledTaskStore] = {}

    def factory(cid: str) -> ScheduledTaskStore:
        del cid  # the store is per-agent; conversation is a column on the job
        if "a" not in stores:
            stores["a"] = ScheduledTaskStore(tmp, agent="a")
        return stores["a"]

    return factory


def _toolset(tmp: str, *, queue=None, cid: str = "c1") -> ScheduledTasksToolset:
    return ScheduledTasksToolset(
        store_factory=_factory(tmp), queue=queue, agent="a", conversation_id=cid
    )


# --- schedule_task ----------------------------------------------------------
def test_schedule_task_creates_agent_job() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ts = _toolset(tmp)
        out = asyncio.run(
            ts.schedule_task("0 9 * * *", instructions="post the article")
        )
        assert "created" in out
        job_id = out.split()[2]
        store = ts._store()
        job = store.get_job(job_id)
        assert job is not None
        assert job.instructions == "post the article"
        assert job.script is None
        assert job.notify == NotifyMode.always
        # the toolset stamps the job's agent + conversation columns
        assert job.agent == "a"
        assert job.conversation_id == "c1"


def test_schedule_task_creates_script_job() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ts = _toolset(tmp)
        out = asyncio.run(
            ts.schedule_task("0 9 * * *", script="echo hi", notify="on_trigger")
        )
        job_id = out.split()[2]
        job = ts._store().get_job(job_id)
        assert job is not None
        assert job.script == "echo hi"
        assert job.instructions is None
        assert job.notify == NotifyMode.on_trigger


def test_schedule_task_requires_exactly_one_mode() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ts = _toolset(tmp)
        with pytest.raises(ModelRetry):
            asyncio.run(ts.schedule_task("0 9 * * *"))
        with pytest.raises(ModelRetry):
            asyncio.run(
                ts.schedule_task("0 9 * * *", instructions="a", script="echo hi")
            )


def test_schedule_task_rejects_bad_notify() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ts = _toolset(tmp)
        with pytest.raises(ModelRetry):
            asyncio.run(
                ts.schedule_task("0 9 * * *", script="echo hi", notify="sometimes")
            )


# --- list_scheduled_tasks ---------------------------------------------------
def test_list_scheduled_tasks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ts = _toolset(tmp)
        assert asyncio.run(ts.list_scheduled_tasks()) == (
            "No scheduled tasks in this conversation."
        )
        asyncio.run(ts.schedule_task("0 9 * * *", instructions="a"))
        asyncio.run(ts.schedule_task("0 10 * * *", script="echo hi"))
        out = asyncio.run(ts.list_scheduled_tasks())
        assert "0 9 * * *" in out
        assert "0 10 * * *" in out


# --- update_scheduled_task --------------------------------------------------
def test_update_scheduled_task_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ts = _toolset(tmp)
        out = asyncio.run(ts.schedule_task("0 9 * * *", instructions="a"))
        job_id = out.split()[2]
        asyncio.run(ts.update_scheduled_task(job_id, schedule="0 11 * * *"))
        job = ts._store().get_job(job_id)
        assert job is not None
        assert job.schedule == "0 11 * * *"
        assert job.instructions == "a"


def test_update_scheduled_task_switches_mode() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ts = _toolset(tmp)
        out = asyncio.run(ts.schedule_task("0 9 * * *", script="echo hi"))
        job_id = out.split()[2]
        asyncio.run(ts.update_scheduled_task(job_id, instructions="now do this"))
        job = ts._store().get_job(job_id)
        assert job is not None
        assert job.instructions == "now do this"
        assert job.script is None  # mode switch cleared the script


def test_update_scheduled_task_missing_job_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ts = _toolset(tmp)
        with pytest.raises(ModelRetry):
            asyncio.run(ts.update_scheduled_task("nope", schedule="0 9 * * *"))


# --- remove_scheduled_task --------------------------------------------------
def test_remove_scheduled_task_archives() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ts = _toolset(tmp)
        out = asyncio.run(ts.schedule_task("0 9 * * *", instructions="a"))
        job_id = out.split()[2]
        assert asyncio.run(ts.remove_scheduled_task(job_id)) == (
            f"Removed scheduled task {job_id}."
        )
        assert ts._store().get_job(job_id).archived is True
        assert ts._store().list_jobs() == []  # archived excluded by default
        with pytest.raises(ModelRetry):
            asyncio.run(ts.remove_scheduled_task("nope"))


# --- run_scheduled_task -----------------------------------------------------
def test_run_scheduled_task_enqueues() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        queue = FakeQueue()
        ts = _toolset(tmp, queue=queue)
        out = asyncio.run(ts.schedule_task("0 9 * * *", instructions="a"))
        job_id = out.split()[2]
        assert asyncio.run(ts.run_scheduled_task(job_id)) == (
            f"Queued scheduled task {job_id} to run now."
        )
        assert len(queue.enqueued) == 1
        cid, job = queue.enqueued[0]
        assert cid == "c1"
        assert job.id == job_id


def test_run_scheduled_task_unwired_queue_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ts = _toolset(tmp, queue=None)
        out = asyncio.run(ts.schedule_task("0 9 * * *", instructions="a"))
        job_id = out.split()[2]
        with pytest.raises(ModelRetry):
            asyncio.run(ts.run_scheduled_task(job_id))


# --- capability wiring ------------------------------------------------------
def test_scheduled_tasks_cataloged_and_wired() -> None:
    from paf.capabilities.build import build_runtime_capabilities
    from paf.capabilities.profiles import AgentProfile
    from paf.capabilities.registry import CAPABILITY_TYPES

    assert "ScheduledTasks" in CAPABILITY_TYPES
    profile = AgentProfile(name="t", capability_names=("ScheduledTasks",))
    caps = build_runtime_capabilities(profile, agent_name="a")
    assert any(type(c).__name__ == "ScheduledTasks" for c in caps)
