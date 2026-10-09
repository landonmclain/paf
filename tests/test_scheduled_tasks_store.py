"""Tests for the scheduled-tasks store (Phase 1).

Covers the data model (``ScheduledJob`` / ``RunRecord`` / ``NotifyMode``) and
the ``ScheduledTaskStore``: master-list CRUD + soft-delete, run-history
append/read, and the change callback the scheduler will wake on.
"""

from __future__ import annotations

import tempfile

import pytest

from paf.capabilities.scheduled_tasks import (
    NotifyMode,
    RunRecord,
    ScheduledJob,
    ScheduledTaskStore,
)


def _job(**kw) -> ScheduledJob:
    """A valid job; name/description are required, so default them."""
    kw.setdefault("name", "test job")
    kw.setdefault("description", "a test job")
    return ScheduledJob(schedule="0 9 * * *", **kw)


# --- model: exactly-one-of + defaults + stamping ------------------------------
def test_job_requires_exactly_one_of_instructions_script() -> None:
    with pytest.raises(ValueError):
        _job()  # neither set
    with pytest.raises(ValueError):
        _job(instructions="x", script="y")  # both set
    job = _job(instructions="post the article")
    assert job.script is None
    job2 = _job(script="check.py")
    assert job2.instructions is None


def test_job_requires_name_and_description() -> None:
    with pytest.raises(ValueError):
        ScheduledJob(schedule="0 9 * * *", instructions="x")  # no name/description
    with pytest.raises(ValueError):
        ScheduledJob(schedule="0 9 * * *", name="n", instructions="x")  # no description
    with pytest.raises(ValueError):
        ScheduledJob(
            schedule="0 9 * * *", description="d", instructions="x"
        )  # no name


def test_job_defaults() -> None:
    job = _job(instructions="x")
    assert job.notify is NotifyMode.always
    assert job.delivery == "file"
    assert job.enabled is True
    assert job.archived is False
    assert job.id
    assert job.created_at is not None
    assert job.last_run is None


def test_job_id_and_created_at_stamped_in_code() -> None:
    a = _job(instructions="x")
    b = _job(instructions="x")
    assert a.id != b.id  # distinct ids, not copied
    assert a.created_at is not None


def test_notify_mode_values() -> None:
    assert NotifyMode.always.value == "always"
    assert NotifyMode.on_trigger.value == "on_trigger"


# --- master list: CRUD + soft-delete -----------------------------------------
def test_store_create_get_list() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ScheduledTaskStore(tmp, agent="a")
        job = _job(instructions="post")
        stored = store.create_job(job)
        assert stored.id == job.id
        assert store.get_job(job.id) == job
        assert [j.id for j in store.list_jobs()] == [job.id]


def test_store_create_duplicate_id_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ScheduledTaskStore(tmp, agent="a")
        job = _job(instructions="post")
        store.create_job(job)
        with pytest.raises(ValueError):
            store.create_job(job)


def test_store_update_job_partial() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ScheduledTaskStore(tmp, agent="a")
        job = _job(instructions="post")
        store.create_job(job)
        updated = store.update_job(job.id, enabled=False, delivery="push")
        assert updated is not None
        assert updated.enabled is False
        assert updated.delivery == "push"
        assert updated.instructions == "post"  # untouched field preserved
        # missing id -> None
        assert store.update_job("nope", enabled=False) is None


def test_store_update_job_name_description() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ScheduledTaskStore(tmp, agent="a")
        job = _job(instructions="post")
        store.create_job(job)
        updated = store.update_job(
            job.id, name="Renamed", description="now runs at 2:40pm"
        )
        assert updated is not None
        assert updated.name == "Renamed"
        assert updated.description == "now runs at 2:40pm"
        assert updated.schedule == "0 9 * * *"  # untouched field preserved
        # partial: only name changes, description untouched
        renamed = store.update_job(job.id, name="Only name")
        assert renamed is not None
        assert renamed.name == "Only name"
        assert renamed.description == "now runs at 2:40pm"


def test_store_remove_is_soft_delete() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ScheduledTaskStore(tmp, agent="a")
        job = _job(instructions="post")
        store.create_job(job)
        assert store.remove_job("nope") is False
        assert store.remove_job(job.id) is True
        # soft-delete: still present, archived, hidden from default list
        got = store.get_job(job.id)
        assert got is not None and got.archived is True
        assert store.list_jobs() == []
        assert [j.id for j in store.list_jobs(include_archived=True)] == [job.id]


def test_store_persists_across_instances() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        a = ScheduledTaskStore(tmp, agent="a")
        job = _job(instructions="post")
        a.create_job(job)
        # distinct instance, same root+agent -> sees the same job
        b = ScheduledTaskStore(tmp, agent="a")
        assert [j.id for j in b.list_jobs()] == [job.id]


def test_store_is_per_agent_and_job_carries_columns() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        a = ScheduledTaskStore(tmp, agent="a")
        b = ScheduledTaskStore(tmp, agent="b")
        job = _job(instructions="x", agent="a", conversation_id="c1")
        a.create_job(job)
        # per-agent: agent a sees it, agent b does not
        assert len(a.list_jobs()) == 1
        assert b.list_jobs() == []
        # the job carries its agent + conversation as columns
        got = a.get_job(job.id)
        assert got is not None
        assert got.agent == "a"
        assert got.conversation_id == "c1"


# --- run history: append + read ----------------------------------------------
def test_store_append_and_read_runs_in_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ScheduledTaskStore(tmp, agent="a")
        r1 = RunRecord(job_id="j1", agent="a", conversation_id="c1")
        r2 = RunRecord(
            job_id="j2", agent="a", conversation_id="c1", status="done", detail="ok"
        )
        store.append_run(r1)
        store.append_run(r2)
        runs = store.read_runs()
        assert [r.job_id for r in runs] == ["j1", "j2"]
        assert runs[1].status == "done"
        assert runs[1].detail == "ok"


def test_store_run_history_persists_across_instances() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        a = ScheduledTaskStore(tmp, agent="a")
        a.append_run(RunRecord(job_id="j1", agent="a", conversation_id="c1"))
        b = ScheduledTaskStore(tmp, agent="a")
        assert [r.job_id for r in b.read_runs()] == ["j1"]


# --- change callback ----------------------------------------------------------
def test_store_on_change_fires_on_mutation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        calls: list[str] = []
        store = ScheduledTaskStore(tmp, agent="a", on_change=lambda: calls.append("x"))
        job = _job(instructions="post")
        store.create_job(job)
        assert calls == ["x"]
        store.get_job(job.id)  # read-only, no callback
        assert calls == ["x"]
        store.update_job(job.id, enabled=False)
        assert calls == ["x", "x"]
        store.remove_job(job.id)
        assert calls == ["x", "x", "x"]
