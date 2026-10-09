"""Tests for the scheduled-task run layer (Phase 3).

Covers ``ScheduledTaskQueue.process_job``: the agent-executed path (runs a turn
via the ``run_turn`` seam, appends a minimal RunRecord), the script-executed
path (runs the script, notify mode by exit code), and the ProcessEvent feed.
"""

from __future__ import annotations

import asyncio
import tempfile

from paf.background_processes._events import (
    ProcessEvent,
    ProcessEventType,
    clear_agent_emitters,
    get_agent_emitter,
)
from paf.capabilities._sandbox import ExecResult
from paf.capabilities.scheduled_tasks import (
    NotifyMode,
    ScheduledJob,
    ScheduledTaskStore,
)
from paf.capabilities.scheduled_tasks._queue import ScheduledTaskQueue


class FakeExecutor:
    """SandboxedExecutor stand-in returning a canned ExecResult."""

    def __init__(self, result: ExecResult) -> None:
        self.result = result
        self.calls: list[tuple[str, object, object, float]] = []

    async def run(self, command, *, cwd, env, timeout) -> ExecResult:
        self.calls.append((command, cwd, env, timeout))
        return self.result


class FakeRunTurn:
    """Records ``(agent, cid, content)`` turn requests instead of running them."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, agent: str, cid: str, content: str) -> None:
        self.calls.append((agent, cid, content))


def _store_factory(tmp: str):
    stores: dict[str, ScheduledTaskStore] = {}

    def factory(agent: str) -> ScheduledTaskStore:
        if agent not in stores:
            stores[agent] = ScheduledTaskStore(tmp, agent=agent)
        return stores[agent]

    return factory


def _agent_job(**kw) -> ScheduledJob:
    return ScheduledJob(
        schedule="0 9 * * *", instructions="post the article", agent="a", **kw
    )


def _script_job(**kw) -> ScheduledJob:
    return ScheduledJob(schedule="0 9 * * *", script="echo hi", agent="a", **kw)


# --- agent-executed path (q3a) ----------------------------------------------
def test_agent_executed_runs_turn_and_appends_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        factory = _store_factory(tmp)
        run_turn = FakeRunTurn()
        queue = ScheduledTaskQueue(
            store_factory=factory,
            run_turn=run_turn,
            executor=FakeExecutor(ExecResult("", "", 0)),
        )
        job = _agent_job()
        asyncio.run(queue.process_job("c1", job, None))

        assert run_turn.calls == [("a", "c1", "post the article")]
        runs = factory("a").read_runs()
        assert len(runs) == 2  # running + done
        rec = runs[-1]
        assert rec.job_id == job.id
        assert rec.agent == "a"
        assert rec.conversation_id == "c1"
        assert rec.status == "done"
        assert rec.timestamp is not None


# --- script-executed path (q3b) ---------------------------------------------
def test_script_executed_runs_script_and_notifies_always() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        factory = _store_factory(tmp)
        run_turn = FakeRunTurn()
        executor = FakeExecutor(ExecResult("ok", "", 0))
        queue = ScheduledTaskQueue(
            store_factory=factory,
            run_turn=run_turn,
            executor=executor,
        )
        job = _script_job(notify=NotifyMode.always)
        asyncio.run(queue.process_job("c1", job, None))

        assert executor.calls and executor.calls[0][0] == "echo hi"
        # notify=always -> the agent is in the loop every run
        assert len(run_turn.calls) == 1
        assert run_turn.calls[0][1] == "c1"
        assert "exit code 0" in run_turn.calls[0][2]
        assert len(factory("a").read_runs()) == 2


def test_script_executed_on_trigger_skips_notify_on_success() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        factory = _store_factory(tmp)
        run_turn = FakeRunTurn()
        executor = FakeExecutor(ExecResult("ok", "", 0))
        queue = ScheduledTaskQueue(
            store_factory=factory,
            run_turn=run_turn,
            executor=executor,
        )
        job = _script_job(notify=NotifyMode.on_trigger)
        asyncio.run(queue.process_job("c1", job, None))

        assert executor.calls
        # exit 0 -> no trigger -> agent not notified
        assert run_turn.calls == []


def test_script_executed_on_trigger_notifies_on_nonzero_exit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        factory = _store_factory(tmp)
        run_turn = FakeRunTurn()
        executor = FakeExecutor(ExecResult("", "boom", 1))
        queue = ScheduledTaskQueue(
            store_factory=factory,
            run_turn=run_turn,
            executor=executor,
        )
        job = _script_job(notify=NotifyMode.on_trigger)
        asyncio.run(queue.process_job("c1", job, None))

        assert executor.calls
        # non-zero exit -> trigger -> agent notified
        assert len(run_turn.calls) == 1
        assert "exit code 1" in run_turn.calls[0][2]


# --- ProcessEvent feed (q3c) ------------------------------------------------
def test_process_job_feeds_process_events() -> None:
    clear_agent_emitters()
    seen: list[ProcessEvent] = []
    get_agent_emitter("a").on_any(seen.append)
    with tempfile.TemporaryDirectory() as tmp:
        factory = _store_factory(tmp)
        queue = ScheduledTaskQueue(
            store_factory=factory,
            run_turn=FakeRunTurn(),
            executor=FakeExecutor(ExecResult("", "", 0)),
        )
        job = _agent_job()
        asyncio.run(queue.process_job("c1", job, None))

    statuses = [e.status for e in seen]
    assert ProcessEventType.running in statuses
    assert ProcessEventType.done in statuses
    # conversation-scoped by origin
    assert all(e.lane == "conversation" for e in seen)
    assert all(e.scope == "c1" for e in seen)


# --- single queue routes by the job's agent (cross-agent) --------------------
def test_queue_routes_by_job_agent() -> None:
    # A single queue shared across agents: a job owned by agent "b" must land
    # its run record in agent b's store and route its turn to agent b, not a.
    with tempfile.TemporaryDirectory() as tmp:
        factory = _store_factory(tmp)
        run_turn = FakeRunTurn()
        queue = ScheduledTaskQueue(
            store_factory=factory,
            run_turn=run_turn,
            executor=FakeExecutor(ExecResult("", "", 0)),
        )
        job = ScheduledJob(
            schedule="0 9 * * *", instructions="hi", agent="b", conversation_id="c9"
        )
        asyncio.run(queue.process_job("c9", job, None))

        assert len(factory("b").read_runs()) == 2  # running + done in b's store
        assert factory("a").read_runs() == []  # nothing leaked into a's store
        assert run_turn.calls == [("b", "c9", "hi")]
