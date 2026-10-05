"""Background work gets its own trace instead of nesting in an unrelated turn.

``asyncio.create_task`` copies the current OTel span into every task, so a
background worker spawned during (or after) a turn inherited that turn's span:
memory-owner / reviewer / summarizer runs and notify turns showed up nested deep
inside whichever conversation turn happened to trigger them. These tests pin the
fix: ``root_span(new_trace=True)`` roots a fresh trace, queues opt in per job via
``BackgroundQueueService.job_trace``, and the notify span is a new trace grouped
by ``langfuse.session.id``.

Spans are captured with the SDK's in-memory exporter; the process tracer lookup
is monkeypatched rather than touching the process-global TracerProvider.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from paf.background import BackgroundQueueService, QueueState
from paf.observability.instrumentation import root_span


@pytest.fixture()
def spans(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Route the process tracer (``trace.get_tracer``) to an in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(trace, "get_tracer", lambda *a, **k: tracer)
    return exporter


def _by_name(exporter: InMemorySpanExporter) -> dict[str, Any]:
    return {s.name: s for s in exporter.get_finished_spans()}


# -- root_span ---------------------------------------------------------------------


def test_root_span_nests_by_default_and_new_trace_detaches(spans) -> None:
    with root_span("outer"):
        with root_span("nested"):
            pass
        with root_span("detached", new_trace=True):
            with root_span("inside-detached"):
                pass
    got = _by_name(spans)
    outer = got["outer"].context.trace_id
    assert got["nested"].context.trace_id == outer
    assert got["nested"].parent.span_id == got["outer"].context.span_id
    assert got["detached"].context.trace_id != outer
    assert got["detached"].parent is None
    # Work under the detached span nests under IT (its own trace).
    assert got["inside-detached"].context.trace_id == got["detached"].context.trace_id


# -- queue worker ----------------------------------------------------------------------


@dataclass
class _Job:
    value: str


class _Queue(BackgroundQueueService[_Job]):
    name = "trace-test"

    def __init__(self, *, own_trace: bool) -> None:
        super().__init__()
        self.own_trace = own_trace

    def job_trace(self, session_id: str, job: _Job) -> tuple[str, dict] | None:
        if not self.own_trace:
            return None
        return f"paf.test.{job.value}", {"agent": session_id}

    async def process_job(
        self, session_id: str, job: _Job, state: QueueState[_Job]
    ) -> None:
        with root_span(f"work.{job.value}"):
            pass


def _run_queue(own_trace: bool) -> None:
    async def scenario() -> None:
        queue = _Queue(own_trace=own_trace)
        # The worker is spawned while a "turn" span is current — exactly the
        # situation that leaked turn context into background work.
        with root_span("turn"):
            queue.enqueue("agent-a", _Job("j1"))
        await queue.drain("agent-a")

    asyncio.run(scenario())


def test_queue_job_with_job_trace_gets_its_own_trace(spans) -> None:
    _run_queue(own_trace=True)
    got = _by_name(spans)
    job = got["paf.test.j1"]
    assert job.parent is None
    assert job.context.trace_id != got["turn"].context.trace_id
    assert dict(job.attributes)["agent"] == "agent-a"
    assert got["work.j1"].context.trace_id == job.context.trace_id


def test_queue_job_without_job_trace_keeps_inherited_parent(spans) -> None:
    _run_queue(own_trace=False)
    got = _by_name(spans)
    assert "paf.test.j1" not in got
    assert got["work.j1"].context.trace_id == got["turn"].context.trace_id


# -- the real queues opt in --------------------------------------------------------


def test_memory_and_summarizer_queues_name_their_own_traces() -> None:
    from paf.capabilities.memory._background import (
        ConsolidationJob,
        MemoryConsolidationQueue,
    )
    from paf.capabilities.thread_summarizer._background import (
        ThreadSummarizeJob,
        ThreadSummarizerQueue,
    )

    # job_trace does not touch instance state, so skip the heavy constructors.
    owner = MemoryConsolidationQueue.__new__(MemoryConsolidationQueue)
    assert owner.job_trace(
        "demo_assistant", ConsolidationJob(run_id="r1", turn_id=4)
    ) == (
        "paf.memory.consolidate.demo_assistant",
        {"agent": "demo_assistant", "run_id": "r1", "turn_id": 4},
    )
    summarizer = ThreadSummarizerQueue.__new__(ThreadSummarizerQueue)
    job = ThreadSummarizeJob(conversation_id="demo:paf-abc")
    assert summarizer.job_trace("demo:paf-abc", job) == (
        "paf.thread_summarize.demo:paf-abc",
        {"langfuse.session.id": "demo:paf-abc"},
    )


# -- notify ------------------------------------------------------------------------


def test_notify_span_is_a_new_trace_in_the_conversation_session(
    spans, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ui import sse

    seen: dict[str, Any] = {}

    class FakeSession:
        def notify(self, message: str) -> str:
            seen["message"] = message
            seen["span"] = trace.get_current_span()
            return "enqueue-1"  # steered into a live run -> no turn_started emit

    async def fake_get_session(
        conversation_id: str, agent: str, **_: object
    ) -> FakeSession:
        return FakeSession()

    monkeypatch.setattr(sse, "_get_session", fake_get_session)

    async def scenario() -> None:
        # The notify fires from a worker whose inherited span is the OLD turn.
        with root_span("old-turn"):
            await sse._notify_agent(
                "demo_assistant", "shell.demo:paf-abc.3", "demo:paf-abc", True, "exit 0"
            )

    asyncio.run(scenario())
    got = _by_name(spans)
    notify = got["paf.notify.shell.demo:paf-abc.3"]
    assert notify.parent is None
    assert notify.context.trace_id != got["old-turn"].context.trace_id
    assert dict(notify.attributes)["langfuse.session.id"] == "demo:paf-abc"
    # Session.notify (and so the notify turn's task) runs inside the new span.
    assert seen["span"].get_span_context().span_id == notify.context.span_id
