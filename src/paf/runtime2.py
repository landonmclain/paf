"""Next-generation paf ``Session`` — ``agent.iter()`` control loop, additive events.

``runtime2`` is the promoted successor to :mod:`paf.runtime`. It is a **sibling
module** (not an in-place rewrite) so it can be developed and tested against the
live ``run_stream_events()`` path without touching it. Design, decisions, and
the verified pydantic-ai 2.33 API live in the vault spec:

    Hermes/Projects/Personal/Portfolio-GitHub/Designs/paf-runtime2-Session.md

Locked decisions (do not revisit without reopening the workstream):
- Sibling ``src/paf/runtime2.py``; promotes to THE single runtime, then
  ``runtime.py`` is deleted.
- GO BIG: switching bar = ``coding_agent`` event parity
  (ActionStep running|done|error, MaintenanceStep, TurnUpdate).
- Reuse shared core via imports (store/build_agent/resume/memory-queue).
- Same public ``Session`` contract; new event kinds are **additive**.
- ``Session2`` is a **subclass** of ``Session`` (drop-in for consumers that
  type against ``Session``), overriding only the control loop + new surfaces.

PHASE 0 (done): compile-clean stub. Additive event dataclasses forward-declared.
PHASE 1 (done, commit c91e147): ``stream()`` switched from ``run_stream_events()``
to ``agent.iter()`` node walk, reusing the existing per-event dispatch verbatim
(byte-identical 4-kind contract for text/thinking/tool_call/tool_result).
``cancel()`` wired to the Session-held CancellationToken.
PHASE 2 (this file): the forward-declared kinds are now WIRED. (1) The loop
emits ``ToolRunning`` at tool-call start and ``ToolError`` on a failed/retried
result, keyed by the part's ``tool_call_id`` (stable ``step_id``). (2) The
background-task registry + ``subscribe_tasks()`` replay lands consumer #1:
memory consolidation publishes queued/running/done via ``BackgroundEvent``.
``enqueue()`` remains a Phase 3 stub. No ``runtime.py`` or ``ui/`` consumer is
touched (events are additive).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic_ai import (
    AgentRunResult,
    CancellationToken,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
)
from pydantic_ai._agent_graph import CallToolsNode, End, ModelRequestNode
from pydantic_ai.messages import (
    RetryPromptPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
)

from paf.capabilities import PROFILES
from paf.observability.instrumentation import _json_dumps, root_span
from paf.runtime import (
    Session,
    StreamEvent,
    _part_leading_text,
    _run_input,
    _run_output,
)
from paf.specs import resolve_profile_name

log = logging.getLogger(__name__)

__all__ = [
    "Session2",
    "StreamEvent",
    "ToolRunning",
    "ToolError",
    "BackgroundEvent",
    "BackgroundTask",
    "MaintenanceEvent",
    "PROCESSING",
    "DONE",
    "ERROR",
    "QUEUED",
    "SKIPPED",
]


# ---------------------------------------------------------------------------
# Forward-declared additive event kinds (Phase 0: declared, not yet wired).
#
# Mapping to coding_agent parity (chat_runner.py):
#   ActionStep        -> ToolRunning + ToolError (status running|done|error)
#   MaintenanceStep   -> MaintenanceEvent (status done|queued|skipped|error)
#   TurnUpdate/result -> BackgroundEvent (paf owns transport; richer event)
# The existing ``text`` / ``thinking`` / ``tool_call`` / ``tool_result`` kinds
# are inherited unchanged from :class:`StreamEvent` in ``paf.runtime``.
# ---------------------------------------------------------------------------

# Step lifecycle statuses (coding_agent ActionStep.status / MaintenanceStep.status).
PROCESSING = "running"
DONE = "done"
ERROR = "error"
QUEUED = "queued"
SKIPPED = "skipped"

StepStatus = Literal["running", "done", "error", "queued", "skipped"]


@dataclass(frozen=True)
class ToolRunning:
    """Emitted when a tool call starts (coding_agent ActionStep ``running``).

    ``step_id`` correlates with the ``ToolError``/``ToolResult`` that finishes
    this step, letting the UI flip ``running -> done|error`` by id.
    """

    tool_name: str
    step_id: str


@dataclass(frozen=True)
class ToolError:
    """Emitted when a tool call fails (coding_agent ActionStep ``error``)."""

    step_id: str
    error: str


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
class MaintenanceEvent:
    """Post-turn maintenance reporting — memory schedule, turn-log, etc.

    coding_agent's ``MaintenanceStep`` (done|queued|skipped|error). Fired after
    a turn completes, never on the turn's hot path.
    """

    label: str
    status: StepStatus


@dataclass
class BackgroundTask:
    """Registry entry for background work (consumer #1 = memory consolidation).

    Mirrors coding_agent ``MaintenanceStep``: one row per background task with a
    stable ``id`` the UI can mutate ``queued -> running -> done|error|skipped``
    by. ``progress`` is 0..1 when known; ``detail`` is branch-specific free text
    (e.g. ``"5 turns -> 1 summary"``). Lives in the Session's task registry and
    is REPLAYED to late-joining ``subscribe_tasks`` subscribers, so a web tab
    that connects mid-fold sees ``running`` instead of nothing.
    """

    id: str
    label: str
    status: StepStatus = QUEUED
    progress: float | None = None
    detail: str = ""
    updated_at: str = ""

    def as_event(self) -> BackgroundEvent:
        """The additive event form subscribers receive on each transition."""
        return BackgroundEvent(
            kind=f"task:{self.label}",
            status=self.status,
            detail=self.detail,
            metadata={"id": self.id, "progress": self.progress},
        )


#: The per-tool lifecycle events that ride the turn ``stream()`` (in addition
#: to the inherited :class:`StreamEvent` kinds). Background/Maintenance are NOT
#: here — they live on their own ``subscribe_tasks`` timeline, never the turn.
TurnStreamEvent = StreamEvent | ToolRunning | ToolError


def _dispatch_stream_event(event: Any) -> list[TurnStreamEvent]:
    """Map ONE pydantic-ai stream node event to the turn ``StreamEvent``(s).

    The exact dispatch the old ``run_stream_events()`` loop used (PartStart
    leading text, PartDelta text/thinking, FunctionToolCall -> tool_call,
    FunctionToolResult -> tool_result) PLUS the Phase 2 additive kinds:
    ``ToolRunning`` is emitted right after each ``tool_call`` (tool-call START),
    and ``ToolError`` when a tool result came back failed/retried. Extracted to
    a pure function so the Phase 1+2 event contract is unit-testable offline
    with synthetic events — no model, no live tool invocation required.
    """
    out: list[TurnStreamEvent] = []
    if isinstance(event, PartStartEvent):
        part = event.part
        delta = _part_leading_text(part)
        if delta:
            kind = "thinking" if isinstance(part, ThinkingPart) else "text"
            out.append(StreamEvent(kind=kind, text=delta))
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta):
            out.append(StreamEvent(kind="text", text=event.delta.content_delta))
        elif isinstance(event.delta, ThinkingPartDelta):
            out.append(
                StreamEvent(kind="thinking", text=event.delta.content_delta)
            )
    elif isinstance(event, FunctionToolCallEvent):
        tool_name = str(event.part.tool_name or "")
        step_id = str(event.part.tool_call_id or "")
        out.append(
            StreamEvent(
                kind="tool_call",
                tool_name=tool_name,
                args=(
                    event.part.args
                    if isinstance(event.part.args, str)
                    else str(event.part.args)
                ),
            )
        )
        # Phase 2: tool-call START (coding_agent ActionStep running). step_id
        # = the stable tool_call_id, so the UI flips ONE row running->done|error.
        out.append(ToolRunning(tool_name=tool_name, step_id=step_id))
    elif isinstance(event, FunctionToolResultEvent):
        part = event.part
        tool_name = str(getattr(part, "tool_name", "") or "")
        step_id = str(getattr(part, "tool_call_id", "") or "")
        content = getattr(part, "content", None)
        out.append(
            StreamEvent(
                kind="tool_result",
                tool_name=tool_name,
                text=(
                    content
                    if isinstance(content, str)
                    else str(content or "")
                ),
            )
        )
        # Phase 2: tool ERROR. A real tool failure comes back as a
        # ToolReturnPart with outcome != success; a RetryPromptPart means the
        # model was told to retry because the tool errored. Both -> ToolError.
        is_error = isinstance(part, RetryPromptPart) or (
            isinstance(part, ToolReturnPart) and part.outcome != "success"
        )
        if is_error:
            out.append(ToolError(step_id=step_id, error=_tool_error_detail(part)))
    return out


def _tool_error_detail(part: ToolReturnPart | RetryPromptPart) -> str:
    """Human-readable reason for a failed tool result.

    ``ToolReturnPart`` carries ``outcome`` and ``content`` (the tool's error
    message); ``RetryPromptPart`` carries the model-retry prompt. Both get
    truncated so a runaway exception doesn't flood the UI row.
    """
    if isinstance(part, RetryPromptPart):
        c = part.content if isinstance(part.content, str) else str(part.content or "")
        return (c or "tool retried").strip()[:500]
    # ToolReturnPart
    detail = part.content
    if not isinstance(detail, str):
        detail = str(detail or "")
    out = getattr(part, "outcome", "error")
    return f"[{out}] {(detail or 'tool returned a failure state').strip()[:500]}"


class Session2(Session):
    """Next-gen ``Session``: ``agent.iter()`` control loop + additive events.

    Subclasses :class:`paf.runtime.Session`, reusing its entire init path
    (store, agent build, memory queue, resume/restore/close) unchanged. Phase 1
    swaps the control loop from ``run_stream_events()`` to an ``agent.iter()``
    node walk; the additive event kinds land in Phase 2.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Resolve the spec-declared profile when the caller didn't pass one
        # explicitly, matching how the CLIs pick the default profile
        # (`resolve_profile_name` -> PROFILES[name]). This is what makes
        # `Session2(agent_name="runtime2_test")` build the LEAN Shell-only
        # toolset instead of silently falling back to ASSISTANT_PROFILE.
        if kwargs.get("profile") is None:
            agent_name = kwargs.get("agent_name", "assistant")
            if args and isinstance(args[0], str):
                agent_name = args[0]
            profile_name = resolve_profile_name(agent_name)
            if profile_name and profile_name in PROFILES:
                kwargs["profile"] = PROFILES[profile_name]
        super().__init__(*args, **kwargs)
        # Session-held CancellationToken -> stop (Phase 3 refines; the token is
        # created here so cancel() works and iter() has a token from day one).
        self._token = CancellationToken()
        # Phase 3 wiring: self._run (AgentRun handle) for steer/enqueue.
        self._run: Any = None
        # Background-task registry (Phase 2): id -> BackgroundTask + the
        # subscriber callbacks. Subscribers are REPLAYED active tasks on
        # subscribe so a late-joining connection sees running, not nothing.
        self._background_tasks: dict[str, BackgroundTask] = {}
        self._task_subscribers: list[Callable[[BackgroundEvent], Awaitable[Any] | Any]] = (
            []
        )
        self._consolidation_seq = 0

    # -- control loop (Phase 1) ---------------------------------------------

    async def stream(
        self,
        prompt: str,
        *,
        conversation_id: str | None = None,
    ) -> AsyncIterator[TurnStreamEvent]:
        """Run one turn, yielding renderable :class:`StreamEvent` objects.

        PHASE 1: driven by ``agent.iter()`` — the docs-endorsed control+injection
        surface — instead of ``run_stream_events()``. Walks graph nodes, consumes
        each streamable node's ``AgentStreamEvent`` iterator, and reuses the
        exact per-event dispatch the old flat loop used (PartStart leading text,
        PartDelta text/thinking, FunctionToolCall -> tool_call,
        FunctionToolResult -> tool_result). This keeps the emitted stream
        byte-identical to the current runtime for the existing 4 kinds.

        Parities preserved from :meth:`paf.runtime.Session.stream`:
        - ``[date: ...]`` UTC stamp on every user prompt (born_on provenance).
        - ONE root span per turn with ``langfuse.observation.input`` (set up
          front) and ``langfuse.observation.output`` (set once the settled
          ``run.result`` is known, before the span closes).
        - On completion ``self.last_result`` = settled ``AgentRunResult`` and
          ``self.history`` += ``result.new_messages()``.
        - Background memory-consolidation trigger fired post-turn.

        The chosen ``conversation_id`` (explicit > session default) is threaded
        through so persistent storage restores prior history.
        """
        cid = conversation_id if conversation_id is not None else self.conversation_id
        await self._ensure_resumed(cid)

        # Stamp the current UTC timestamp onto the start of EVERY user message
        # (same ``[date: ...]`` convention the benchmark harness prepends), so
        # the model always has an explicit date+time anchor and the
        # consolidation owner can derive a real ``born_on`` instead of
        # inventing one. Second precision, ``+00:00`` suffix.
        now = datetime.now(UTC).replace(microsecond=0)
        prompt = f"[date: {now.isoformat()}] {prompt}"

        result: AgentRunResult | None = None
        # Trace hygiene: open ONE root span per turn (see runtime.py stream()).
        with root_span(
            f"paf.run.{cid or 'default'}",
            **( {"langfuse.observation.input": _run_input(prompt, cid)}),
        ) as _span:
            async with self._agent.iter(
                prompt,
                message_history=self.history,
                conversation_id=cid,
                cancellation_token=self._token,
            ) as run:
                async for node in run:
                    if isinstance(node, (ModelRequestNode, CallToolsNode)):
                        # Both streamable node types expose @asynccontextmanager
                        # stream(ctx) yielding an AgentStreamEvent iterator; the
                        # union includes PartStart/PartDelta/FunctionToolCall/
                        # FunctionToolResult — the same set as run_stream_events.
                        async with node.stream(run.ctx) as stream:
                            async for event in stream:
                                # Reuse the SAME per-event dispatch the old flat
                                # loop used (Phase 1) + the Phase 2 additive
                                # kinds. Extracted to a pure module function so
                                # it is unit-testable with synthetic events.
                                for ev in _dispatch_stream_event(event):
                                    yield ev
                    elif isinstance(node, End):
                        break
            result = run.result
            assert result is not None, "iter() ended before AgentRun.result settled"
            # Set the trace output NOW, while the root span is still open.
            _span.set_attribute(
                "langfuse.observation.output", _json_dumps(_run_output(result))
            )
        self.last_result = result
        # ACKNOWLEDGE: new_messages() is ONLY this turn; older history lives in
        # self.history. Accumulate, don't overwrite.
        self.history = [*self.history, *result.new_messages()]
        # Fire the background memory-consolidation trigger (if enabled).
        await self._maybe_trigger_consolidation(cid)

    # -- steering / control (Phase 3; cancel wired in Phase 1) --------------

    async def cancel(self) -> None:
        """Stop the in-flight turn (CancellationToken cancel).

        Returns once the token is cancelled; the running ``stream()`` loop
        observes it at its next natural boundary and ends with ``RunCancelled``
        (carrying completed history for resume). No-op if nothing is running.
        """
        self._token.cancel()

    async def enqueue(self, content: str, *, priority: str = "asap") -> str | None:
        """Steer the running agent at the next node boundary (AgentRun.enqueue).

        Phase 3: requires keeping a handle to the active ``AgentRun`` during the
        iter() walk. Not implemented yet — the Phase 1 loop proves the event
        parity first.
        """
        raise NotImplementedError("Session2.enqueue(): Phase 3")

    def subscribe_tasks(
        self, cb: Callable[[BackgroundEvent], Awaitable[Any] | Any]
    ) -> Callable[[BackgroundEvent], Awaitable[Any] | Any]:
        """Register a consumer for background-task lifecycle.

        The callback is invoked on every ``BackgroundEvent`` (queued/running/
        done/error/skipped transition) for every registered task. On subscribe,
        ALREADY-ACTIVE tasks (status queued|running) are REPLAYED synchronously
        first, so a late-joining connection (e.g. a web tab that opens mid-fold)
        sees ``running`` instead of missing the transition entirely.

        Returns the callback unchanged (convenient: ``cb = s.subscribe_tasks(cb)``).
        """
        if cb not in self._task_subscribers:
            self._task_subscribers.append(cb)
        # Replay active tasks so the subscriber doesn't miss what's in flight.
        for task in self._background_tasks.values():
            if task.status in (QUEUED, PROCESSING):
                cb(task.as_event())
        return cb

    def _publish_task(self, task: BackgroundTask) -> None:
        """Store a task in the registry and notify all subscribers."""
        task.updated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        self._background_tasks[task.id] = task
        ev = task.as_event()
        for cb in self._task_subscribers:
            try:
                result = cb(ev)
                if inspect.isawaitable(result):
                    asyncio.ensure_future(result)  # fire-and-forget, never block
            except Exception:  # a subscriber must never break the session
                log.exception("background-task subscriber failed task=%s", task.id)

    async def _maybe_trigger_consolidation(self, cid: str | None) -> None:
        """Fire the background-consolidation hook, publishing task lifecycle.

        Consumer #1 of the ``BackgroundEvent`` surface (mirrors coding_agent
        ``MaintenanceStep``). Overrides :meth:`paf.runtime.Session._maybe_trigger_consolidation`
        to first publish a ``queued`` task row, then ``await`` the base trigger.
        The base uses the shared ``MemoryConsolidationQueue`` unchanged; the
        wrap in :meth:`_ensure_memory_queue` flips the row to ``done``/``error``
        once the off-turn worker actually folds (or fails) — the completion is
        published on the background timeline, never inside ``stream()``.
        """
        if not self.memory_consolidation or cid is None:
            return
        self._consolidation_seq += 1
        task_id = f"consolidation.{cid}.{self._consolidation_seq}"
        task = BackgroundTask(id=task_id, label=f"Consolidating memory ({cid})")
        self._publish_task(task)  # queued
        try:
            await super()._maybe_trigger_consolidation(cid)
            # Base builds the queue (and enqueues) ONLY when the watermark
            # says enough turns have accumulated. If the queue is still None,
            # no job is due this turn — mark skipped, don't leave a phantom
            # "running" row in the registry.
            if self._memory_queue is None:
                task.status = SKIPPED
                task.detail = "no turns past the consolidation watermark"
            elif self._background_tasks.get(task_id) is not None:
                task.status = PROCESSING  # job handed to the worker
            self._publish_task(task)
        except Exception as exc:  # trigger path itself failed (rare)
            task.status = ERROR
            task.detail = str(exc)
            self._publish_task(task)

    def _ensure_memory_queue(self):
        """Build the shared queue with a completion-publishing fold wrap.

        Reuses the base queue build (/also/ the wrapper) but wraps
        ``self._consolidate_span`` so the off-turn worker's fold updates the
        consolidation ``BackgroundTask`` to ``done`` (or ``error`` on throw)
        and publishes it — the honest completion edge for consumer #1.
        """
        # Mark the currently-queued/running consolidation task done on success.
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

        return self._make_queue_with_wrapped_fold(
            _complete,
            self._consolidate_span,
        )

    def _make_queue_with_wrapped_fold(
        self, complete: Callable[[StepStatus, str], None], inner_span
    ):
        """Construct the memory queue, wrapping the fold seam with a publisher.

        Kept separate so the wrap logic is unit-testable without a model call.
        """
        from pydantic_ai_harness.conversation_search import SnapshotHistorySource

        from paf.capabilities.memory._background import (
            MemoryConsolidationQueue,
            resolve_unseen_span,
        )
        from paf.capabilities.memory._watermark import (
            MemoryWatermark,
        )

        source = SnapshotHistorySource(self._store)

        async def span_source(run_id: str, up_to_turn_id: int):
            wm = MemoryWatermark.load(self._memory_directory, run_id=run_id)
            return await resolve_unseen_span(
                source, wm, tip_run=run_id, tip_turn=up_to_turn_id
            )

        real_fold = inner_span

        async def publish_fold(messages, directory, watermark):
            try:
                n = (
                    await real_fold(messages, directory, watermark)
                    if real_fold is not None
                    else 0
                )
                complete(DONE, f"{n} records folded")
                return n
            except Exception as exc:
                complete(ERROR, str(exc))
                raise

        self._memory_queue = MemoryConsolidationQueue(
            memory_directory=self._memory_directory,
            agent_name=self.agent_name,
            span_source=span_source,
            consolidate_span=publish_fold,
        )
        return self._memory_queue
