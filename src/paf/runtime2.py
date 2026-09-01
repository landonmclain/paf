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
PHASE 1 (this file): ``stream()`` switched from ``run_stream_events()`` to the
``agent.iter()`` node walk, reusing the existing per-event dispatch verbatim.
The forward-declared kinds (ToolRunning/ToolError/Background/Maintenance) are
declared but NOT yet wired — phases 2/3 add them. ``cancel()`` is wired to the
Session-held CancellationToken; ``enqueue()``/``subscribe_tasks()`` still stub.
No ``runtime.py`` or ``ui/`` consumer is touched.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
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
from pydantic_ai.messages import TextPartDelta, ThinkingPart, ThinkingPartDelta

from paf.observability.instrumentation import _json_dumps, root_span
from paf.runtime import (
    Session,
    StreamEvent,
    _part_leading_text,
    _run_input,
    _run_output,
)

__all__ = [
    "Session2",
    "StreamEvent",
    "ToolRunning",
    "ToolError",
    "BackgroundEvent",
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


class Session2(Session):
    """Next-gen ``Session``: ``agent.iter()`` control loop + additive events.

    Subclasses :class:`paf.runtime.Session`, reusing its entire init path
    (store, agent build, memory queue, resume/restore/close) unchanged. Phase 1
    swaps the control loop from ``run_stream_events()`` to an ``agent.iter()``
    node walk; the additive event kinds land in Phase 2.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Session-held CancellationToken -> stop (Phase 3 refines; the token is
        # created here so cancel() works and iter() has a token from day one).
        self._token = CancellationToken()
        # Phase 2/3 wiring: self._run (AgentRun handle) for steer/enqueue, and
        # task registry for BackgroundEvent replay land after the loop proves out.

    # -- control loop (Phase 1) ---------------------------------------------

    async def stream(
        self,
        prompt: str,
        *,
        conversation_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
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
                                if isinstance(event, PartStartEvent):
                                    part = event.part
                                    delta = _part_leading_text(part)
                                    if delta:
                                        kind = (
                                            "thinking"
                                            if isinstance(part, ThinkingPart)
                                            else "text"
                                        )
                                        yield StreamEvent(kind=kind, text=delta)
                                elif isinstance(event, PartDeltaEvent):
                                    if isinstance(event.delta, TextPartDelta):
                                        yield StreamEvent(
                                            kind="text", text=event.delta.content_delta
                                        )
                                    elif isinstance(event.delta, ThinkingPartDelta):
                                        yield StreamEvent(
                                            kind="thinking",
                                            text=event.delta.content_delta,
                                        )
                                elif isinstance(event, FunctionToolCallEvent):
                                    yield StreamEvent(
                                        kind="tool_call",
                                        tool_name=str(event.part.tool_name or ""),
                                        args=(
                                            event.part.args
                                            if isinstance(event.part.args, str)
                                            else str(event.part.args)
                                        ),
                                    )
                                elif isinstance(event, FunctionToolResultEvent):
                                    yield StreamEvent(
                                        kind="tool_result",
                                        tool_name=str(event.part.tool_name or ""),
                                        text=(
                                            event.part.content
                                            if isinstance(event.part.content, str)
                                            else str(event.part.content or "")
                                        ),
                                    )
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

    def subscribe_tasks(self) -> Any:
        """Replay registry for BackgroundEvent emission. Phase 2."""
        raise NotImplementedError("Session2.subscribe_tasks(): Phase 2")
