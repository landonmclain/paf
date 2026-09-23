"""The paf ``Session`` runtime — ``agent.iter()`` control loop + additive events.

:class:`Session` is a single self-contained class: profile-scoped conversation
identity (``{profile}:{cid}`` at the store boundary), resume/restore, the
streaming loop, and per-turn steering (``cancel``/``enqueue``). Background
machinery (memory consolidation/review, thread summarization) lives in
:class:`paf.background.BackgroundCoordinator`, attached by the app/CLI via
:meth:`attach_background` — a bare Session (``background is None``) is turn-only.

The public contract: ``stream()``/``run()``/``run_sync()``/``restore``/
``refresh``/``close()``, plus ``history`` and ``last_result``. The session holds
the plain conversation id; the profile scope is applied at the store boundary.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from opentelemetry import trace as _otel_trace
from pydantic_ai import (
    Agent,
    AgentRun,
    AgentRunResult,
    CancellationToken,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
)
from pydantic_ai.agent import CallToolsNode, ModelRequestNode
from pydantic_ai.exceptions import (
    ModelHTTPError,
    RunCancelled,
    UnexpectedModelBehavior,
)
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_harness.compaction import compact_now
from pydantic_ai_harness.step_persistence import continue_run
from pydantic_graph import End

from paf.background import (
    DONE,
    ERROR,
    PROCESSING,
    QUEUED,
    SKIPPED,
    BackgroundCoordinator,
    BackgroundEvent,
    BackgroundLanes,
    BackgroundTask,
)
from paf.capabilities import PROFILES
from paf.capabilities.compaction import build_boundary_compaction_for_agent
from paf.capabilities.persistence import _build_store
from paf.degeneration_guard import (
    DegenerationError,
    DegenerationGuard,
    ToolCallGuard,
)
from paf.observability.instrumentation import (
    _json_dumps,
    root_span,
    setup_observability,
    shutdown_observability,
)
from paf.specs import build_agent, resolve_profile_name, spec_usage_limits

log = logging.getLogger(__name__)


__all__ = [
    "Session",
    "StreamEvent",
    "ToolRunning",
    "ToolError",
    "BackgroundEvent",
    "BackgroundTask",
    "MaintenanceEvent",
    "TurnHandle",
    "PROCESSING",
    "DONE",
    "ERROR",
    "QUEUED",
    "SKIPPED",
]


# ---------------------------------------------------------------------------
# StreamEvent — the 4-kind renderable event
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StreamEvent:
    """One renderable event from an agent run.

    UI-agnostic: a caller (REPL, web socket, TTS pipeline) maps these onto
    whatever it shows. ``kind`` is one of:

    - ``"text"``      a token chunk of the model's answer (``text``)
    - ``"thinking"``  a reasoning token (suppress or render faintly)
    - ``"tool_call"`` the model invoked a tool (``tool_name``, ``args``)
    - ``"tool_result"`` a tool returned (``tool_name``, ``text``)

    The settled ``AgentRunResult`` lands on ``self.last_result`` when the run
    completes and ``self.history`` accumulates the turn's messages — callers
    never need to reconstruct the transcript from these events.
    """

    kind: Literal["text", "thinking", "tool_call", "tool_result"]
    text: str = ""
    tool_name: str = ""
    args: str = ""


# ---------------------------------------------------------------------------
# Additive event kinds that ride the turn stream, alongside StreamEvent.
# ---------------------------------------------------------------------------

# Step lifecycle status for the turn-side ``MaintenanceEvent``.
StepStatus = Literal["running", "done", "error", "queued", "skipped"]


@dataclass(frozen=True)
class ToolRunning:
    """Emitted when a tool call starts.

    ``step_id`` correlates with the ``ToolError``/``ToolResult`` that finishes
    this step, letting the UI flip ``running -> done|error`` by id.
    """

    tool_name: str
    step_id: str


@dataclass(frozen=True)
class ToolError:
    """Emitted when a tool call fails."""

    step_id: str
    error: str


@dataclass(frozen=True)
class MaintenanceEvent:
    """Post-turn maintenance reporting — memory schedule, turn-log, etc.

    Fired after a turn completes, never on the turn's hot path.
    """

    label: str
    status: StepStatus


#: The per-tool lifecycle events that ride the turn ``stream()``, in addition
#: to the :class:`StreamEvent` kinds. Background/Maintenance events live on
#: their own ``subscribe_tasks`` timeline, never the turn.
TurnStreamEvent = StreamEvent | ToolRunning | ToolError


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _part_leading_text(part: Any) -> str:
    """Pull the initial text out of a stream part's ``.content`` field.

    The first chunk of a part sits in ``.content`` (a ``str`` or list of
    ``str``); later chunks arrive as ``TextPartDelta.content_delta``. Non-text
    parts return ``""``.
    """
    content = getattr(part, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(chunk) for chunk in content)
    return ""


def _run_input(prompt: str, cid: str | None) -> dict:
    """Serialize a turn's input for the Langfuse trace root span.

    ``langfuse.observation.input`` is what Langfuse's OTLP mapper shows in the
    trace-list Input column. That column is the *user's* input for that turn —
    the current prompt — NOT the accumulated history. pydantic-ai's own
    `gen_ai.input.messages` on the agent/generation spans already carries full
    per-message context (and reads coherently when you click into the span), so
    dumping the whole transcript here would just bloat the list view and
    duplicate what's a click away. cid is included for at-a-glance grouping.
    """
    return {"prompt": prompt, "conversation_id": cid}


def _run_output(result: Any) -> dict:
    """Serialize a turn's settled output for the Langfuse trace root span."""
    return {"output": result.output if result is not None else None}


def _is_turn_start(msg: Any) -> bool:
    """True if ``msg`` is a user request that opens a fresh turn.

    A `ModelRequest` opens a new turn only when it carries a real string
    `UserPromptPart` (the user's own prompt) — versus an injected/context
    part whose content is a non-str list. This intentionally mirrors the web
    renderer's turn-boundary test so `resume_window` rounding and the UI see
    the same turns.
    """
    if not isinstance(msg, ModelRequest):
        return False
    return any(
        isinstance(p, UserPromptPart) and isinstance(p.content, str)
        for p in (getattr(msg, "parts", None) or [])
    )


def _dispatch_stream_event(event: Any) -> list[TurnStreamEvent]:
    """Map ONE pydantic-ai stream node event to the turn ``StreamEvent``(s).

    PartStart leading text, PartDelta text/thinking, FunctionToolCall ->
    tool_call, FunctionToolResult -> tool_result. ``ToolRunning`` is emitted
    right after each ``tool_call`` (tool-call START), and ``ToolError`` when a
    tool result came back failed/retried. A pure function so the event contract
    is unit-testable offline with synthetic events — no model, no live tool
    invocation required.
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
                StreamEvent(kind="thinking", text=event.delta.content_delta or "")
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
        # Tool-call START. step_id = the stable tool_call_id, so the UI flips
        # ONE row running->done|error.
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
                text=(content if isinstance(content, str) else str(content or "")),
            )
        )
        # Tool ERROR. A real tool failure comes back as a ToolReturnPart with
        # outcome != success; a RetryPromptPart means the model was told to
        # retry because the tool errored. Both -> ToolError.
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


def _profile_name(profile: Any) -> str | None:
    """The namespace segment for a profile (``AgentProfile.name``, a bare name
    str, or ``None``) — used to scope conversation identity at the store.

    ``Session`` resolves spec-declared profiles to ``AgentProfile`` objects, so
    ``.name`` is the usual path; the fallbacks keep it robust for raw args.
    """
    name = getattr(profile, "name", None)
    if name:
        return name
    if isinstance(profile, str) and profile:
        return profile
    return None


class TurnHandle:
    """A Session-owned, running (or finished) turn with re-attachable subscribers.

    Owns the background asyncio task that drives ``Session.stream()`` and the
    per-subscriber event fan-out. The turn runs to completion regardless of
    whether any subscriber is attached; a subscriber attaches with
    :meth:`subscribe` and receives events from its attach point onward
    (fold-state up to now comes from history replay, not this stream).

    This is the durable-turns primitive: turn lifetime is decoupled from any
    single SSE request. A second ``start_turn`` on the same Session re-attaches
    to the running handle instead of starting a second turn.
    """

    def __init__(
        self,
        session: Session,
        prompt: str,
        conversation_id: str | None,
        token: CancellationToken,
    ) -> None:
        self._session = session
        self._token = token
        self._subscribers: set[asyncio.Queue[TurnStreamEvent | None]] = set()
        # Set by the pump on an unexpected end (not a clean finish/cancel).
        self.error: BaseException | None = None
        self._task = asyncio.create_task(self._pump(prompt, conversation_id))

    @property
    def done(self) -> bool:
        """True once the turn task has settled (any outcome)."""
        return self._task.done()

    @property
    def has_subscribers(self) -> bool:
        """True while at least one SSE subscriber is attached (a live stream).

        The idle sweeper uses this to distinguish a turn that is actively
        streaming to a connected client (not reapable) from a detached turn
        with no subscriber (an orphan, reapable as the cost guard).
        """
        return bool(self._subscribers)

    def cancel(self) -> None:
        """Stop the underlying turn task (hard stop; token cancel is gentler)."""
        if not self._task.done():
            self._task.cancel()

    async def _pump(self, prompt: str, conversation_id: str | None) -> None:
        try:
            async for ev in self._session.stream(
                prompt, conversation_id=conversation_id, _token=self._token
            ):
                self._publish(ev)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the loop
            self.error = exc
            log.warning("turn pump ended with error: %r", exc)
        finally:
            self._publish(None)  # end sentinel to every subscriber
            self._session._clear_turn(self)

    def _publish(self, ev: TurnStreamEvent | None) -> None:
        for q in list(self._subscribers):
            q.put_nowait(ev)

    async def subscribe(self) -> AsyncIterator[TurnStreamEvent]:
        """Yield this turn's events from now until it ends (empty if done)."""
        q: asyncio.Queue[TurnStreamEvent | None] = asyncio.Queue()
        self._subscribers.add(q)
        # A subscriber arriving after the turn settled gets the end sentinel
        # immediately (its content comes from history replay, not this stream).
        if self.done:
            q.put_nowait(None)
        try:
            while True:
                ev = await q.get()
                if ev is None:
                    return
                yield ev
        finally:
            self._subscribers.discard(q)


# ---------------------------------------------------------------------------
# The single self-contained Session
# ---------------------------------------------------------------------------
class Session:
    """The turn runtime—``agent.iter()`` control loop + additive events.

    The TURN CORE only: profile-scoped conversation identity (``{profile}:{cid}``
    at the store boundary), resume/restore, the streaming loop, and per-turn
    steering (``cancel``/``enqueue``). Background machinery (memory
    consolidation/review, thread summarization) lives in
    :class:`paf.background.BackgroundCoordinator`, attached by the app/CLI via
    :meth:`attach_background` — a bare Session (``background is None``) is
    turn-only. The machine holds the plain conversation id; the profile scope is
    applied at the store boundary, so callers never think about it.
    """

    # Class-level default so a Session always exposes ``background`` (bare by
    # default); the app/CLI attaches a coordinator when the profile calls for it.
    background: BackgroundLanes | None = None

    def __init__(
        self,
        agent_name: str = "assistant",
        *,
        profile=None,
        conversation_id: str | None = None,
        trace: bool = True,
        resume: bool = True,
        resume_window: int | None = None,
        store=None,
        on_complete=None,
    ) -> None:
        # Resolve the spec-declared profile when the caller didn't pass one
        # explicitly, matching how clients pick the default profile
        # (`resolve_profile_name` -> PROFILES[name]).
        if profile is None:
            profile_name = resolve_profile_name(agent_name)
            if profile_name and profile_name in PROFILES:
                profile = PROFILES[profile_name]
        self.agent_name = agent_name
        self.profile = profile
        self.conversation_id = conversation_id
        self.trace = trace
        self.resume = resume
        self.resume_window = resume_window
        self._resumed = False
        self._restored = False
        # Default to the canonical shared flat-file store (same directory
        # build_persistence()/cli use). Injectable for tests / custom roots.
        self._store = store if store is not None else _build_store()
        if trace:
            setup_observability()
        self._agent: Agent[Any, Any] = build_agent(
            agent_name, profile=profile, on_complete=on_complete
        )
        # Tier A turn-boundary compaction strategy (routine trim). Built from the
        # spec's `compact:` block; None when `compact.turn_boundary` is explicitly
        # false (control arm). Driven via compact_now at the top of stream(). Must
        # NOT be added to agent.capabilities or its before_model_request hook would
        # fire mid-turn and recreate cache busts.
        self._boundary_compaction = build_boundary_compaction_for_agent(agent_name)
        # Per-run usage limits from the spec's `usage_limits:` block (None when
        # absent -> pydantic-ai's default UsageLimits()). Passed to iter() below.
        self._usage_limits = spec_usage_limits(agent_name)
        self.history: list = []
        # Background orchestrator — BARE by default. The app/CLI attaches a
        # :class:`paf.background.BackgroundLanes` facade (via
        # ``BackgroundCoordinator.for_session(self)``) when the spec/profile
        # calls for memory/thread machinery; a memory-less profile (e.g.
        # ``contextbench``) stays ``None`` so it carries no scaffolding.
        self.background: BackgroundLanes | None = None
        # Session-held CancellationToken -> stop (created here so cancel()
        # works and iter() has a token from the start).
        self._token = CancellationToken()
        # Live AgentRun handle for steer/enqueue.
        self._run: Any = None
        # Session-owned running turn handle (durable turns). None when idle.
        self._turn: TurnHandle | None = None

    # -- introspection ---------------------------------------------------------

    @property
    def agent(self) -> Agent[Any, Any]:
        """The built agent."""
        return self._agent

    @property
    def name(self) -> str:
        return self._agent.name or self.agent_name

    @property
    def restored_count(self) -> int:
        """Number of prior messages loaded by ``continue_run`` resume (0 if none)."""
        return len(self.history) if self._restored else 0

    # -- profile namespace (conversation identity scoping) --------------------

    def _scoped_cid(self, cid: str | None) -> str | None:
        """Scope a conversation id by the session's profile: ``{profile}:{cid}``.

        The store/resume/agent-run access run under the profile-scoped id, so
        two profiles using the SAME plain ``conversation_id`` never cross-talk.
        Applied once at the boundary (``stream()`` selects the plain cid, then
        passes the scoped one downstream), so the whole turn (and its persisted
        runs) live under one namespace identity.
        """
        pname = _profile_name(self.profile)
        if not pname or cid is None:
            return cid
        return f"{pname}:{cid}"

    def attach_background(self) -> BackgroundLanes | None:
        """Attach the two-lane background coordinator for this session (or keep it bare).

        The app/CLI calls this once after constructing the Session: it builds a
        :class:`paf.background.BackgroundLanes` facade (shared agent lane +
        per-session conversation lane) scoped to this session's store/profile
        and assigns it to ``self.background``. A memory-less profile (e.g.
        ``contextbench``) yields ``None`` — the session stays turn-only.
        Returns the attached facade (or None).
        """
        self.background = BackgroundCoordinator.for_session(self)
        return self.background

    # -- resume ---------------------------------------------------------------

    async def restore(self, *, conversation_id: str | None = None) -> int:
        """Load prior history from the store now, returning ``restored_count``.

        Explicit form of the auto-resume — call it before the first run (e.g. to
        show "resumed N prior messages" in a banner). Same semantics as
        ``_ensure_resumed``: ``continue_run`` full snapshot (compaction summary
        included), idempotent, safe to call early or never.
        """
        cid = conversation_id if conversation_id is not None else self.conversation_id
        await self._ensure_resumed(self._scoped_cid(cid))
        return self.restored_count

    async def refresh(self, *, conversation_id: str | None = None) -> int:
        """Re-read prior history from the store, discarding the cached copy.

        Unlike :meth:`restore` — which is one-shot per session (guarded by the
        ``_resumed`` flag) — this forces a fresh ``continue_run`` so a *page
        reload* shows turns that were persisted after the session first resumed.
        Returns the new ``restored_count`` (0 if none).
        """
        cid = conversation_id if conversation_id is not None else self.conversation_id
        self._resumed = False
        self._restored = False
        self.history = []
        await self._ensure_resumed(self._scoped_cid(cid))
        return self.restored_count

    async def _recover_interrupted(self, run_id: str, history: list[Any]) -> list[Any]:
        """Trim a recovered interrupted snapshot to the last RESOLVED boundary.

        ``continue_run(..., include_interrupted=True)`` returns the full
        interrupted run's history, which may end on a partially-executed tool
        sequence. We must NOT feed unresolved (pending / side-effecting) tool
        calls back into context — pydantic-ai would re-execute or synthesize
        their returns. So we drop the trailing tail of messages that reference
        tool calls the store still lists as unresolved, keeping every completed
        tool return and the user prompt. A run with no unresolved tool effects
        is returned intact.
        """
        try:
            unresolved = {
                rec.tool_call_id
                for rec in await self._store.list_unresolved_tool_effects(run_id=run_id)
            }
        except Exception:  # noqa: BLE001 - store read is best-effort
            return history
        if not unresolved:
            return history
        # Walk back from the end, dropping the tail that references unresolved
        # tool calls (a ModelResponse carrying an unresolved ToolCallPart, or
        # the ModelRequest that echoes it / the partial empty response after).
        keep_end = len(history)
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            tainted = False
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart) and (
                        part.tool_call_id in unresolved
                    ):
                        tainted = True
                        break
            if tainted:
                keep_end = i
                continue
            break
        return history[:keep_end]

    async def _ensure_resumed(self, conversation_id: str | None) -> None:
        """Seed ``self.history`` from the store exactly once (before the first run).

        ``continue_run`` loads the latest settled snapshot for the conversation's
        most recent run and returns it as a message history — M1 semantics, full
        snapshot (compaction summary included). When ``resume_window`` is set, the
        history is trimmed to the last N messages so a fresh process does not load
        the entire conversation into context (older turns stay recoverable via the
        journal / ``search_conversation_history``). Idempotent: only acts when
        resume is enabled, the conversation has a stable id, and this is the first
        run. A conversation with no prior runs stays empty (``restored_count == 0``).

        ``conversation_id`` here is ALREADY profile-scoped by the caller
        (``_scoped_cid``).
        """
        if self._resumed or not self.resume or conversation_id is None:
            self._resumed = True
            return
        self._resumed = True
        try:
            runs = await self._store.list_runs(conversation_id=conversation_id)
            if not runs:
                return
            history = []
            for run in reversed(runs):
                # 1) Normal path: settled (complete) snapshot.
                try:
                    history = await continue_run(self._store, run_id=run.run_id)
                    break
                except Exception:  # noqa: BLE001
                    # 2) Broken turn: recover the interrupted snapshot, trimmed
                    #    to the last RESOLVED tool boundary so we never feed
                    #    pending/replayable side effects back into context.
                    try:
                        raw = await continue_run(
                            self._store,
                            run_id=run.run_id,
                            include_interrupted=True,
                        )
                    except Exception:  # noqa: BLE001 - genuinely unrecoverable
                        continue
                    history = await self._recover_interrupted(run.run_id, raw)
                    break
            if not history:
                return
            if self.resume_window is not None and self.resume_window >= 0:
                if not self.resume_window:
                    history = []
                else:
                    start = max(0, len(history) - self.resume_window)
                    if start > 0 and not _is_turn_start(history[start]):
                        for i in range(start - 1, -1, -1):
                            if _is_turn_start(history[i]):
                                start = i
                                break
                    history = history[start:]
            self.history = history
            self._restored = True
        except Exception:  # noqa: BLE001 - resume is sugar; never break the run
            self._restored = False

    # -- async core -----------------------------------------------------------

    async def stream(
        self,
        prompt: str,
        *,
        conversation_id: str | None = None,
        _token: CancellationToken | None = None,
    ) -> AsyncIterator[TurnStreamEvent]:
        """Run one turn, yielding renderable :class:`StreamEvent` objects.

        Driven by ``agent.iter()``: walks graph nodes, consumes each streamable
        node's ``AgentStreamEvent`` iterator, and dispatches each event
        (PartStart leading text, PartDelta text/thinking, FunctionToolCall ->
        tool_call, FunctionToolResult -> tool_result).

        Behavior:
        - ``[date: ...]`` UTC stamp on every user prompt (born_on provenance).
        - ONE root span per turn with ``langfuse.observation.input`` (set up
          front) and ``langfuse.observation.output`` (set once the settled
          ``run.result`` is known, before the span closes).
        - On completion ``self.last_result`` = settled ``AgentRunResult`` and
          ``self.history`` += ``result.new_messages()``.
        - Background memory-consolidation trigger fired post-turn.

        The chosen ``conversation_id`` (explicit > session default) is threaded
        through (profile-scoped via ``_scoped_cid``) so persistent storage
        restores prior history and the whole turn lives under one namespace
        identity.
        """
        cid = conversation_id if conversation_id is not None else self.conversation_id
        scoped = self._scoped_cid(cid)
        # A fresh turn must NOT inherit a previously-cancelled token: cancel()
        # is one-way on CancellationToken, so a token cancelled by an earlier
        # stop would instantly cancel every later run ("The agent run was
        # cancelled."). Recreate it per turn so stop only ever affects the
        # current turn.
        self._token = _token if _token is not None else CancellationToken()
        await self._ensure_resumed(scoped)

        # Stamp the current UTC timestamp onto the start of EVERY user message,
        # so the model always has an explicit date+time anchor and the
        # consolidation owner can derive a real ``born_on`` instead of
        # inventing one. Second precision, ``+00:00`` suffix.
        now = datetime.now(UTC).replace(microsecond=0)
        prompt = f"[date: {now.isoformat()}] {prompt}"

        result: AgentRunResult | None = None
        # Local ref to the live AgentRun for partial-state recovery on a
        # mid-stream model failure (see `except (ModelHTTPError, ...)` below).
        # Declared before the run so it's bound even if iter() itself raises.
        _active_run: AgentRun[Any, Any] | None = None
        with root_span(
            f"paf.run.{scoped or 'default'}",
            **({"langfuse.observation.input": _run_input(prompt, scoped)}),
        ) as _span:
            # Tier A — turn-boundary compaction (routine trim). Run the window
            # trim over the settled prior-turn history BEFORE request #1 inside
            # the turn's root span, so the first request of the turn is small and
            # the prompt prefix stays stable for the whole turn (no mid-turn
            # cache bust), AND so the `compact_messages` span parents under the
            # turn's trace (Langfuse) instead of a no-op tracer. `compact_now`
            # drives the boundary strategy outside an active run (the
            # harness-sanctioned seam for between-turn compaction).
            # Turn-bound-bounded: `self.history` here is only PRIOR turns (the
            # current prompt is added inside agent.iter), so the window never
            # touches the active turn. The tracer must be an OTel tracer on the
            # process-wide provider — the root_span is current here, so the
            # span auto-parents under the turn; without it `compact_now`
            # defaults to NoOpTracer and the span is invisible in Langfuse.
            # No-op when disabled (None) or when there's no history to trim.
            boundary_compaction = getattr(self, "_boundary_compaction", None)
            if boundary_compaction is not None and self.history:
                boundary_model = getattr(self._agent, "model", None)
                if boundary_model is not None:
                    compacted = await compact_now(
                        boundary_compaction,
                        list(self.history),
                        model=boundary_model,
                        deps=None,
                        tracer=_otel_trace.get_tracer("paf"),
                        tokenizer=getattr(boundary_compaction, "tokenizer", None),
                    )
                    if compacted != self.history:
                        self.history = compacted

            try:
                async with self._agent.iter(
                    prompt,
                    message_history=self.history,
                    conversation_id=scoped,
                    cancellation_token=self._token,
                    usage_limits=getattr(self, "_usage_limits", None),
                ) as run:
                    self._run = run
                    _active_run = run
                    # Per-turn degeneration guards (loop detectors). New for each
                    # turn; see only THIS turn's stream.
                    #  - `guard` (DegenerationGuard): thinking-narration repeats.
                    #  - `tool_guard` (ToolCallGuard): the same tool re-issued
                    #    with the same UI-identically-rendered args (a stuck
                    #    tool-call loop). Both raise DegenerationError on
                    #    detection -> the except (ModelHTTPError,
                    #    UnexpectedModelBehavior) handler below preserves the
                    #    partial and surfaces a clean retry.
                    guard = DegenerationGuard()
                    tool_guard = ToolCallGuard()
                    try:
                        async for node in run:
                            if isinstance(node, (ModelRequestNode, CallToolsNode)):
                                # Tool-call loop guard: feed each COMPLETED tool
                                # call (full args from the node's model_response)
                                # and abort if the same call recurs too many times.
                                if isinstance(node, CallToolsNode):
                                    for part in node.model_response.parts:
                                        if isinstance(part, ToolCallPart):
                                            tg = tool_guard.add_tool_call(
                                                part.tool_name, part.args
                                            )
                                            if tg is not None:
                                                raise DegenerationError(tg)
                                async with node.stream(run.ctx) as stream:
                                    async for event in stream:
                                        # Guard hook: feed THINKING-only stream to
                                        # the degeneration detector and abort if it
                                        # trips (a thinking-narration repeat-loop).
                                        if isinstance(
                                            event, PartDeltaEvent
                                        ) and isinstance(
                                            event.delta, ThinkingPartDelta
                                        ):
                                            guard_result = guard.stream_thinking(
                                                event.delta.content_delta or ""
                                            )
                                            if guard_result is not None:
                                                raise DegenerationError(guard_result)
                                        for ev in _dispatch_stream_event(event):
                                            yield ev
                            elif isinstance(node, End):
                                break
                    finally:
                        self._run = None
                result = run.result
            except RunCancelled as exc:
                # Stop (application-requested cancellation via cancel()): a
                # normal turn-ending, NOT an error. Preserve the partial turn's
                # completed history so the next message in this session can
                # continue from it. ``new_messages()`` excludes message_history
                # and older turns; unresolved tool calls are already closed out
                # with synthesized ``outcome='interrupted'`` returns by
                # pydantic-ai, so feeding this back never re-runs pending side
                # effects. We return (clean end of stream) rather than re-raising
                # RunCancelled so _turn_frames does not emit an error frame
                # ("The agent run was cancelled.") -- the UI ends the turn itself.
                self.history = [*self.history, *exc.new_messages()]
                return
            except (ModelHTTPError, UnexpectedModelBehavior) as exc:
                # A mid-stream/upstream failure from the model client (e.g.
                # OpenRouter `502 Network connection lost` on deepseek-v4-flash).
                # The turn never settled, but pydantic-ai already captured the
                # PARTIAL messages (streamed text/thinking + any tool calls) in
                # the run's message history — the exact same snapshot RunCancelled
                # uses. Persist them so the partial survives page refresh/resume
                # instead of being silently lost, then re-raise so the SSE layer
                # surfaces a clean error frame to the user.
                if _active_run is not None:
                    new_msgs = _active_run.new_messages()
                    if new_msgs:
                        self.history = [*self.history, *new_msgs]
                        _span.set_attribute(
                            "langfuse.observation.output",
                            _json_dumps(
                                {
                                    "partial_saved": len(new_msgs),
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            ),
                        )
                raise
            assert result is not None, "iter() ended before AgentRun.result settled"
            _span.set_attribute(
                "langfuse.observation.output", _json_dumps(_run_output(result))
            )
        self.last_result = result
        self.history = [*self.history, *result.new_messages()]
        # Post-turn background work (consolidation/review/thread-summarize) is
        # scheduled by the attached coordinator. Bare sessions (no coordinator,
        # e.g. a memory-less profile) skip straight past — the turn core never
        # cares about background machinery.
        if self.background is not None:
            await self.background.after_turn(scoped)

    async def run(
        self,
        prompt: str,
        *,
        conversation_id: str | None = None,
    ) -> str:
        """Run one turn and return the complete final text (async).

        Thin convenience over :meth:`stream` that keeps only the answer text
        (``kind == "text"``) and discards tool/thinking events.
        """
        parts: list[str] = []
        async for event in self.stream(prompt, conversation_id=conversation_id):
            if isinstance(event, StreamEvent) and event.kind == "text":
                parts.append(event.text)
        return "".join(parts)

    # -- thin sync convenience ------------------------------------------------

    def run_sync(self, prompt: str, *, conversation_id: str | None = None) -> str:
        """Run one turn synchronously; returns the final text.

        Thin wrapper for scripts/tests that don't manage their own loop: creates
        a short-lived loop, runs the async core inside it, returns the result.
        """
        return asyncio.run(self.run(prompt, conversation_id=conversation_id))

    # -- durable-turn ownership -----------------------------------------------

    def start_turn(
        self, prompt: str, *, conversation_id: str | None = None
    ) -> TurnHandle:
        """Start a Session-owned turn, or re-attach to the running one.

        The turn runs in a background task decoupled from any request; callers
        attach with :meth:`TurnHandle.subscribe`. If a turn is already running
        on this session, returns that running handle (the prompt is ignored —
        re-attach, per the durable-turns design); start a new turn by cancelling
        or letting the current one finish first. The turn's CancellationToken is
        created synchronously so an immediate ``cancel()`` is never dropped.
        """
        if self._turn is not None:
            return self._turn
        token = CancellationToken()
        self._token = token
        self._turn = TurnHandle(self, prompt, conversation_id, token)
        return self._turn

    def _clear_turn(self, handle: TurnHandle) -> None:
        """Drop the owned-turn ref once its pump settles (idempotent)."""
        if self._turn is handle:
            self._turn = None

    # -- steering / control ---------------------------------------------------

    async def cancel(self) -> None:
        """Stop the in-flight turn (CancellationToken cancel).

        Returns once the token is cancelled; the running ``stream()`` loop
        observes it at its next natural boundary and ends with ``RunCancelled``
        (carrying completed history for resume). No-op if nothing is running.
        """
        self._token.cancel()

    async def enqueue(self, content: str, *, priority: str = "asap") -> str | None:
        """Steer the running agent at the next node boundary (AgentRun.enqueue).

        Injects ``content`` as a pending message that the active ``AgentRun``
        picks up at its next natural boundary (not mid-generation). Requires a
        run to be live (``self._run``); returns the ``enqueue_id`` on success
        or ``None`` if nothing is running. ``cancel()`` (not enqueue) is the
        way to interrupt mid-generation.
        """
        run = self._run
        if run is None:
            return None
        return run.enqueue(content, priority=priority)

    # -- observability lifecycle ---------------------------------------------

    def close(self) -> None:
        """Flush buffered spans and disable tracing before process exit."""
        if self._turn is not None:
            self._turn.cancel()  # stop a running owned turn (best-effort)
        if self.trace:
            shutdown_observability()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
