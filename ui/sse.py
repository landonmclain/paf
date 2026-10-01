"""paf SSE transport — thin HTTP stream over the ``Session`` runtime.

Step 0 of the paf-ui plan (see vault ``Web-UI-Event-Streams-Candidate.md``):
a minimal FastAPI surface that exposes the runtime's turn event stream + the
background-task timeline over standard SSE, with REST controls for
steer/cancel, so a decoupled web frontend (paf-ui, Next.js + AI Elements) can
consume paf without Gradio.

Design (locked 2026-09-02, SSE + REST, per-lane; updated 2026-09-13 Phase 1 of
Unified-Background-Process-Feed-Spec):
- One ``Session`` per conversation id, held in an in-memory registry. Each
  active conversation opens its OWN ``/api/chat`` SSE stream; steer/cancel are
  per-lane REST POSTs addressed to that lane's conversation id.
- The runtime owns resume/date-stamping/``{profile}:{cid}`` scoping — we pass
  the raw prompt + a plain conversation id and let ``Session.stream()`` do it.
- ``/api/chat`` emits turn events as ``event: <kind>`` frames ONLY. Background
  task frames are NOT merged into the turn stream — they ride the dedicated
  ``GET /api/processes/stream`` feed (turn-independent, no turn-end sentinel),
  so background maintenance can never interleave into (and tear) a thinking/
  text block.

Transport only — no runtime changes, no pydantic-ai imports beyond Session.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Message/part types for serializing restored conversation history
# (session.history is a list of pydantic-ai ModelMessage).
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from paf.background_processes._events import (
    ProcessEvent,
    ProcessEventType,
    get_agent_emitter,
)
from paf.capabilities import PROFILES
from paf.capabilities.planning._events import PlanEventType
from paf.capabilities.planning._registry import get_plan_emitter
from paf.capabilities.planning._store import FilePlanStore
from paf.config import PROJECT_ROOT
from paf.conversations import (
    ConversationMetaStore,
    ProjectExistsError,
    ProjectNotFoundError,
    ProjectStore,
    default_project_store,
)
from paf.observability.instrumentation import root_span
from paf.runtime import Session, StreamEvent, ToolError, ToolRunning, TurnStreamEvent

if TYPE_CHECKING:
    from paf.capabilities._storage import FlatFileStorage

log = logging.getLogger(__name__)

# Default agent when the caller doesn't specify one — mirrors ui/web.py's
# "demo" profile convention so a bare /api/chat gives a working demo lane.
DEFAULT_AGENT = "demo_assistant"


def _profile_for(agent_name: str) -> Any | None:
    """Resolve the spec-declared profile for an agent, matching ui/web.py.

    demo_assistant -> DEMO_PROFILE ("demo"); everything else -> the spec's
    declared profile (or None, letting the Session fall back to its default).
    """
    from paf.specs import resolve_profile_name

    pname = resolve_profile_name(agent_name)
    if pname and pname in PROFILES:
        return PROFILES[pname]
    return PROFILES.get("demo") if agent_name == DEFAULT_AGENT else None


# ---------------------------------------------------------------------------
# Session registry: "{agent}:{conversation_id}" -> Session. One Session per lane.
# ---------------------------------------------------------------------------
_sessions: dict[str, Session] = {}
_sessions_last_used: dict[str, float] = {}

# Idle TTL: a lane with no activity for IDLE_TTL_SECONDS is closed and evicted
# (Session.close() flushes spans; a re-touch lazily recreates + resumes from the
# store via the same {agent}:{conversation_id}, so an idle lane is cheap, not lost).
IDLE_TTL_SECONDS = 30 * 60  # 30 min


def _session_key(agent: str, conversation_id: str) -> str:
    return f"{agent}:{conversation_id}"


def _now() -> float:
    # Wall clock, not monotonic: last-used doubles as the lane's `updated_at`,
    # which is compared against persisted run start times (epoch) and shown in
    # the UI. The idle sweeper only needs differences, which this still gives.
    return time.time()


async def _get_session(conversation_id: str, agent: str) -> Session:
    key = _session_key(agent, conversation_id)
    session = _sessions.get(key)
    if session is None:
        session = Session(
            agent,
            profile=_profile_for(agent),
            conversation_id=conversation_id,
            resume_window=300,
            on_complete=functools.partial(_notify_agent, agent),
        )
        session.attach_background()
        _sessions[key] = session
    _sessions_last_used[key] = _now()
    return session


def _notify_message(run_id: str, detail: str) -> str:
    """The ``[system]`` text a completion notify delivers to the parent.

    Subagent tasks (``subagent.<cid>.<n>``) hand over a ready-made message —
    the whole output or a truncated head plus a ``check_task`` pointer, built by
    ``paf.capabilities.subagents._notify`` — so it is delivered verbatim. Shell
    processes pass a short detail (``exit 0``) that is wrapped here.
    """
    if run_id.startswith("subagent."):
        return detail
    return f"[system] Background process {run_id} finished: {detail}"


async def _notify_agent(
    agent: str, run_id: str, conversation_id: str | None, ok: bool, detail: str
) -> None:
    """Deliver a background-process completion message to the originating lane.

    The shell queue fires this when a ``notify_on_completion`` process finishes,
    and the Subagents capability when a background delegation settles.
    The queue passes the already-scoped cid (``{profile}:{cid}``); we strip the
    profile prefix and look up (or create) the registry Session so the notify
    lands in the SAME session the UI's ``/api/chat``/history reads — making the
    completion visible live instead of only after a server restart. Delivery
    goes through ``Session.notify`` (the Phase-3 steer gate): the message is
    steered into a live AgentRun when one is running, else it becomes a new
    detached turn — never a raw ``stream()``, which would clobber a live turn's
    ``_token``/``_run``/``history``.
    """
    if conversation_id is None:
        return
    raw_cid = (
        conversation_id.split(":", 1)[1] if ":" in conversation_id else conversation_id
    )
    session = await _get_session(raw_cid, agent)
    # A NEW trace, grouped with the conversation by session id. This runs in the
    # background worker, whose inherited current span is the turn that started
    # the job (often long finished) — nesting under it buried the notify turn
    # and everything it triggers (memory, summaries) inside that old trace. The
    # notify turn's task is created inside this span, so it nests here instead.
    with root_span(
        f"paf.notify.{run_id}",
        new_trace=True,
        **{"langfuse.session.id": conversation_id},
        agent=agent,
        conversation_id=conversation_id,
        run_id=run_id,
        ok=ok,
        detail=detail,
    ):
        started = session.notify(_notify_message(run_id, detail))
    # Only when the notify actually STARTED a new turn (idle lane) do we signal
    # the conversation-lane process feed so the UI re-attaches and streams the
    # model response. When it was steered into a live turn, that turn is already
    # streaming to the UI, so no signal is needed.
    if started is None:
        await _emit_turn_started(agent, conversation_id, run_id)


async def _emit_turn_started(agent: str, conversation_id: str, run_id: str) -> None:
    """Signal the conversation-lane process feed that a notify turn started.

    Emits a ``turn_started`` ProcessEvent on the per-agent emitter so the
    matching conversation feed forwards it as a ``turn:`` frame. The UI uses
    that frame to re-attach to the detached notify turn (``rejoinTurn``) and
    stream its model response. Only called when the notify actually started a
    new turn (idle lane); a steered notify is already part of a streaming turn.
    """
    try:
        emitter = get_agent_emitter(agent)
        await emitter.emit_process(
            ProcessEvent(
                id=run_id,
                label="",
                status=ProcessEventType.turn_started,
                lane="conversation",
                scope=conversation_id,
            )
        )
    except Exception:  # noqa: BLE001 - a broken signal must never break the notify
        log.warning("turn-started emit failed", exc_info=True)


# ---------------------------------------------------------------------------
# Conversation meta + project registry stores
#
# User-owned stores (``paf.conversations``) behind the shared storage seam,
# kept OUT of ``data/memory/`` — conversation titles/projects are a
# UI/conversation concern, not agent memory. Conversation meta is per agent:
# ``<project_root>/data/conversations/<agent>/conversation-titles.json``, keyed
# by bare conversation_id (the per-agent file already scopes it, so a rename
# can never bleed into another agent's lane of the same id). The project
# registry is global: ``<project_root>/data/projects/projects.json``.
# ---------------------------------------------------------------------------


def _title_store(agent: str) -> FlatFileStorage:
    """The seam-backed storage rooted at the neutral conversations dir.

    Tests swap this for a temp-rooted ``FlatFileStorage`` — the storage root is
    the only thing the seam needs to change.
    """
    from paf.capabilities._storage import FlatFileStorage

    return FlatFileStorage(PROJECT_ROOT / "data" / "conversations")


def _meta_store(agent: str) -> ConversationMetaStore:
    """One agent's conversation meta (title + project per lane)."""
    return ConversationMetaStore(_title_store(agent), agent)


def _project_store() -> ProjectStore:
    """The global project registry."""
    return default_project_store()


def _session_is_busy(session: Session) -> bool:
    """True if a session has a live turn with an attached subscriber.

    The idle sweeper must not reap a session that is actively streaming to a
    connected client (a live stream can outlive the attach-time last_used
    timestamp). A running turn with NO subscriber is an orphan and stays
    reapable — that is the cost guard.
    """
    turn = getattr(session, "_turn", None)
    return (
        turn is not None
        and not getattr(turn, "done", True)
        and bool(getattr(turn, "has_subscribers", False))
    )


def _close_session(key: str) -> None:
    """Close + evict one lane (flush spans, drop the worker). Tolerant of races."""
    session = _sessions.pop(key, None)
    _sessions_last_used.pop(key, None)
    if session is not None:
        try:
            session.close()
        except Exception:  # noqa: BLE001 - closing must never break the request
            log.exception("close failed for session %s", key)
            session = None


async def _close_idle_loop() -> None:
    """Periodically close lanes idle past ``IDLE_TTL_SECONDS`` (started on startup).

    A session mid-live-stream (a running turn with an attached subscriber) is
    never reaped even if its attach-time last_used has gone stale; a running
    turn with no subscriber is an orphan and is reaped (the cost guard), which
    cancels it via ``Session.close()``.
    """
    while True:
        await asyncio.sleep(60)  # sweep every minute
        cutoff = _now() - IDLE_TTL_SECONDS
        stale: list[str] = []
        for key, t in list(_sessions_last_used.items()):
            if t >= cutoff:
                continue
            session = _sessions.get(key)
            if session is not None and _session_is_busy(session):
                continue
            stale.append(key)
        for key in stale:
            log.info(
                "closing idle session %s (no activity in %ds)", key, IDLE_TTL_SECONDS
            )
            _close_session(key)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start the idle sweeper on boot; close every session on shutdown.

    Holds a process-lifetime *anchor* ref on observability so the 30-min idle
    sweeper can never drive the shared OTel provider's ref-count to 0 mid-run.
    If it did, ``shutdown_observability()`` tears the GLOBAL provider down and
    ``trace.set_tracer_provider`` is set-once (a dead provider cannot be
    re-armed) — so every later lane silently loses tracing. The anchor keeps
    refs >= 1 for the process life; the balance in the `finally` releases it on
    a clean shutdown so a genuine process exit still flushes + shuts down.
    """
    from paf.background import _background_runner
    from paf.observability.instrumentation import (
        setup_observability,
        shutdown_observability,
    )

    setup_observability()  # process anchor: refs never hit 0 during idle eviction
    sweeper = asyncio.create_task(_close_idle_loop())
    try:
        yield
    finally:
        sweeper.cancel()
        for key in list(_sessions.keys()):
            _close_session(key)
        await _background_runner.shutdown()
        shutdown_observability()  # release the anchor (all lanes already closed)


app = FastAPI(title="paf SSE transport", lifespan=_lifespan)

# Narrow CORS for the paf-ui dev server (localhost:3000) + the Tailscale host
# used for phone/other-device access. The browser POSTs cross-origin to this
# :8090 service; without this the frontend fetch is blocked. Keep it minimal —
# no wildcard, only the origins paf-ui actually runs on.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",  # paf-ui dev falls to :3001 while Langfuse holds :3000
        "http://127.0.0.1:3001",
        "http://100.107.176.116:3000",
        "http://100.107.176.116:3001",  # Tailscale host (paf-ui via phone/devices)
        "http://paf.example.com:3001",  # Tailscale machine name (preferred)
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# SSE framing + event serialization
# ---------------------------------------------------------------------------
def _frame(event: str, data: dict[str, Any]) -> str:
    """One SSE frame: ``event: <event>`` + ``data: <json>`` + blank line."""
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _serialize_turn(ev: TurnStreamEvent) -> tuple[str, dict[str, Any]]:
    """Map a turn event to (sse-event-name, json-payload)."""
    if isinstance(ev, StreamEvent):
        return ev.kind, {
            "text": ev.text,
            "tool_name": ev.tool_name or None,
            "args": ev.args or None,
        }
    if isinstance(ev, ToolRunning):
        return "tool_running", {"tool_name": ev.tool_name, "step_id": ev.step_id}
    if isinstance(ev, ToolError):
        return "tool_error", {"step_id": ev.step_id, "error": ev.error}
    return "unknown", {"raw": repr(ev)}


def _task_frame(got: Any) -> tuple[str, dict[str, Any]] | None:
    """Map a ProcessEvent -> `task:` SSE frame. Returns None when the event is
    the stream-termination sentinel (None)."""
    if got is None:
        return None
    status = getattr(got.status, "value", got.status)
    return (
        "task",
        {
            "id": got.id,
            "label": got.label,
            "status": status,
            "progress": got.progress,
            "detail": got.detail,
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict[str, bool]:
    """Liveness probe for the systemd unit / load balancer / UI boot check."""
    return {"ok": True}


def _serialize_plan_items(items: list[Any]) -> list[dict[str, Any]]:
    """Serialize plan items to the UI shape ({id, content, status, ...})."""
    return [
        {
            "id": getattr(i, "id", ""),
            "content": getattr(i, "content", ""),
            "status": getattr(i, "status", "pending"),
            "active_form": getattr(i, "active_form", "") or "",
            "parent_id": getattr(i, "parent_id", None),
            "depends_on": list(getattr(i, "depends_on", []) or []),
        }
        for i in items
    ]


async def _plan_snapshot(scoped_cid: str) -> list[dict[str, Any]]:
    """Read the authoritative plan for a scoped conversation id.

    Uses the SAME FilePlanStore seam as the agent's producer (same root +
    scoped cid + shared emitter key), so this endpoint and the agent's writes
    can never point at different files. Returns empty list when no plan exists.
    """
    store = FilePlanStore(
        root=PROJECT_ROOT / "data" / "plans", conversation_id=scoped_cid
    )
    items = await store.get_items()
    return _serialize_plan_items(items)


def _serialize_history(msgs: list[Any]) -> list[dict[str, Any]]:
    """Flatten restored pydantic-ai messages into UI-shaped history entries.

    A ``ModelRequest`` with a real string ``UserPromptPart`` is a user turn.
    A ``ModelResponse`` becomes a structured assistant turn: a ``log`` array of
    ``thinking`` / ``tool`` / ``text`` entries (mirroring the live turn's
    ``TurnLogEntry``) so restored turns render the full timeline — thinking
    blocks, per-tool rows, and the final answer — not just the final text.
    """
    out: list[dict[str, Any]] = []
    for msg in msgs:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if (
                    isinstance(part, UserPromptPart)
                    and isinstance(part.content, str)
                    and part.content
                ):
                    out.append({"role": "user", "text": part.content})
        elif isinstance(msg, ModelResponse):
            log: list[dict[str, Any]] = []
            # Coalesce consecutive ThinkingParts into one block (they stream in
            # many small pieces, matching how the live UI groups by "chunk").
            pending_thinking: list[str] = []
            for part in msg.parts:
                if isinstance(part, ThinkingPart) and isinstance(part.content, str):
                    if part.content:
                        pending_thinking.append(part.content)
                    continue
                if pending_thinking:
                    log.append(
                        {"kind": "thinking", "text": "\n\n".join(pending_thinking)}
                    )
                    pending_thinking = []
                if isinstance(part, TextPart) and isinstance(part.content, str):
                    if part.content:
                        log.append({"kind": "text", "text": part.content})
                elif isinstance(part, ToolCallPart):
                    log.append(
                        {
                            "kind": "tool",
                            "stepId": f"h-{part.tool_call_id}",
                            "toolName": part.tool_name,
                            "args": part.args,
                            "status": "done",
                            "result": None,
                        }
                    )
                elif isinstance(part, ToolReturnPart):
                    # Fill in the result/error on the last matching tool entry.
                    for e in reversed(log):
                        if (
                            e.get("kind") == "tool"
                            and e.get("toolName") == part.tool_name
                            and e.get("stepId") == f"h-{part.tool_call_id}"
                        ):
                            if part.outcome.is_error():
                                e["status"] = "error"
                                e["error"] = str(part.content)
                            else:
                                e["result"] = (
                                    part.content
                                    if isinstance(part.content, (str, int, float, bool))
                                    or part.content is None
                                    else str(part.content)
                                )
                            break
            if pending_thinking:
                log.append({"kind": "thinking", "text": "\n\n".join(pending_thinking)})
            if log:
                out.append({"role": "assistant", "log": log})
    return out


@app.get("/api/conversations/{conversation_id}/plan")
async def conversation_plan(
    conversation_id: str, agent: str = DEFAULT_AGENT
) -> dict[str, Any]:
    """Authoritative plan snapshot for a conversation (source of truth).

    Reads the per-conversation ``FilePlanStore`` via the same seam the agent's
    own producer uses (same root + scoped cid + shared emitter key), so this
    endpoint and the agent's writes can never drift apart. The sidebar loads
    this on conversation switch / turn-settle / page refresh; live mid-turn
    ticks ride the ``plan`` SSE frames on ``/api/chat`` instead.

    Returns real ids/statuses; ``items: []`` when the conversation has no plan.
    """
    session = await _get_session(conversation_id, agent)
    scoped = session._scoped_cid(conversation_id) or conversation_id
    items = await _plan_snapshot(scoped)
    return {
        "conversation_id": conversation_id,
        "agent": agent,
        "items": items,
        "item_count": len(items),
    }


@app.get("/api/conversations/{conversation_id}/history")
async def conversation_history(
    conversation_id: str, agent: str = DEFAULT_AGENT
) -> dict[str, Any]:
    """Return the prior turns for a conversation (server-authoritative history).

    Restores the lane's prior messages from the store via the shared session
    registry (same {profile}:{cid} scoping + resume as /api/chat), then flattens
    them to ``[{role, text}]`` for the UI to render on load. Cheap when there is
    no prior history (``messages: []``).
    """
    session = await _get_session(conversation_id, agent)
    restored = await session.restore()
    return {
        "conversation_id": conversation_id,
        "messages": _serialize_history(session.history),
        "restored": restored,
    }


def _last_prompt(session: Session) -> str | None:
    """Best-effort last user prompt from a session's history (None if none)."""
    try:
        for msg in reversed(session.history):
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart) and isinstance(
                        part.content, str
                    ):
                        return part.content[:120]
    except Exception:  # noqa: BLE001 - title is sugar; never fail a list
        return None
    return None


@app.get("/api/conversations")
async def list_conversations(agent: str = DEFAULT_AGENT) -> dict[str, Any]:
    """List known conversation lanes for the left sidebar, newest first.

    Returns id + last user prompt + last-activity time. The result is a UNION of
    two sources so a sidebar survives daemon restarts:

    1. The live in-memory registry (current titles / last-activity, real-time).
    2. The persisted step store (``FileStepStore.list_runs``), grouped by
       conversation id — this is what makes idle/restarted lanes still appear.
       Only lanes scoped to this agent's profile (``<profile>:<cid>``) are
       included so CLI/smoke/benchmark/UUID lanes in the shared store are not
       surfaced.
    """
    profile = _profile_for(agent)
    store_prefix = f"{getattr(profile, 'name', agent)}:"
    agent_prefix = f"{agent}:"

    by_lane: dict[str, dict[str, Any]] = {}

    # 1. Live registry (real-time titles / activity).
    for key, last in _sessions_last_used.items():
        if not key.startswith(agent_prefix):
            continue
        conversation_id = key[len(agent_prefix) :]
        session = _sessions.get(key)
        by_lane[conversation_id] = {
            "conversation_id": conversation_id,
            "agent": agent,
            "title": _last_prompt(session) if session is not None else None,
            "updated_at": last,
        }

    # 2. Persisted store backstop (durability across restarts).
    try:
        from paf.capabilities.persistence import _build_store

        store = _build_store()
        for run in await store.list_runs():
            cid = run.conversation_id
            if not cid or not cid.startswith(store_prefix):
                continue
            lane_id = cid[len(store_prefix) :]
            started = run.started_at.timestamp() if run.started_at else 0.0
            cur = by_lane.get(lane_id)
            if cur is not None and (cur.get("updated_at") or 0) >= started:
                continue  # live registry already has this lane, at least as fresh
            by_lane[lane_id] = {
                "conversation_id": lane_id,
                "agent": agent,
                # No live session to read a last prompt from — show the id so
                # persisted lanes are identifiable, not a generic placeholder.
                "title": lane_id,
                "updated_at": started,
            }
    except Exception:  # noqa: BLE001 - store scan must never break the endpoint
        log.warning("store backstop scan failed", exc_info=True)

    out = list(by_lane.values())

    # 3. Overlay user-set meta (conversation-titles.json) — the user's rename
    #    always wins over the auto-derived title, and survives daemon restarts;
    #    the assigned project (if any) rides along on every lane.
    meta = _meta_store(agent).load()
    for lane in out:
        saved = meta.get(lane["conversation_id"])
        saved = saved if isinstance(saved, dict) else {}
        if saved.get("title"):
            lane["title"] = saved["title"]
        elif not lane.get("title"):
            lane["title"] = "New conversation"
        lane["project"] = saved.get("project") or None
        lane["hidden"] = bool(saved.get("hidden"))
    out.sort(key=lambda c: c.get("updated_at") or 0, reverse=True)
    return {"conversations": out}


@app.post("/api/conversations")
async def create_conversation(
    agent: str = DEFAULT_AGENT, project: str | None = None
) -> dict[str, Any]:
    """Allocate a fresh conversation lane id for the \"New conversation\" flow.

    The session itself is lazily created on first /api/chat; here we just mint a
    stable id and register its last-used timestamp so it appears in the list.
    ``project`` (optional) assigns the new lane to a registered project.
    """
    if project:
        _require_assignable_project(project)
    conversation_id = f"paf-{uuid.uuid4().hex[:12]}"
    _sessions_last_used[_session_key(agent, conversation_id)] = _now()
    if project:
        _meta_store(agent).set_project(conversation_id, project)
    return {"conversation_id": conversation_id, "agent": agent, "project": project}


def _require_assignable_project(slug: str) -> None:
    """404 for an unknown project, 400 for an archived one."""
    entry = _project_store().get(slug)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown project {slug!r}"
        )
    if entry.get("archived"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"project {slug!r} is archived",
        )


@app.patch("/api/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str, body: dict[str, Any], agent: str = DEFAULT_AGENT
) -> dict[str, Any]:
    """Set (or clear) a conversation's title, project, and/or hidden flag, durably.

    Persists to the per-agent ``conversation-titles.json`` sidecar under
    ``data/conversations/<agent>/`` — kept OUT of ``data/memory/`` — so a change
    is exclusive to one agent's lane and survives restarts. Each field is
    applied only when its key is present:

    - ``title``: a non-empty title replaces the derived one; an empty string
      clears it back to the auto-derived title (the project is kept).
    - ``project``: a registered, non-archived project slug assigns it; ``null``
      or ``""`` unassigns. Assignment affects memory extraction from here on.
    - ``hidden``: true hides the lane from the default sidebar view; false
      shows it again. Nothing is deleted.
    """
    if not any(k in body for k in ("title", "project", "hidden")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="title, project, or hidden required",
        )
    if "title" in body and body["title"] is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="title must be a string"
        )

    project: str | None = None
    if "project" in body:
        project = str(body["project"] or "").strip() or None
        if project is not None:
            _require_assignable_project(project)

    store = _meta_store(agent)
    saved = store.get(conversation_id)
    if "title" in body:
        saved = store.set_title(conversation_id, str(body["title"]))
    if "project" in body:
        saved = store.set_project(conversation_id, project)
    if "hidden" in body:
        saved = store.set_hidden(conversation_id, bool(body["hidden"]))

    return {
        "conversation_id": conversation_id,
        "agent": agent,
        "title": saved.get("title") or None,
        "project": saved.get("project") or None,
        "hidden": bool(saved.get("hidden")),
    }


# ---------------------------------------------------------------------------
# Projects — the user's registry (create / list / edit / archive)
# ---------------------------------------------------------------------------


@app.get("/api/projects")
async def list_projects(include_archived: bool = False) -> dict[str, Any]:
    """Registered projects, sorted by name; archived ones only when asked."""
    return {"projects": _project_store().list(include_archived=include_archived)}


@app.post("/api/projects", status_code=status.HTTP_201_CREATED)
async def create_project(body: dict[str, Any]) -> dict[str, Any]:
    """Register a project. The slug is derived from ``name`` and never changes."""
    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="name required"
        )
    try:
        return _project_store().create(name, str(body.get("description") or ""))
    except ProjectExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"project {exc.args[0]!r} already exists",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@app.patch("/api/projects/{slug}")
async def update_project(slug: str, body: dict[str, Any]) -> dict[str, Any]:
    """Edit a project's name/description, or (un)archive it. Never deletes."""
    if not any(k in body for k in ("name", "description", "archived")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="name, description, or archived required",
        )
    name, description, archived = (
        body.get("name"),
        body.get("description"),
        body.get("archived"),
    )
    try:
        return _project_store().update(
            slug,
            name=None if name is None else str(name),
            description=None if description is None else str(description),
            archived=None if archived is None else bool(archived),
        )
    except ProjectNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown project {slug!r}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@app.get("/api/conversations/{conversation_id}/tasks")
async def conversation_tasks(
    conversation_id: str, agent: str = DEFAULT_AGENT
) -> dict[str, Any]:
    """Background-task history for a conversation, newest first.

    Background tasks (thread summarizer, memory reviewer, consolidation) can
    finish AFTER the turn's SSE stream closes, so a failure is never visible on
    the live turn channel. This endpoint lets a UI query "did anything fail for
    this conversation" — on load, on turn-end, or on-demand — and surface e.g.
    ``status: error`` rows with their ``detail``. Rows are advisory/transient
    (kept in the session registry only), not a durable audit log.
    """
    session = await _get_session(conversation_id, agent)
    bg = session.background
    scoped_cid = session._scoped_cid(conversation_id) or conversation_id
    tasks = bg.task_history(scoped_cid) if bg is not None else []
    return {
        "conversation_id": conversation_id,
        "agent": agent,
        "tasks": [
            {
                "id": t.id,
                "label": t.label,
                "status": t.status,
                "progress": t.progress,
                "detail": t.detail,
                "updated_at": t.updated_at,
            }
            for t in tasks
        ],
    }


@app.get("/api/processes/stream")
async def processes_stream(
    request: Request,
    conversation_id: str | None = None,
    agent: str = DEFAULT_AGENT,
    lane: Literal["agent", "conversation"] = "conversation",
) -> StreamingResponse:
    """Long-lived, turn-independent SSE feed of background processes/lifecycle.

    Decoupled from ``/api/chat`` (see the Unified-Background-Process-Feed-Spec,
    Phase 1): background ``task:`` frames are no longer merged into the turn
    stream — they ride this dedicated feed instead, which opens ONCE per lane
    and stays open as long as the client wants, with no turn-end sentinel.

    Lane-aware: ``lane=agent`` streams the shared per-agent lane (memory
    consolidation + review, keyed by ``agent``, no conversation); ``lane=
    conversation`` streams one conversation's lane (thread summarizer, keyed by
    ``conversation_id``). A consumer subscribes to each lane it wants,
    separately.

    Wire contract is identical to the old in-turn ``task:`` frames (id / label /
    status / progress / detail / updated_at), so the adapter and ProcessesPanel
    consume them unchanged. On connect we first restore the lane's current task
    history (terminal rows, e.g. a finished/failed summarizer), then subscribe
    for live transitions (queued -> running -> done|error|skipped).
    """
    return _process_response(_process_feed, conversation_id, agent, lane, request)


def _process_response(
    feed: Callable[[Any, str | None, Request], AsyncIterator[str]],
    conversation_id: str | None,
    agent: str,
    lane: str,
    request: Request,
) -> StreamingResponse:
    """Adapter from a lane-scoped feed generator to an SSE StreamingResponse."""

    async def gen() -> AsyncIterator[str]:
        session = await _get_session(conversation_id, agent)
        bg = session.background
        if bg is None:
            return
        if lane == "agent":
            lane_bg = bg.agent
            scope: str | None = None
        else:
            lane_bg = bg.conversation
            scope = session._scoped_cid(conversation_id) or conversation_id
        if lane_bg is None:
            return
        async for frame in feed(lane_bg, scope, request):
            yield frame

    return StreamingResponse(gen(), media_type="text/event-stream")


def _process_feed(bg: Any, scope: str | None, request: Request) -> AsyncIterator[str]:
    """Stream background-process frames for one lane, never tying to a turn.

    On connect: restore the lane's current task history (terminal + active
    rows), then subscribe to the lane coordinator for live transitions. The
    feed stays open until the client disconnects — no turn-end sentinel.
    ``scope`` is the conversation id for the conversation lane, or None for the
    agent lane (which reads all agent-scoped rows).
    """

    async def _inner() -> AsyncIterator[str]:
        q: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        closed = {"v": False}

        def _publish(frame: tuple[str, dict[str, Any]] | None) -> None:
            if frame is None or closed["v"]:
                return
            q.put_nowait(frame)

        # 1. Restore terminal + active history first (so a late-joining tab sees
        #    the full current state, not just live deltas).
        if bg is not None:
            for task in bg.task_history(scope):
                _publish(
                    (
                        "task",
                        {
                            "id": task.id,
                            "label": task.label,
                            "status": task.status,
                            "progress": task.progress,
                            "detail": task.detail,
                        },
                    )
                )

        # 1b. Always-on "idle" snapshot of the 3 background jobs so the panel is
        # never empty: consolidation / review / summarizer show their idle state +
        # count + reason even when nothing is running. Rows use STABLE job ids so
        # a later live transition (queued/running/done) coalesces over the idle
        # row instead of duplicating it. Re-published on a slow heartbeat so the
        # countdown ticks and a completed run returns its row to idle.
        async def _publish_idle() -> None:
            if bg is None:
                return
            try:
                idle = await bg.job_status_summary(scope)
                for st in idle:
                    _publish(
                        (
                            "task",
                            {
                                "id": st.id,
                                "label": st.label,
                                "status": st.status,
                                "progress": st.progress,
                                "detail": st.detail,
                            },
                        )
                    )
            except Exception:
                log.warning("idle process snapshot failed", exc_info=True)

        await _publish_idle()

        # 1c. Restore this conversation's shell background processes. Shell rows
        #     live in the durable shell store (NOT the coordinator's
        #     task_history restored above), so a reconnect must read them here
        #     or a running / recently-finished shell process would vanish after
        #     a refresh or conversation switch. Agent lane has no scope -> skip.
        if scope is not None:
            try:
                from paf.capabilities.shell._background import conversation_shell_frames

                for frame in conversation_shell_frames(
                    PROJECT_ROOT / "data" / "processes", bg.agent_name, scope
                ):
                    _publish(("task", frame))
            except Exception:
                log.warning("shell process restore failed", exc_info=True)
            # Subagent tasks live in their own store too; restore them the same
            # way so running / recently finished rows survive a refresh.
            try:
                from paf.capabilities.subagents._store import (
                    conversation_subagent_frames,
                )

                for frame in conversation_subagent_frames(
                    PROJECT_ROOT / "data" / "processes", bg.agent_name, scope
                ):
                    _publish(("task", frame))
            except Exception:
                log.warning("subagent task restore failed", exc_info=True)

        # No wall-clock heartbeat: every background job is TURN-gated, so task
        # events (SKIPPED/PROCESSING) already fire on every ``after_turn`` and
        # the subscriber below refreshes the panel live. The only residue a timer
        # previously covered was returning a *completed* run's row back to idle;
        # that is handled explicitly when a task reaches a terminal state.

        # 2. Subscribe for live events on the shared per-agent emitter, filtered
        #    to THIS lane/scope. The emitter is per-agent (both lanes of an agent
        #    share one bus), so we filter: an agent-lane feed keeps only agent
        #    events; a conversation-lane feed keeps only that conversation's
        #    events. The durable store read above (task_history) is the reconnect
        #    history; the emitter is live-only. When a task reaches a TERMINAL
        #    state we also re-publish the idle snapshot so its stable row returns
        #    to idle (a completed fold no longer shows as running).
        emitter = None
        sub = None
        if bg is not None:
            emitter = get_agent_emitter(bg.agent_name)
            lane = bg.lane or "agent"

            def _matches(ev) -> bool:
                if lane == "agent":
                    return ev.lane == "agent"
                return ev.lane == "conversation" and ev.scope == scope

            def _on_task(ev) -> None:
                if not _matches(ev):
                    return
                # A turn_started event is a re-attach signal, not a process row:
                # forward it as a `turn:` frame so the UI can rejoin and stream
                # the detached notify turn.
                if getattr(ev.status, "value", ev.status) == "turn_started":
                    _publish(("turn", {"run_id": ev.id}))
                    return
                _publish(_task_frame(ev))
                if getattr(ev.status, "value", ev.status) in (
                    "done",
                    "error",
                    "skipped",
                ):
                    task = asyncio.create_task(_publish_idle())
                    task.add_done_callback(
                        lambda t: (
                            t.exception()
                            if not t.cancelled() and t.exception()
                            else None
                        )
                    )

            sub = emitter.on_any(_on_task)

        # Watch for the client disconnecting. This feed is intentionally
        # never-ending ("no turn-end sentinel"), so without a disconnect signal
        # it pins the SSE socket open forever — and uvicorn's graceful shutdown
        # waits on open connections until systemd's TimeoutStopSec SIGKILLs the
        # process (the ~90s restart stalls). When the client goes away, publish
        # the None sentinel so the loop ends via the same path as a natural stop.
        async def _watch_disconnect() -> None:
            try:
                while not await request.is_disconnected():
                    await asyncio.sleep(0.5)
            except Exception:  # noqa: BLE001 - a broken watch must not panic
                pass
            # Put the None sentinel DIRECTLY on the queue — NOT via _publish,
            # which deliberately swallows `None` (its guard is `frame is None ->
            # return`, meant to stop stray end-signals, but ours is the real one).
            q.put_nowait(None)

        disconnect_watcher = asyncio.create_task(_watch_disconnect())

        try:
            while True:
                frame = await q.get()
                if frame is None:
                    break
                yield _frame(*frame)
        finally:
            disconnect_watcher.cancel()
            closed["v"] = True
            if sub is not None and emitter is not None:
                try:
                    emitter.off_any(sub)  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001 - best-effort teardown
                    log.warning(
                        "processes_stream subscriber teardown failed", exc_info=True
                    )

    return _inner()


def _turn_frames(
    session: Session,
    prompt: str,
    conversation_id: str,
    request: Request,
    *,
    rejoin: bool = False,
) -> AsyncIterator[str]:
    """Subscribe to one Session-owned turn, forwarding its events as SSE.

    The turn starts (or re-attaches to) on the Session via ``start_turn`` and
    runs in a background task OWNED BY THE SESSION — not by this request. This
    generator is a *subscriber*: on client disconnect it detaches (stops
    writing to the dead socket) but the turn keeps running server-side, so a
    returning client can re-attach. Cancellation is exclusively the explicit
    ``POST /api/cancel``.

    ``rejoin=True`` is the refresh/visibility re-attach: attach to the in-flight
    turn if one is running (the prompt is ignored), else yield an EMPTY stream
    without starting a new turn — the client renders settled history instead.

    Background / maintenance work (``task:`` frames) is NOT mixed in here — it
    lives on its own dedicated ``GET /api/processes/stream`` feed so it never
    interleaves into the turn's content stream (which could split a thinking
    block mid-token). See the Unified-Background-Process-Feed-Spec.
    """

    async def _inner() -> AsyncIterator[str]:
        out_q: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        _scoped = getattr(session, "_scoped_cid", None)
        scoped_cid = (
            _scoped(conversation_id) if _scoped is not None else None
        ) or conversation_id

        if rejoin:
            # Refresh/visibility re-attach: subscribe to the in-flight turn if
            # one is running; otherwise yield an empty stream (the client
            # renders history) WITHOUT starting a new turn.
            handle = getattr(session, "_turn", None)
            if handle is None or handle.done:
                return
        else:
            # Start (or re-attach to) the Session-owned turn. If a turn is
            # already running on this lane, start_turn returns that running
            # handle and the prompt is ignored (re-attach, per the durable-turns
            # design).
            handle = session.start_turn(prompt, conversation_id=conversation_id)

        async def _on_plan(_ev: Any) -> None:
            # Full-snapshot per mutation (Plan-Live-Sidebar-Spec): read the
            # authoritative store and push the whole items array. Idempotent
            # against dropped/out-of-order frames; the panel is always a direct
            # mirror of server truth with zero client merge logic.
            try:
                items = await _plan_snapshot(scoped_cid)
            except Exception:  # noqa: BLE001 - a subscriber must never break the turn
                log.warning("plan snapshot for %s failed", scoped_cid, exc_info=True)
                return
            out_q.put_nowait(("plan", {"items": items}))

        # Observe the conversation's shared plan emitter (the SAME emitter the
        # agent's FilePlanStore writes through, via _registry). Any plan
        # mutation → a snapshot frame lands on the turn stream, mid-turn.
        emitter = get_plan_emitter(scoped_cid)
        plan_registrations: list[tuple[PlanEventType, Any]] = [
            (t, emitter.on(t, _on_plan)) for t in PlanEventType
        ]

        async def _forward_turn() -> None:
            # Forward the Session-owned turn's events as SSE frames from the
            # subscriber's attach point onward. A turn that ended in error has
            # handle.error set -> surface an error frame before the end.
            async for ev in handle.subscribe():
                await out_q.put(_serialize_turn(ev))
            if handle.error is not None:
                await out_q.put(("error", {"detail": str(handle.error)}))
            await out_q.put(None)  # sentinel: turn stream finished

        # Watch for the client disconnecting: end THIS SSE stream (detach the
        # subscriber) but leave the Session-owned turn running so a re-attach
        # can pick it up. Without it, the generator pins the socket open until
        # the turn finishes (uvicorn graceful shutdown waits on it).
        async def _watch_disconnect() -> None:
            try:
                while not await request.is_disconnected():
                    await asyncio.sleep(0.5)
            except Exception:  # noqa: BLE001 - a broken watch must not panic
                pass
            out_q.put_nowait(None)

        forward = asyncio.create_task(_forward_turn())
        disconnect_watcher = asyncio.create_task(_watch_disconnect())
        try:
            while True:
                item = await out_q.get()
                if item is None:
                    break  # turn finished OR client disconnected -- close the stream
                yield _frame(*item)
        finally:
            for evt_type, cb in plan_registrations:
                emitter.off(evt_type, cb)
            # Detach only: cancel the subscriber tasks. The Session-owned turn
            # (handle) keeps running — NO session.cancel() here.
            forward.cancel()
            disconnect_watcher.cancel()
            await asyncio.gather(forward, disconnect_watcher, return_exceptions=True)

    return _inner()


@app.post("/api/chat")
async def chat(body: dict[str, Any], request: Request) -> StreamingResponse:
    """Open an SSE stream for one turn on a conversation.

    Body: ``{conversation_id, prompt, agent?, rejoin?}``. Returns
    ``text/event-stream`` of TurnStreamEvent frames; forwards ``task:`` frames
    for the background timeline while the stream is open.

    ``rejoin: true`` (used for refresh/visibility re-attach) does NOT require a
    prompt: it attaches to the in-flight turn if one is running on the lane,
    else returns an empty stream without starting a new turn.
    """
    conversation_id = body.get("conversation_id")
    rejoin = bool(body.get("rejoin", False))
    if not conversation_id:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "conversation_id required"
        )
    if not rejoin and not body.get("prompt"):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "conversation_id + prompt required (or rejoin: true)",
        )
    session = await _get_session(
        str(conversation_id), str(body.get("agent", DEFAULT_AGENT))
    )

    frames = _turn_frames(
        session,
        str(body.get("prompt") or ""),
        str(conversation_id),
        request,
        rejoin=rejoin,
    )
    return StreamingResponse(frames, media_type="text/event-stream")


@app.post("/api/steer")
async def steer(body: dict[str, Any]) -> dict[str, Any]:
    """Inject a message into a running conversation (``Session.enqueue``)."""
    conversation_id = body.get("conversation_id")
    session = await _get_session(
        str(conversation_id), str(body.get("agent", DEFAULT_AGENT))
    )
    enq_id = await session.enqueue(
        str(body.get("content", "")), priority=str(body.get("priority", "asap"))
    )
    return {"enqueue_id": enq_id}


@app.post("/api/cancel")
async def cancel(body: dict[str, Any]) -> dict[str, Any]:
    """Stop the running turn on a conversation (``Session.cancel``)."""
    conversation_id = body.get("conversation_id")
    session = await _get_session(
        str(conversation_id), str(body.get("agent", DEFAULT_AGENT))
    )
    await session.cancel()
    return {"cancelled": True}


# --- entry point -----------------------------------------------------------
def main(port: int = 8090) -> None:
    """Run the transport standalone (uvicorn). Port 8090 keeps clear of Gradio's :8080."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
