"""paf SSE transport — thin HTTP stream over the ``Session`` runtime.

Step 0 of the paf-ui plan (see vault ``Web-UI-Event-Streams-Candidate.md``):
a minimal FastAPI surface that exposes the runtime's turn event stream + the
background-task timeline over standard SSE, with REST controls for
steer/cancel, so a decoupled web frontend (paf-ui, Next.js + AI Elements) can
consume paf without Gradio.

Design (locked 2026-09-02, SSE + REST, per-lane):
- One ``Session`` per conversation id, held in an in-memory registry. Each
  active conversation opens its OWN ``/api/chat`` SSE stream; steer/cancel are
  per-lane REST POSTs addressed to that lane's conversation id.
- The runtime owns resume/date-stamping/``{profile}:{cid}`` scoping — we pass
  the raw prompt + a plain conversation id and let ``Session.stream()`` do it.
- ``/api/chat`` emits turn events as ``event: <kind>`` frames and ALSO forwards
  the background-task timeline (``subscribe_tasks``) as ``event: task`` frames,
  so the UI sees live "memory consolidating" rows without a separate channel.

Transport only — no runtime changes, no pydantic-ai imports beyond Session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
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

from paf.capabilities import PROFILES
from paf.runtime import Session, StreamEvent, ToolError, ToolRunning, TurnStreamEvent

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
    return time.monotonic()


async def _get_session(conversation_id: str, agent: str) -> Session:
    key = _session_key(agent, conversation_id)
    session = _sessions.get(key)
    if session is None:
        session = Session(
            agent,
            profile=_profile_for(agent),
            conversation_id=conversation_id,
            resume_window=15,
        )
        _sessions[key] = session
    _sessions_last_used[key] = _now()
    return session


# ---------------------------------------------------------------------------
# Conversation titles store
#
# The ONE durable store of user-set conversation names. It is a proper
# consumer of the shared storage seam (``paf.capabilities._storage``) — not a
# flat-file-fixed write — so its backend is swappable like any other stored
# artifact. It lives OUTSIDE ``data/memory/`` deliberately: conversation titles
# are a UI/conversation concern, not agent memory, so they must not be coupled
# to the memory capability (which can be disabled or re-architected). Home:
# ``<project_root>/data/conversations/<agent>/conversation-titles.json``.
# Keyed by bare conversation_id because the per-agent key already scopes it
# (namespaced by agent, so a rename can never bleed into another agent's lane
# of the same id). Survives restarts; default edit is ``write`` (atomic
# rewrite of the id -> {title, updated_at} map).
# ---------------------------------------------------------------------------

_TITLES_KIND = "conversation-titles"


def _title_store(agent: str) -> "FlatFileStorage":
    """A seam-backed store rooted at the neutral conversations dir, per agent.

    <project_root>/data/conversations/<agent>/conversation-titles.json — a
    dedicated home that shares NO location or import with the memory
    capability, routed through the shared StorageService (FlatFileStorage).
    """
    from paf.capabilities._storage import FlatFileStorage
    from paf.config import PROJECT_ROOT

    root = PROJECT_ROOT / "data" / "conversations"
    return FlatFileStorage(root)


def _load_titles(agent: str) -> dict[str, dict[str, Any]]:
    """Read the agent's titles (None -> {}); never raises."""
    data = _title_store(agent).read(_TITLES_KIND, agent)
    return data if isinstance(data, dict) else {}


def _save_titles(agent: str, titles: dict[str, dict[str, Any]]) -> None:
    """Write the agent's titles atomically through the storage seam."""
    _title_store(agent).write(_TITLES_KIND, agent, titles)


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
    """Periodically close lanes idle past ``IDLE_TTL_SECONDS`` (started on startup)."""
    while True:
        await asyncio.sleep(60)  # sweep every minute
        cutoff = _now() - IDLE_TTL_SECONDS
        stale = [k for k, t in _sessions_last_used.items() if t < cutoff]
        for key in stale:
            log.info("closing idle session %s (no activity in %ds)", key, IDLE_TTL_SECONDS)
            _close_session(key)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start the idle sweeper on boot; close every session on shutdown."""
    sweeper = asyncio.create_task(_close_idle_loop())
    try:
        yield
    finally:
        sweeper.cancel()
        for key in list(_sessions.keys()):
            _close_session(key)


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
    """Map a BackgroundEvent -> `task:` SSE frame. Returns None when the event is
    the stream-termination sentinel (None)."""
    if got is None:
        return None
    meta = getattr(got, "metadata", {}) or {}
    return (
        "task",
        {
            "id": meta.get("id") or got.kind,
            "label": got.kind,
            "status": got.status,
            "progress": meta.get("progress"),
            "detail": got.detail,
            "updated_at": "",
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict[str, bool]:
    """Liveness probe for the systemd unit / load balancer / UI boot check."""
    return {"ok": True}


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


@app.get("/api/conversations/{conversation_id}/history")
async def conversation_history(conversation_id: str, agent: str = DEFAULT_AGENT) -> dict[str, Any]:
    """Return the prior turns for a conversation (server-authoritative history).

    Restores the lane's prior messages from the store via the shared session
    registry (same {profile}:{cid} scoping + resume as /api/chat), then flattens
    them to ``[{role, text}]`` for the UI to render on load. Cheap when there is
    no prior history (``messages: []``).
    """
    session = await _get_session(conversation_id, agent)
    restored = await session.restore()
    return {"conversation_id": conversation_id, "messages": _serialize_history(session.history), "restored": restored}


def _last_prompt(session: Session) -> str | None:
    """Best-effort last user prompt from a session's history (None if none)."""
    try:
        for msg in reversed(session.history):
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart) and isinstance(part.content, str):
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
        conversation_id = key[len(agent_prefix):]
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
            lane_id = cid[len(store_prefix):]
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

    # 3. Overlay user-set titles (conversation-titles.json) — the user's rename
    #    always wins over the auto-derived one, and survives daemon restarts.
    titles = _load_titles(agent)
    for lane in out:
        saved = titles.get(lane["conversation_id"])
        if saved and isinstance(saved, dict) and saved.get("title"):
            lane["title"] = saved["title"]
        elif not lane.get("title"):
            lane["title"] = "New conversation"
    out.sort(key=lambda c: c.get("updated_at") or 0, reverse=True)
    return {"conversations": out}



@app.post("/api/conversations")
async def create_conversation(agent: str = DEFAULT_AGENT) -> dict[str, Any]:
    """Allocate a fresh conversation lane id for the \"New conversation\" flow.

    The session itself is lazily created on first /api/chat; here we just mint a
    stable id and register its last-used timestamp so it appears in the list.
    """
    conversation_id = f"paf-{uuid.uuid4().hex[:12]}"
    _sessions_last_used[_session_key(agent, conversation_id)] = _now()
    return {"conversation_id": conversation_id, "agent": agent}


@app.patch("/api/conversations/{conversation_id}")
async def set_conversation_title(
    conversation_id: str, body: dict[str, Any], agent: str = DEFAULT_AGENT
) -> dict[str, Any]:
    """Set (or clear) a conversation's user-facing title, durably.

    Persists to the per-agent ``conversation-titles.json`` sidecar under
    ``data/conversations/<agent>/`` — kept OUT of ``data/memory/`` so titles are
    not coupled to the memory capability — so a rename is exclusive to one
    agent's lane and survives restarts. A non-empty title replaces the
    derived one; an empty string clears it back to the auto-derived title.
    """
    raw = body.get("title")
    if raw is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="title required")
    title = str(raw).strip()

    titles = _load_titles(agent)
    if title:
        titles[conversation_id] = {"title": title, "updated_at": _now()}
    else:
        titles.pop(conversation_id, None)
    _save_titles(agent, titles)

    return {
        "conversation_id": conversation_id,
        "agent": agent,
        "title": title or None,
    }


def _turn_frames(session: Session, prompt: str, conversation_id: str) -> AsyncIterator[str]:
    """Run one turn, forwarding turn events + background-task frames as SSE."""

    async def _inner() -> AsyncIterator[str]:
        turn_q: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()
        task_q: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

        async def _pump_turn() -> None:
            try:
                async for ev in session.stream(prompt, conversation_id=conversation_id):
                    await turn_q.put(_serialize_turn(ev))
            except Exception as exc:  # noqa: BLE001 - surface as an error frame
                log.warning("turn stream ended: %r", exc)
                await turn_q.put(("error", {"detail": str(exc)}))
            finally:
                await turn_q.put(None)  # sentinel: turn stream finished

        # subscribe_tasks fires the callback both synchronously (replay of
        # already-active tasks) and via ensure_future (live events). put_nowait
        # is sync-safe on an asyncio.Queue from the same loop thread. The
        # `dead` flag guards against a stale fire after this generator closes.
        dead = {"v": False}

        def _on_task(ev: Any) -> None:
            if not dead["v"]:
                task_q.put_nowait(_task_frame(ev))

        session.subscribe_tasks(_on_task)

        turn = asyncio.create_task(_pump_turn())
        try:
            # Drive the TURN to completion. The stream ends exactly when the turn
            # ends (clean EOF -> client onDone -> UI shows "done"). Background
            # task frames are forwarded only while the turn is still producing, so
            # a trailing consolidation frame can never keep the connection open or
            # become the terminal event.
            while True:
                item = await turn_q.get()
                if item is None:
                    break  # turn finished -- close the SSE stream now
                yield _frame(*item)
                # Opportunistically flush any pending task frame (non-blocking).
                while not task_q.empty():
                    t = task_q.get_nowait()
                    if t is None:
                        break
                    yield _frame(*t)

            # Turn done: drain any task frames still queued (e.g. a consolidation
            # process that transitioned running->done just as the turn ended) so
            # the UI's process row reflects the last known state instead of being
            # silently dropped. Non-blocking: we only surface frames already
            # produced; we never wait on the background worker itself.
            while not task_q.empty():
                t = task_q.get_nowait()
                if t is None:
                    break
                yield _frame(*t)
        finally:
            # Cooperative teardown: cancel the session's CancellationToken, NOT
            # the turn task directly. turn.cancel() injects asyncio.CancelledError
            # into session.stream()'s generator from a DIFFERENT task/context than
            # the one that opened its OTel root_span, which makes OTel's
            # context.detach() raise `ValueError: <Token> was created in a
            # different Context`, the span is never ended/exported (the "tracing
            # sometimes just stops" bug) and we get `aclose(): already running`.
            # session.cancel() lets stream() end via its own RunCancelled path in
            # the SAME task/context that owns the span -> span ends, trace lands,
            # partial history preserved (commit 24cd3aa). We then drain the turn
            # task so it settles before the generator is torn down.
            dead["v"] = True
            try:
                await session.cancel()
            except Exception:  # noqa: BLE001 - best-effort; never mask teardown
                log.warning("session.cancel() during stream teardown failed", exc_info=True)
            if not turn.done():
                turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)

    return _inner()


@app.post("/api/chat")
async def chat(body: dict[str, Any]) -> StreamingResponse:
    """Open an SSE stream for one turn on a conversation.

    Body: ``{conversation_id, prompt, agent?}``. Returns ``text/event-stream``
    of TurnStreamEvent frames; forwards ``task:`` frames for the background
    timeline while the stream is open.
    """
    conversation_id = body.get("conversation_id")
    prompt = body.get("prompt")
    if not conversation_id or not prompt:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "conversation_id + prompt required")
    session = await _get_session(str(conversation_id), str(body.get("agent", DEFAULT_AGENT)))

    frames = _turn_frames(session, str(prompt), str(conversation_id))
    return StreamingResponse(frames, media_type="text/event-stream")


@app.post("/api/steer")
async def steer(body: dict[str, Any]) -> dict[str, Any]:
    """Inject a message into a running conversation (``Session.enqueue``)."""
    conversation_id = body.get("conversation_id")
    session = await _get_session(str(conversation_id), str(body.get("agent", DEFAULT_AGENT)))
    enq_id = await session.enqueue(
        str(body.get("content", "")), priority=str(body.get("priority", "asap"))
    )
    return {"enqueue_id": enq_id}


@app.post("/api/cancel")
async def cancel(body: dict[str, Any]) -> dict[str, Any]:
    """Stop the running turn on a conversation (``Session.cancel``)."""
    conversation_id = body.get("conversation_id")
    session = await _get_session(str(conversation_id), str(body.get("agent", DEFAULT_AGENT)))
    await session.cancel()
    return {"cancelled": True}


# --- entry point -----------------------------------------------------------
def main(port: int = 8090) -> None:
    """Run the transport standalone (uvicorn). Port 8090 keeps clear of Gradio's :8080."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
