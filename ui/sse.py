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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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
        "http://100.107.176.116:3000",  # Tailscale host (paf-ui via phone/devices)
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict[str, bool]:
    """Liveness probe for the systemd unit / load balancer / UI boot check."""
    return {"ok": True}


def _turn_frames(session: Session, prompt: str, conversation_id: str) -> AsyncIterator[str]:
    """Run one turn, forwarding turn events + background-task frames as SSE."""

    async def _inner() -> AsyncIterator[str]:
        out: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

        async def _pump_turn() -> None:
            try:
                async for ev in session.stream(prompt, conversation_id=conversation_id):
                    await out.put(_serialize_turn(ev))
            except Exception as exc:  # noqa: BLE001 - surface as an error frame
                log.warning("turn stream ended: %r", exc)
                await out.put(("error", {"detail": str(exc)}))

        async def _pump_tasks() -> None:
            queue: asyncio.Queue[Any] = asyncio.Queue()
            # callback is handed to subscribe_tasks, which calls it both synchronously
            # (replay of already-active tasks) and via ensure_future (live events).
            # put_nowait is sync-safe on an asyncio.Queue from the same loop thread, so
            # both paths land the event in the queue without an unawaited-coroutine warning.

            session.subscribe_tasks(queue.put_nowait)
            try:
                while True:
                    got = await queue.get()
                    if got is None:
                        break
                    # subscribe_tasks delivers BackgroundEvent (kind/status/detail/metadata).
                    meta = getattr(got, "metadata", {}) or {}
                    await out.put(
                        (
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
                    )
            except Exception as exc:  # noqa: BLE001
                log.debug("task pump ended: %r", exc)

        turn = asyncio.create_task(_pump_turn())
        tasks = asyncio.create_task(_pump_tasks())
        try:
            while not turn.done():
                event, data = await out.get()
                yield _frame(event, data)
        finally:
            turn.cancel()
            tasks.cancel()
            await asyncio.gather(turn, tasks, return_exceptions=True)

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
