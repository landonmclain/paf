"""paf Web UI — Gradio, one file, mobile-first.

A self-contained web chat UI for the pydantic-agent-framework (paf) that renders
the same live step model as `ui/cli3.py`, but as Gradio ``ChatMessage`` metadata
accordions instead of Rich panels:

    thinking  -> accordion "🧠 Thinking" (collapsible, faint)
    tool call -> accordion "🛠️ <human label>" (args inside)
    response  -> normal assistant markdown message

While a tool runs the accordion shows a live spinner (``status: "pending"``,
open); on ``tool_result`` it flips to ``status: "done"`` and the accordion
closes. The bot function is an **async generator** — it ``await``-iterates paf's
``Session.stream`` and yields the growing message list on every event, which
Gradio consumes for live streaming.

Usage:
    python ui/web.py [agent_name] [port]
    # default agent_name "demo_assistant"; "assistant" (specs/assistant.yaml)
    # for the full profile. conversation_id defaults to web-{agent_name}
    # (single-lane persistence, same as cli3).

The server binds 0.0.0.0:<port> (default 8080) so it's reachable on the LAN and
hence via Tailscale from a phone. Keep it plain HTTP behind Tailscale (TS
provides authn); do not expose it to the public internet.

Needs OPENAI_API_KEY (live model), matching cli3.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import sys
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gradio as gr

from paf.capabilities import PROFILES
from paf.runtime import Session, StreamEvent

DEFAULT_AGENT = "demo_assistant"
DEFAULT_PORT = 8080
# Conversation-lane epoch. The web UI is a single-lane resume (one conversation
# id per agent) — bumping this number starts a BRAND-NEW empty lane, forgetting
# every prior run's restored history for that agent. Handy after crash pollution
# or when you want a clean scrape. Keep the agent name / page title stable; only
# the stored conversation id changes.
CONV_EPOCH = 6

# Page-level CSS. Injected via `launch(css=...)` (Gradio 6 moved the css/theme
# params from Blocks/ChatInterface constructors to launch()). Goals: (1) kill the
# outer chatbot card's border + opaque background so the transcript reads edge-to-
# edge like a native chat instead of a framed box — the biggest space-waster on
# mobile is that 1px border + dark card inset; (2) trim the page's horizontal
# padding on small screens while keeping the text input a distinct rounded field.
_CSS = """
/* Page container: transparent, edge-to-edge. The app holds the chatbot card,
   so the outer chrome shouldn't add its own background or centering column. */
.gradio-container {
    background: transparent !important;
    max-width: none !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}
/* The page wrapper that pads the whole app. Gradio's default
   `padding: var(--size-4) var(--size-8)` resolves to ~32px columns on mobile —
   the space-waster. Collapse the horizontal gutters; keep a little vertical. */
.app.svelte-99kmwu,
.main.app {
    padding: 20px 2px !important;
}
/* The component card that wraps the chatbot — kill its border, shadow and
   opaque fill so there's no visible outer box; messages float on the page.
   (Gradio renders the chatbot's outer card with classes `block flex
   auto-margin`; the text input lives in a *different* wrapper so it keeps its
   distinct rounded field.) */
.gradio-container .block.flex.auto-margin,
#chatbot {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
/* Make the chat fill the viewport height instead of the fixed `height=640`
   pixel cap (which leaves the chat ~half-filled on desktop). Turn the page
   into a column flex so the chatbot card can grow to use the free space,
   and drop the fixed height on the chatbot's internal scroll area. */
.gradio-container {
    display: flex !important;
    flex-direction: column !important;
}
.gradio-container .block.flex.auto-margin {
    flex: 1 1 auto !important;
    min-height: 0 !important;
}
/* The explicit outer `gr.Blocks(fill_width=True)` wrapper (added for
   `demo.load` on page refresh) renders as a `.column svelte-siq5d6` that ships
   with `flex: 0 1 auto` — it does NOT grow, so it caps the chatbot card at its
   fixed `height=640` and the chat stops filling the viewport. Force the column
   wrapper(s) to grow too so the card can stretch (regression 2026-08-21, seen
   once the chat was wrapped in gr.Blocks). Hash class is stable for Gradio 6.25. */
.gradio-container .column.svelte-siq5d6 {
    flex: 1 1 auto !important;
    min-height: 0 !important;
}
#chatbot {
    height: 100% !important;
}
/* Mobile doesn't zoom into a focused input only when its font-size is >= 16px.
   Gradio's text box ships at 14px, which triggers the classic focus-zoom that
   forces the user to pinch back out after typing. Bump the input (and, for
   safety, the wrapper) to 16px so iOS/Android leave the viewport alone. */
textarea,
input[type="text"],
input[type="search"] {
    font-size: 16px !important;
}
/* Hide Gradio's default footer bar ("Runs · Use via API · Built with Gradio").
   Selection is by the semantic `<footer>` tag (stable across Gradio versions —
   the hashed class differs per build), so it survives upgrades. Removes the
   per-browser run-history / API links and reclaims the bottom strip, which is
   wasted space on mobile (2026-08-21). */
footer {
    display: none !important;
}
"""

# ---------------------------------------------------------------------------
# Turn model (same as cli3) — fold StreamEvents into ordered blocks.
# ---------------------------------------------------------------------------


@dataclass
class ThinkingBlock:
    text: str
    started: float = field(default_factory=time.perf_counter)
    # Wall-clock duration in seconds once the block is DONE; None while pending.
    elapsed: float | None = None


@dataclass
class ToolBlock:
    name: str
    args: str
    status: str = "\u2026"  # …
    started: float = field(default_factory=time.perf_counter)
    # Wall-clock duration in seconds once the block is DONE; None while pending.
    elapsed: float | None = None


@dataclass
class TurnStream:
    """Accumulates the live turn. ``blocks`` preserves interleaved order so each
    model step renders as its own accordion (thinking -> tool -> thinking ->
    tool -> response). ``response`` stays separate so the answer renders last."""

    blocks: list = field(default_factory=list)  # ThinkingBlock | ToolBlock
    response: list[str] = field(default_factory=list)

    def add_thinking(self, text: str) -> None:
        if self.blocks and isinstance(self.blocks[-1], ThinkingBlock):
            self.blocks[-1].text += text
        else:
            self.blocks.append(ThinkingBlock(text))

    def add_tool(self, name: str, args: str) -> None:
        # A tool supersedes any open trailing thinking box — that thinking block
        # ends here, so stamp its wall-clock duration before starting the tool.
        if self.blocks and isinstance(self.blocks[-1], ThinkingBlock) and self.blocks[-1].elapsed is None:
            self.blocks[-1].elapsed = time.perf_counter() - self.blocks[-1].started
        self.blocks.append(ToolBlock(name, args))

    def finish_latest_tool(self) -> None:
        for block in reversed(self.blocks):
            if isinstance(block, ToolBlock) and block.status != "done":
                block.status = "done"
                block.elapsed = time.perf_counter() - block.started
                return

    def finish_open_blocks(self, now: float | None = None) -> None:
        """Close the trailing blocks of a finished turn and stamp their duration.

        Called once at turn end. The only block that can still be open is the
        trailing thinking box (no explicit finish event) or an in-flight tool;
        intermediate thinking blocks were already stamped in ``add_tool``. Leaves
        already-done blocks (with durations) untouched so earlier sub-second
        timings are not silently overwritten by the whole-turn wall clock.
        """
        now = now if now is not None else time.perf_counter()
        for block in self.blocks:
            if isinstance(block, ToolBlock) and block.status != "done":
                block.status = "done"
            if block.elapsed is None:
                block.elapsed = now - block.started


def _safe(s: str) -> str:
    """Make a model-produced string safe under utf-8 (callers may not set it).
    FunctionToolCallEvent.args can carry lone UTF-16 surrogates while streaming;
    drop unencodable chars rather than raise, mirroring cli3."""
    return s.encode("utf-8", errors="replace").decode("utf-8")


def render(event: StreamEvent, out: TurnStream) -> None:
    """Fold one StreamEvent into ``out``."""
    if event.kind == "thinking" and event.text:
        out.add_thinking(event.text)
    elif event.kind == "text" and event.text:
        out.response.append(event.text)
    elif event.kind == "tool_call":
        out.add_tool(event.tool_name or "<unknown>", _safe(event.args))
    elif event.kind == "tool_result":
        out.finish_latest_tool()


# ---------------------------------------------------------------------------
# Human-readable tool labels (small dict keyed by tool name — conceptually lifted
# from coding_agent's web_app formatter tables, kept minimal).
# ---------------------------------------------------------------------------


def _basename(label: str) -> str:
    try:
        return str(Path(label).name)
    except Exception:  # noqa: BLE001
        return label


def _clip(s: str, limit: int = 120) -> str:
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: max(0, limit - 1)] + "\u2026"


def _title_args(raw: str) -> str:
    """Pick a short, human-friendly value for the accordion *title* from the
    raw tool args string.

    Tool args arrive as JSON (e.g. ``{"path": "_consolidate.py"}``). Showing the
    whole JSON is noisy (``read_file: {"path": "_consolidate.py"}``); instead we
    pull the first meaningful string value so the title reads ``read_file:
    _consolidate.py``. Non-JSON args (bare strings, ``{}``) pass through.
    """
    try:
        obj = json.loads(raw)
    except Exception:  # noqa: BLE001
        return _clip(raw)
    if isinstance(obj, str):
        return _basename(_clip(obj))
    if isinstance(obj, dict):
        # Prefer a path-ish first value; fall back to the first string value.
        for v in obj.values():
            if isinstance(v, str) and v:
                return _basename(_clip(v))
        # No string value: show the key list compactly for context.
        if obj:
            return _clip(",".join(obj.keys()))
        return ""
    if isinstance(obj, list):
        strs = [v for v in obj if isinstance(v, str) and v]
        if strs:
            return _basename(_clip(strs[0]))
        return _clip(raw)
    return _clip(raw)


# name -> (accordion-title-fmt {a}, open-body-fmt {a})
_TOOL_LABELS: dict[str, tuple[str, str]] = {
    "read_file": ("read_file: {a}", "Reads a file (path: **{a}**)."),
    "write_file": ("write_file: {a}", "Writes content to **{a}**."),
    "edit_file": ("edit_file: {a}", "Edits **{a}**."),
    "list_directory": ("list_directory: {a}", "Lists **{a}**."),
    "search_files": ("search_files: {a}", "Searches files. Pattern: `{a}`."),
    "find_files": ("find_files: {a}", "Finds files: `{a}`."),
    "run_command": ("run_command: {a}", "Runs shell command: `{a}`."),
    "git_status": ("git_status", "Shows git working-tree status."),
    "git_changed_files": ("git_changed_files", "Lists git-changed files."),
    "git_diff": ("git_diff", "Shows the git diff."),
    "git_checkpoint": ("git_checkpoint: {a}", "Saves a git checkpoint for `{a}`."),
    "git_restore_paths": ("git_restore_paths: {a}", "Restores paths: `{a}`."),
    "web_search": ("web_search: {a}", "Searches the web for: {a}."),
    "list_skills": ("list_skills", "Lists available skills."),
    "load_skill": ("load_skill: {a}", "Loads skill **{a}**."),
    "search_conversation_history": ("search_history: {a}", "Searches past conversations: {a}."),
    "read_turn": ("read_turn: {a}", "Reads a past turn: {a}."),
    "memory": ("memory: {a}", "Durable memory: {a}."),
}


def human_tool_label(name: str, args: str) -> tuple[str, str]:
    """Return (accordion title, open-body text) for a tool call.

    Falls back to raw ``name(args)`` for unknown tools so they still render."""
    entry = _TOOL_LABELS.get(name)
    if entry:
        title_fmt, body_fmt = entry
        a = _title_args(args)  # short, JSON-unwrapped value for the title
        # body keeps full args
        return title_fmt.format(a=a), body_fmt.format(a=args)
    return f"{name}({_clip(args)})", args


# ---------------------------------------------------------------------------
# Session lifetime — build once per agent, reuse across turns (single-lane).
# ---------------------------------------------------------------------------

_sessions: dict[str, Session] = {}


def get_session(agent_name: str, profile_name: str | None = None) -> Session:
    """Build (or reuse) the single-lane Session for ``agent_name``.

    cli3 holds a ``with Session(...)`` open for the process; here we cache the
    session per agent and seed history on first use (mirroring the startup
    banner). Sessions flush on process exit via the atexit hook."""
    session = _sessions.get(agent_name)
    if session is None:
        if profile_name is None:
            profile_name = "demo" if agent_name == "demo_assistant" else "assistant"
        profile = PROFILES.get(profile_name)
        session = Session(
            agent_name,
            profile=profile,
            conversation_id=f"web-{agent_name}-e{CONV_EPOCH}",
            resume_window=15,
        )
        _sessions[agent_name] = session
    return session


def close_all() -> None:
    """Best-effort flush/close of cached sessions on shutdown."""
    for s in _sessions.values():
        try:
            s.close()  # sync flush (spans flushed); mirrors `with Session(...)`
        except Exception:  # noqa: BLE001
            pass
    _sessions.clear()


# ---------------------------------------------------------------------------
# Gradio render — fold a TurnStream into ChatMessages with metadata accordions.
# ---------------------------------------------------------------------------


def _elapsed_label(started: float, now: float | None = None) -> str:
    """Human label for seconds since ``started``, e.g. ``· 14s`` (or ``· 1m 03s``).

    Rendered onto the *open* (pending) accordion title so it's clear the spinner
    is still working and for how long, instead of silently frozen.
    """
    secs = int((now if now is not None else time.perf_counter()) - started)
    if secs >= 60:
        return f"· {secs // 60}m {secs % 60:02d}s"
    return f"· {secs}s"


def _done_label(elapsed: float) -> str:
    """Compact duration label for a COMPLETED block, e.g. ``· 5s`` / ``· 1.4s``.

    Unlike :func:`_elapsed_label` (whole seconds, for the live spinner) this keeps
    one decimal for sub-second tools so a fast read_file shows ``· 0.3s`` instead
    of a rounded ``· 0s``.
    """
    if elapsed >= 60:
        return f"· {int(elapsed) // 60}m {int(elapsed) % 60:02d}s"
    if elapsed >= 10:  # whole seconds above 10s reads cleaner
        return f"· {elapsed:.0f}s"
    return f"· {elapsed:.1f}s"


def _thinking_message(block: ThinkingBlock, done: bool, now: float | None = None) -> gr.ChatMessage:
    title = "🧠 Thinking"
    if not done:
        title += " " + _elapsed_label(block.started, now)
    elif block.elapsed is not None:
        title += " " + _done_label(block.elapsed)
    return gr.ChatMessage(
        role="assistant",
        content=block.text or "…",
        metadata={
            "title": title,
            "status": "done" if done else "pending",
        },
    )


def _tool_message(block: ToolBlock, done: bool, now: float | None = None) -> gr.ChatMessage:
    title, body = human_tool_label(block.name, block.args)
    if not done:
        title += " " + _elapsed_label(block.started, now)
    elif block.elapsed is not None:
        title += " " + _done_label(block.elapsed)
    return gr.ChatMessage(
        role="assistant",
        content=body,
        metadata={
            "title": f"🛠️ {title}",
            "status": "done" if done else "pending",
        },
    )


def _response_message(out: TurnStream) -> gr.ChatMessage:
    text = "".join(out.response)
    return gr.ChatMessage(role="assistant", content=text or "…")


def turn_to_messages(out: TurnStream) -> list[gr.ChatMessage]:
    """Render the accumulated turn. Done thinking/tool accordions collapse;
    pending ones carry a live spinner. The response renders last."""
    messages: list[gr.ChatMessage] = []
    for block in out.blocks:
        if isinstance(block, ThinkingBlock):
            messages.append(_thinking_message(block, done=True))
        elif isinstance(block, ToolBlock):
            messages.append(_tool_message(block, done=(block.status == "done")))
    messages.append(_response_message(out))
    return messages


def _boot_banner(session: Session) -> list[gr.ChatMessage]:
    cid = session.conversation_id or "?"
    return [
        gr.ChatMessage(
            role="assistant",
            content=f"(new conversation: `{cid}`)",
            metadata={"title": "ℹ️ session", "status": "done"},
        )
    ]


# ---------------------------------------------------------------------------
# Bot function — async generator, one turn, yields growing history live.
# ---------------------------------------------------------------------------


async def _seed_once(session: Session) -> None:
    """Seed durable history on first run (single-lane resume), like cli3."""
    if not getattr(session, "_web_booted", False):
        session._web_booted = True
        try:
            await session.restore()
        except Exception:  # noqa: BLE001 - persistence is optional sugar
            pass


# Cache of the rendered restored transcript per session (built once after
# `restore()`), so `event_messages` can seed the web chat with prior turns.
_session_history: dict[str, list[dict]] = {}


def _render_session_history(session: Session) -> list[dict]:
    """Convert restored ``Session.history`` into user/assistant message dicts.

    The store's raw history is pydantic-ai's internal message list — each user
    turn expands into multiple ``ModelRequest`` / ``ModelResponse`` messages
    (thinking, tool calls, mid-stream draft text, then the final response). We
    fold those back into discrete turns using the SAME block model + renderers
    as the live path (``TurnStream`` / ``turn_to_messages``), so restored
    history looks identical to a freshly-streamed turn:

        user  ->  "reply pong"
        🧠 Thinking      (accordion, collapsed)
        🛠️ read_file…   (accordion, collapsed)
        assistant        "pong"

    Folding rules per message:
      * ``UserPromptPart``        -> starts the next turn (emits its user bubble)
      * ``ThinkingPart``          -> accumulated into the turn's thinking block
      * ``TextPart`` alongside a ``ToolCallPart`` in the SAME ModelResponse is
        mid-stream draft prose (the live stream's ``text`` only fires after the
        tool loop ends) -> treated as thinking, exactly like a live turn
      * ``ToolCallPart``          -> a tool accordion in the turn
      * ``ToolReturnPart`` /
        ``RetryPromptPart``       -> marks the most recent tool accordion done
      * final ``TextPart`` on a ModelResponse with no tools -> the response
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        RetryPromptPart,
        TextPart,
        ThinkingPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    out: list[dict] = []

    # Split the flat message list into (user_prompt, TurnStream) turns.
    turns: list[tuple[str, TurnStream]] = []
    cur_user: str | None = None
    cur: TurnStream | None = None

    def _flush() -> None:
        nonlocal cur_user, cur
        if cur_user is not None and cur is not None:
            # Force-close any trailing open tool/thinking block and mark it done,
            # mirroring the live path's turn-end `finish_open_blocks()` (see
            # event_messages). Restored message folds may not see a closing
            # ToolReturnPart for a turn's last tool, which would otherwise leave
            # that block `status="…"` -> a spinner on a PAST turn. Stamping it
            # done here keeps restored turns spinner-free.
            cur.finish_open_blocks()
            turns.append((cur_user, cur))
        cur_user, cur = None, None

    for msg in session.history or []:
        parts = getattr(msg, "parts", []) or []
        if isinstance(msg, ModelRequest):
            # A user request terminates any open assistant turn and (re)opens one.
            # A single request may carry SEVERAL UserPromptParts — the real user
            # prompt plus injected context (e.g. the durable <memory> block, whose
            # content is a non-str list). They are ONE logical user turn: emit a
            # single user bubble using the first real string prompt, and skip the
            # injected context parts.
            user_parts = [p for p in parts if isinstance(p, UserPromptPart)]
            if user_parts:
                _flush()
                cur_user = next(
                    (p.content for p in user_parts if isinstance(p.content, str)),
                    "",
                )
                cur = TurnStream()
            # Tool results within a request close the tool accordion they answer.
            for part in parts:
                if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                    if cur is not None:
                        cur.finish_latest_tool()
        elif isinstance(msg, ModelResponse):
            if cur is None:
                # Orphan response with no preceding user prompt; start a turn.
                cur = TurnStream()
                cur_user = cur_user or ""
            has_tool = any(isinstance(p, ToolCallPart) for p in parts)
            for part in parts:
                if isinstance(part, ThinkingPart):
                    cur.add_thinking(_safe(getattr(part, "content", "") or ""))
                elif isinstance(part, TextPart):
                    text = _safe(getattr(part, "content", "") or "")
                    if has_tool:
                        # Mid-stream draft prose while the model calls tools.
                        cur.add_thinking(text)
                    else:
                        cur.response.append(text)
                elif isinstance(part, ToolCallPart):
                    cur.add_tool(
                        getattr(part, "tool_name", "<unknown>") or "<unknown>",
                        _safe(str(getattr(part, "args", "") or "")),
                    )
    _flush()

    # Render each turn: user bubble, then folded thinking/tool/response blocks
    # (identical look to the live path). Skip turns that produced nothing (e.g.
    # a user prompt whose run was interrupted before any response text or tool
    # call settled) so we don't emit stray "…" placeholder bubbles.
    for user, turn in turns:
        if not turn.blocks and not turn.response:
            continue
        if user:
            out.append({"role": "user", "content": user})
        for m in turn_to_messages(turn):
            out.append(_to_dict(m))
    return out


def session_history_msgs(session: Session) -> list[dict]:
    """Rendered prior turns for ``session`` (cached; recomputed once after boot)."""
    cached = _session_history.get(session.conversation_id or session.agent_name)
    if cached is not None:
        return cached
    rendered = _render_session_history(session)
    _session_history[session.conversation_id or session.agent_name] = rendered
    return rendered


async def _load_history(agent_name: str) -> list[dict]:
    """Freshly-rendered restored history for the chatbot on PAGE LOAD/REFRESH.

    ``session_history_msgs``/``restore()`` are one-shot: they run once at server
    startup and cache, so a browser refresh would otherwise keep showing the
    startup-time transcript even though newer turns were persisted meanwhile
    (the model had them in context; the UI didn't). This re-reads the store via
    ``Session.refresh()`` and re-renders, so a refresh always shows the latest
    persisted turns.
    """
    session = get_session(agent_name)
    _session_history.pop(session.conversation_id or session.agent_name, None)
    try:
        await session.refresh(conversation_id=session.conversation_id)
    except Exception:  # noqa: BLE001 - resume is optional sugar
        pass
    return _render_session_history(session)


def _to_dict(m: Any) -> dict:
    """Convert a gr.ChatMessage (or pass through a plain dict) to the message
    dict shape ChatInterface uses with ``type="messages"``."""
    if isinstance(m, dict):
        return dict(m)
    md = getattr(m, "metadata", None) or None
    d: dict = {"role": getattr(m, "role", "assistant"), "content": getattr(m, "content", "")}
    if md:
        d["metadata"] = dict(md)
    return d


def _assistant_messages(out: TurnStream, now: float | None = None) -> list[dict]:
    """Render the current assistant transcript as message dicts.

    One accordion per block, in stream order, plus the response last. The single
    currently-*open* block — the trailing thinking block, or the running tool —
    is rendered with ``status: "pending"`` so it shows a live spinner and grows
    in place as tokens arrive, plus an elapsed-time label (``· Ns``) that
    advances as the turn streams. Finished blocks render ``"done"`` and collapse.
    """
    msgs: list[dict] = []
    n = len(out.blocks)
    for i, block in enumerate(out.blocks):
        is_last = i == n - 1
        if isinstance(block, ThinkingBlock):
            if is_last and not out.response:
                msgs.append(_to_dict(_thinking_message(block, done=False, now=now)))  # open, live
            else:
                msgs.append(_to_dict(_thinking_message(block, done=True)))
        elif isinstance(block, ToolBlock):
            # A tool is open (pending) only while it's the last block and not done.
            running = is_last and block.status != "done"
            msgs.append(_to_dict(_tool_message(block, done=not running, now=now)))
    if out.response:
        msgs.append(_to_dict(_response_message(out)))
    return msgs


async def event_messages(
    agent_name: str, prompt: str, history: list[Any]
) -> AsyncGenerator[list[Any], None]:
    """One turn: await-iterate paf StreamEvents, yield THIS turn's assistant messages.

    Gradio's ChatInterface owns the full-transcript accumulation: before invoking
    us it appends the user message to `history`, and it appends whatever we yield
    as the assistant turn on every frame. So we must NOT re-embed prior turns or
    the user prompt in what we return — that would duplicate the whole
    conversation inside every response bubble (the earlier "shows entire history
    again" bug). We return only the current turn's assistant transcript; paf's
    `Session` maintains its own durable context, so the model still sees the full
    prior conversation regardless.

    On every event we rebuild the assistant transcript fresh and yield it, so the
    currently-open accordion (the live thinking box, or the running tool) is
    *replaced* in place as it grows — never duplicated. Behind the scenes paf's
    TurnStream coalesces consecutive thinking tokens into a single block, so
    there's exactly one live thinking box.
    """
    session = get_session(agent_name)
    await _seed_once(session)

    out = TurnStream()
    # First frame: nothing (no-op append). Gives Gradio a frame to render the
    # user bubble it already appended, before the model streams anything.
    yield []

    t0 = time.perf_counter()
    async for event in session.stream(prompt, conversation_id=session.conversation_id):
        render(event, out)
        # Rebuild the current turn's assistant transcript fresh each frame
        # (never accumulate via slicing), so Gradio replaces the open accordion
        # in place instead of growing new bubbles. Pass a fresh `now` so the
        # open spinner's elapsed label tracks real wall-clock time.
        yield _assistant_messages(out, now=time.perf_counter())

    # Final frame: force-close any open trailing block, mark everything done,
    # stamp the trailing thinking block's duration (it has no explicit finish
    # event — its runtime ends at turn end; intermediate thinking blocks were
    # already stamped when a tool superseded them), and append a compact
    # total-turn-time footer.
    now_end = time.perf_counter()
    out.finish_open_blocks(now_end)
    msgs = _assistant_messages(out, now=now_end)
    for m in msgs:
        md = m.get("metadata")
        if md:
            md["status"] = "done"
    total_s = now_end - t0
    msgs.append(
        _to_dict(
            gr.ChatMessage(
                role="assistant",
                content="",
                metadata={
                    "title": f"⏱ turn · {total_s:.1f}s",
                    "status": "done",
                },
            )
        )
    )
    yield msgs





# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    argv = sys.argv[1:]
    agent_name = argv[0] if argv else DEFAULT_AGENT
    profile_name = None
    port = DEFAULT_PORT
    if len(argv) > 1:
        if argv[1].isdigit():
            port = int(argv[1])
        else:
            profile_name = argv[1]

    if profile_name is not None and profile_name not in PROFILES:
        print(f"unknown profile {profile_name!r}; known: {sorted(PROFILES)}")
        return 2

    # Build the session up front so a bad profile/agent fails fast at launch,
    # and surface a boot banner in the transcript.
    session = get_session(agent_name, profile_name)
    banner = _boot_banner(session)

    if os.environ.get("WEB_PREVIEW") == "1":
        # Synthetic render check (no live model): dump one turn's message shape.
        out = TurnStream()
        out.add_thinking("thinking about the plan")
        out.add_tool("read_file", "/tmp/example.md")
        out.finish_latest_tool()
        out.add_tool("run_command", "git status")
        out.response.append("**done.** rendered a synthetic turn.")
        for m in banner + turn_to_messages(out):
            print("  ", m)
        print("PREVIEW_OK")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Provide it to call the live model:")
        print("  export OPENAI_API_KEY=sk-...")
        return 2

    import atexit

    atexit.register(close_all)

    # Load durable history up front so the Chatbot opens with the prior
    # conversation visible (single-lane resume), not an empty box.
    try:
        asyncio.run(_seed_once(session))
    except Exception:  # noqa: BLE001 - persistence is optional sugar
        pass
    initial_history: list[dict] = session_history_msgs(session)

    chatbot = gr.Chatbot(height=640, layout="bubble", value=initial_history)
    # ChatInterface is a Blocks subclass but only exposes `.load()` once it's
    # mounted inside an open Blocks context — so wrap it explicitly. That lets
    # us attach a page-load handler that re-renders restored history on every
    # refresh (see _load_history), not just at server startup. Keep `demo` as
    # the enclosing Blocks (do NOT rebind it to the ChatInterface) so the
    # `.load` handler and the ChatInterface mount on the same served graph.
    with gr.Blocks(fill_width=True) as demo:
        # Re-render restored history on every page load/refresh: a refresh must
        # show turns persisted since startup, which startup-time `initial_history`
        # alone would miss.
        demo.load(
            fn=functools.partial(_load_history, agent_name),
            inputs=None,
            outputs=chatbot,
        )
        gr.ChatInterface(
            # functools.partial (NOT a lambda) so Gradio introspects fn as an async
            # generator and uses the streaming `_stream_fn` path. A lambda defeats
            # `inspect.isasyncgenfunction` and Gradio would treat it as a plain fn,
            # returning the generator object instead of streaming it.
            fn=functools.partial(event_messages, agent_name),
            chatbot=chatbot,
            fill_width=True,
        )
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        css=_CSS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
