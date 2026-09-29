"""Simple inline paf chat — streams into normal terminal scrollback.

No full-screen UI and no redrawn frames: the turn prints top-to-bottom as it
streams (dim thinking, one line per tool call, then the answer), so the
terminal's own scrollback, search, and copy/paste all work. The answer streams
raw token-by-token and each finished markdown block is swapped in place for
its rendered form (``MarkdownStream``; ``/markdown`` toggles). Input is a
prompt_toolkit line editor with a bottom toolbar showing background processes
and plan progress.

Turns run in-process as Session-owned turns (``start_turn`` + ``subscribe``),
the same primitive ``ui/sse.py`` uses, inside ONE long-lived event loop — so
background shell/subagent completions are delivered via ``Session.notify`` and
their follow-up turn streams here even while you sit at the prompt.

Usage:
    python ui/chat.py [agent] [--profile P] [--id CID | --new] [--window N]

Keys: Enter sends · Alt+Enter (or Esc, Enter) inserts a newline · Ctrl-C
during a turn cancels it (twice = hard stop) · Ctrl-D exits. /help for commands.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import signal
import sys
import uuid
from pathlib import Path
from typing import Any, ClassVar

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from rich.cells import cell_len
from rich.console import Console, JustifyMethod
from rich.markdown import Heading, Markdown, MarkdownElement

from paf.background_processes._events import get_agent_emitter
from paf.capabilities import PROFILES
from paf.capabilities.planning import FilePlanStore, PlanItem
from paf.capabilities.planning._events import PlanEventType
from paf.capabilities.planning._registry import get_plan_emitter
from paf.config import PROJECT_ROOT
from paf.runtime import Session, StreamEvent, ToolError, ToolRunning, TurnHandle

DEFAULT_AGENT = "demo_assistant"
_DATA = PROJECT_ROOT / "data"
_DATE_STAMP = re.compile(r"^\[date: [^\]]*\]\s*")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ACTIVE = {"queued", "running", "processing"}
_COORDINATOR_JOBS = {"consolidation", "review", "thread_summarize"}
_NOTIFY = object()  # prompt result: a background notify started a turn

# --- terminal styling -------------------------------------------------------

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _sgr(code: str) -> str:
    return f"\x1b[{code}m" if _COLOR else ""


DIM, RED, GREEN, YELLOW, CYAN, MAGENTA, RESET = (
    _sgr(c) for c in ("2", "31", "32", "33", "36", "35", "0")
)


def _out(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


def _width() -> int:
    return shutil.get_terminal_size((100, 24)).columns


def _clip(text: str, limit: int) -> str:
    """Collapse whitespace to one line and clip to ``limit`` columns."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _safe(s: str) -> str:
    """Drop lone surrogates a partially-decoded stream can carry."""
    return s.encode("utf-8", errors="replace").decode("utf-8")


# --- markdown ------------------------------------------------------------------


class _LeftHeading(Heading):
    LEVEL_ALIGN: ClassVar[dict[str, JustifyMethod]] = {
        **Heading.LEVEL_ALIGN,
        "h1": "left",
    }


class _Markdown(Markdown):
    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "heading_open": _LeftHeading,
    }


def render_markdown(text: str) -> str:
    """Render one finished markdown block to ANSI text (Rich as a formatter only)."""
    console = Console(
        width=_width(), force_terminal=_COLOR, no_color=not _COLOR, highlight=False
    )
    with console.capture() as cap:
        console.print(_Markdown(text, code_theme="ansi_dark", hyperlinks=False))
    # Rich pads lists/code/tables with blank rows; blocks are spaced by the
    # stream (one blank line between), so trim blank edge rows.
    lines = cap.get().split("\n")
    blank = [not _ANSI.sub("", ln).strip() for ln in lines]
    if all(blank):
        return ""
    first = blank.index(False)
    last = len(blank) - blank[::-1].index(False)
    return "\n".join(lines[first:last]) + "\n"


def _block_end(text: str) -> int | None:
    """Index where the first finished markdown block ends, or None if still open.

    A block ends at a blank line or a closing code fence; text before an opening
    fence is its own block. Blank lines inside a fence don't end it. Only
    complete lines count, so a half-streamed line never splits a block.
    """
    fence: str | None = None
    pos = 0
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n"):
            return None
        s = line.strip()
        if fence is not None:
            if s.startswith(fence):
                return pos + len(line)
        elif s.startswith(("```", "~~~")):
            if pos:
                return pos
            fence = s[:3]
        elif not s and pos:
            return pos
        pos += len(line)
    return None


def _rows(text: str, width: int) -> int:
    """Terminal rows ``text`` occupies above the cursor (which sits at its end)."""
    lines = text.split("\n")
    up = sum(max(1, -(-cell_len(ln) // width)) for ln in lines[:-1])
    return up + max(0, -(-cell_len(lines[-1]) // width) - 1)


class MarkdownStream:
    """Stream answer text raw, then swap each finished block for its rendering.

    Tokens print as they arrive. When a block completes, the raw rows are
    erased (cursor-up + clear-to-end) and the rendered block is printed in
    their place — so output stays in normal scrollback with no redraw loop.
    A block taller than the screen can't be erased (its top has scrolled
    off), so it stays raw. With ``enabled=False`` (or no tty) text passes
    through untouched.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and _COLOR
        self.tail = ""  # text of the unfinished block(s)
        self.shown = ""  # prefix of ``tail`` already printed raw

    @property
    def col0(self) -> bool:
        return not self.shown or self.shown.endswith("\n")

    def feed(self, text: str) -> None:
        text = text.replace("\t", "    ")
        if not self.enabled:
            self.shown += text
            _out(text)
            return
        self.tail += text
        if not self.shown:
            self.tail = self.tail.lstrip("\n")
        while (cut := _block_end(self.tail)) is not None:
            block, self.tail = self.tail[:cut], self.tail[cut:].lstrip("\n")
            self._emit(block, last=False)
        _out(self.tail[len(self.shown) :])
        self.shown = self.tail

    def finish(self) -> None:
        if self.enabled and self.tail.strip():
            self._emit(self.tail, last=True)
        elif not self.col0:
            _out("\n")
        self.tail = self.shown = ""

    def _emit(self, block: str, *, last: bool) -> None:
        up = _rows(self.shown, _width()) if self.shown else 0
        if up >= shutil.get_terminal_size((100, 24)).lines - 1:
            # Its top has scrolled off: keep the raw text, just finish printing it.
            rest = block[len(self.shown) :]
            _out(rest)
            if not (rest or self.shown).endswith("\n"):
                _out("\n")
        else:
            if self.shown:
                _out("\r" + (f"\x1b[{up}A" if up else "") + "\x1b[J")
            _out(render_markdown(block.strip("\n")))
        if not last:
            _out("\n")
        self.shown = ""


# --- turn rendering ----------------------------------------------------------


class TurnPrinter:
    """Folds one turn's events straight onto stdout, in stream order.

    A tool call prints ``● name(args)`` and leaves the line open; if its result
    arrives before anything else is printed, ``✓``/``✗`` lands on the same
    line. Parallel calls (line already closed) only print a line on error.
    Results carry no step id, so a ``tool_result`` is held until the next event
    shows whether a ``ToolError`` for it follows.
    """

    def __init__(self, show_thinking: bool, markdown: bool) -> None:
        self.show_thinking = show_thinking
        self.md = MarkdownStream(markdown)
        self.mode: str | None = None  # "thinking" | "text" | "tool"
        self.col0 = True  # cursor at column 0
        self.open_step: str | None = None  # tool whose line is still open
        self.pending: dict[str, str] = {}  # step_id -> tool name (awaiting result)
        self.last_call: str | None = None  # tool name awaiting its ToolRunning
        self.resolving: str | None = None  # step whose result arrived, outcome unknown

    def _write(self, s: str) -> None:
        if s:
            _out(s)
            self.col0 = s.endswith("\n")

    def _newline(self) -> None:
        if self.mode == "text":
            self.md.finish()  # render the answer's last block
            self.mode, self.col0 = "text-done", True
        if self.open_step is not None:
            self.open_step = None
        if not self.col0:
            self._write("\n")

    def _settle(self, step: str, error: str | None) -> None:
        name = self.pending.pop(step, "tool")
        if self.open_step == step:
            self.open_step = None
            if error is None:
                self._write(f" {GREEN}✓{RESET}\n")
            else:
                self._write(f" {RED}✗ {_clip(error, _width() - 8)}{RESET}\n")
        elif error is not None:
            self._newline()
            self._write(
                f"  {RED}✗ {name}: {_clip(error, _width() - 8 - len(name))}{RESET}\n"
            )

    def _flush_result(self, error: str | None = None) -> None:
        if self.resolving is not None:
            step, self.resolving = self.resolving, None
            self._settle(step, error)

    def feed(self, ev: Any) -> None:
        if isinstance(ev, ToolError):
            if self.resolving == ev.step_id:
                self._flush_result(ev.error)
            elif ev.step_id in self.pending:
                self._settle(ev.step_id, ev.error)
            return
        self._flush_result()
        if isinstance(ev, ToolRunning):
            if self.last_call is not None:
                self.pending[ev.step_id] = self.last_call
                self.open_step = ev.step_id if not self.col0 else None
                self.last_call = None
            return
        if not isinstance(ev, StreamEvent):
            return
        if ev.kind == "thinking" and ev.text:
            if not self.show_thinking:
                return
            if self.mode != "thinking":
                self._newline()
                self._write(f"{CYAN}thinking:{RESET} ")
                self.mode = "thinking"
            self._write(f"{DIM}{_safe(ev.text)}{RESET}")
        elif ev.kind == "text" and ev.text:
            if self.mode != "text":
                self._newline()
                if self.mode is not None:
                    self._write("\n")
                self.mode = "text"
            self.md.feed(_safe(ev.text))
            self.col0 = self.md.col0
        elif ev.kind == "tool_call":
            self._newline()
            name = ev.tool_name or "?"
            args = _clip(_safe(ev.args), max(10, _width() - len(name) - 10))
            self._write(f"  {YELLOW}●{RESET} {name}{DIM}({args}){RESET}")
            self.mode = "tool"
            self.last_call = name
        elif ev.kind == "tool_result":
            step = next((s for s, n in self.pending.items() if n == ev.tool_name), None)
            self.resolving = step

    def finish(self) -> None:
        self._flush_result()
        self._newline()


# --- history rendering --------------------------------------------------------


def _user_text(msg: Any) -> str | None:
    if not isinstance(msg, ModelRequest):
        return None
    for part in msg.parts:
        if isinstance(part, UserPromptPart) and isinstance(part.content, str):
            return _DATE_STAMP.sub("", part.content)
    return None


def print_history(history: list[Any], turns: int | None = None) -> None:
    """Print prior turns compactly: prompt, tool one-liners, answer text."""
    starts = [i for i, m in enumerate(history) if _user_text(m) is not None]
    if turns is not None and starts:
        history = history[starts[max(0, len(starts) - turns)] :]
    for msg in history:
        text = _user_text(msg)
        if text is not None:
            _out(f"\n{MAGENTA}›{RESET} {text}\n")
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    args = _clip(
                        str(part.args or ""), _width() - len(part.tool_name) - 10
                    )
                    _out(f"  {DIM}● {part.tool_name}({args}){RESET}\n")
                elif isinstance(part, TextPart) and part.content.strip():
                    _out(f"\n{render_markdown(_safe(part.content.strip()))}")


# --- the app ------------------------------------------------------------------


def _notify_message(run_id: str, detail: str) -> str:
    """Same wording ui/sse.py delivers: subagents hand over a ready message."""
    if run_id.startswith("subagent."):
        return detail
    return f"[system] Background process {run_id} finished: {detail}"


class ChatApp:
    def __init__(self, agent: str, profile: Any, cid: str, window: int) -> None:
        self.agent = agent
        self.profile = profile
        self.window = window
        self.cid = cid
        self.session: Session | None = None
        self.scoped = cid
        self.show_thinking = True
        self.markdown = True
        self.draft = ""
        self.notices: list[str] = []
        # process row id -> (label, status, detail)
        self.procs: dict[str, tuple[str, str, str]] = {}
        self.plan: list[PlanItem] = []
        self._plan_subs: list[tuple[Any, Any]] = []
        self._turn: TurnHandle | None = None
        self._sigints = 0

        kb = KeyBindings()

        @kb.add("enter")
        def _submit(event: Any) -> None:
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        hist = _DATA / "chat" / "input-history"
        hist.parent.mkdir(parents=True, exist_ok=True)
        self.ps: PromptSession = PromptSession(
            history=FileHistory(str(hist)),
            key_bindings=kb,
            multiline=True,
            prompt_continuation="  ",
            bottom_toolbar=self._toolbar,
        )
        get_agent_emitter(agent).on_any(self._on_process)

    # -- session lifecycle -----------------------------------------------------

    async def open(self, cid: str) -> None:
        """(Re)bind to a conversation lane: new Session, restore, subscribe."""
        if self.session is not None:
            self._close_session()
        self.cid = cid
        self.session = Session(
            self.agent,
            profile=self.profile,
            conversation_id=cid,
            resume_window=self.window,
            on_complete=self._on_complete,
        )
        self.session.attach_background()
        self.scoped = self.session._scoped_cid(cid) or cid
        emitter = get_plan_emitter(self.scoped)
        self._plan_subs = [(t, emitter.on(t, self._on_plan)) for t in PlanEventType]
        self.procs.clear()

        restored = await self.session.restore()
        if restored:
            print_history(self.session.history)
            _out(f"\n{CYAN}resumed {restored} message(s) · {cid}{RESET}\n")
        else:
            _out(f"{GREEN}new conversation · {cid}{RESET}\n")
        await self.refresh_status()

    def _close_session(self) -> None:
        emitter = get_plan_emitter(self.scoped)
        for t, cb in self._plan_subs:
            emitter.off(t, cb)
        self._plan_subs = []
        if self.session is not None:
            self.session.close()
            self.session = None

    async def shutdown(self) -> None:
        from paf.background import _background_runner

        get_agent_emitter(self.agent).off_any(self._on_process)
        self._close_session()
        await _background_runner.shutdown()

    # -- status (toolbar) ------------------------------------------------------

    async def refresh_status(self) -> None:
        """Re-read idle process rows + plan snapshot (best-effort)."""
        bg = self.session.background if self.session is not None else None
        if bg is not None:
            for lane, scope in ((bg.agent, None), (bg.conversation, self.scoped)):
                if lane is None:
                    continue
                try:
                    for row in await lane.job_status_summary(scope):
                        self.procs[_row_key(row.id)] = (
                            row.label,
                            row.status,
                            row.detail,
                        )
                except Exception:  # noqa: BLE001 - status is sugar
                    pass
            self._restore_task_rows(bg)
        try:
            self.plan = await FilePlanStore(
                root=_DATA / "plans", conversation_id=self.scoped
            ).get_items()
        except Exception:  # noqa: BLE001
            self.plan = []
        self._invalidate()

    def _restore_task_rows(self, bg: Any) -> None:
        """Shell + subagent rows live in their own stores (same as sse.py)."""
        frames: list[dict[str, Any]] = []
        try:
            from paf.capabilities.shell._background import conversation_shell_frames
            from paf.capabilities.subagents._store import conversation_subagent_frames

            root = _DATA / "processes"
            frames += conversation_shell_frames(root, self.agent, self.scoped)
            frames += conversation_subagent_frames(root, self.agent, self.scoped)
        except Exception:  # noqa: BLE001
            pass
        for f in frames:
            self.procs[str(f["id"])] = (
                str(f.get("label", "")),
                str(f.get("status", "")),
                str(f.get("detail") or ""),
            )

    def _on_process(self, ev: Any) -> None:
        if ev.lane == "conversation" and ev.scope != self.scoped:
            return
        status = getattr(ev.status, "value", ev.status)
        if status == "turn_started":
            return  # our own on_complete already knows it started a turn
        self.procs[_row_key(ev.id)] = (ev.label, status, ev.detail)
        if status in ("done", "error", "skipped"):
            asyncio.get_running_loop().create_task(self.refresh_status())
        self._invalidate()

    async def _on_plan(self, _ev: Any) -> None:
        await self.refresh_status()

    def _invalidate(self) -> None:
        if self.ps.app.is_running:
            self.ps.app.invalidate()

    def _toolbar(self) -> HTML:
        active = [(lbl, d) for lbl, st, d in self.procs.values() if st in _ACTIVE]
        failed = [lbl for lbl, st, _ in self.procs.values() if st == "error"]
        if active:
            procs = " · ".join(f"◐ {lbl}" for lbl, _ in active)
        else:
            procs = "background idle" if self.procs else "no background jobs"
        if failed:
            procs += " · ✗ " + ", ".join(failed)
        plan = ""
        if self.plan:
            done = sum(1 for i in self.plan if _status(i) == "completed")
            cur = next((i for i in self.plan if _status(i) == "in_progress"), None)
            plan = f" │ plan {done}/{len(self.plan)}"
            if cur is not None:
                plan += f": {cur.active_form or cur.content}"
        return HTML(
            f" {_esc(self.agent)} · {_esc(self.cid)} │ {_esc(procs)}{_esc(plan)}"
        )

    # -- background notify -----------------------------------------------------

    async def _on_complete(
        self, run_id: str, conversation_id: str | None, ok: bool, detail: str
    ) -> None:
        """A notify_on_completion process finished: steer it in or start a turn."""
        session = self.session
        if session is None or conversation_id != self.scoped:
            self.notices.append(f"background {run_id} finished in another conversation")
            return
        self.notices.append(f"background {run_id} {'finished' if ok else 'failed'}")
        started = session.notify(_notify_message(run_id, detail))
        if started is None and self.ps.app.is_running:
            # Idle at the prompt: stash the draft, pop the prompt, stream it.
            self.draft = self.ps.default_buffer.text
            self.ps.default_buffer.reset()
            self.ps.app.erase_when_done = True  # don't leave an empty prompt line
            self.ps.app.exit(result=_NOTIFY)

    # -- turns -------------------------------------------------------------------

    async def stream(self, handle: TurnHandle) -> None:
        """Render one Session-owned turn; Ctrl-C cancels (twice = hard stop)."""
        loop = asyncio.get_running_loop()
        self._turn, self._sigints = handle, 0
        loop.add_signal_handler(signal.SIGINT, self._on_sigint)
        printer = TurnPrinter(self.show_thinking, self.markdown)
        try:
            async for ev in handle.subscribe():
                printer.feed(ev)
        finally:
            loop.remove_signal_handler(signal.SIGINT)
            self._turn = None
            printer.finish()
        if handle.error is not None:
            _out(f"{RED}error: {handle.error}{RESET}\n")
        elif self._sigints:
            _out(f"{DIM}(cancelled){RESET}\n")
        _out("\n")

    def _on_sigint(self) -> None:
        self._sigints += 1
        if self._turn is None or self.session is None:
            return
        if self._sigints == 1:
            _out(f"\n{DIM}cancelling… (Ctrl-C again to force){RESET}\n")
            asyncio.get_running_loop().create_task(self.session.cancel())
        else:
            self._turn.cancel()

    async def drain_detached(self) -> None:
        """Stream any turn already running on the session (a notify turn)."""
        while self.session is not None:
            handle = self.session._turn
            if handle is None or handle.done:
                return
            await self.stream(handle)

    async def send(self, prompt: str) -> None:
        await self.drain_detached()  # never let start_turn re-attach + drop us
        assert self.session is not None
        _out("\n")
        await self.stream(self.session.start_turn(prompt, conversation_id=self.cid))

    # -- main loop -----------------------------------------------------------

    async def run(self) -> int:
        _out(
            f"{DIM}Enter sends · Alt+Enter newline · Ctrl-C cancels a turn · /help{RESET}\n"
        )
        while True:
            for note in self.notices:
                _out(f"{DIM}⟲ {note}{RESET}\n")
            self.notices.clear()
            await self.drain_detached()
            try:
                text = await self.ps.prompt_async(
                    HTML("<ansimagenta>›</ansimagenta> "), default=self.draft
                )
            except KeyboardInterrupt:
                self.draft = ""
                continue
            except EOFError:
                return 0
            if text is _NOTIFY:
                self.ps.app.erase_when_done = False
                continue  # draft was stashed by _on_complete; re-offered next prompt
            self.draft = ""
            text = text.strip()
            if not text:
                continue
            if text.startswith("/"):
                if not await self.command(text):
                    return 0
                continue
            await self.send(text)

    # -- slash commands --------------------------------------------------------

    async def command(self, line: str) -> bool:
        """Handle a local /command; False means exit."""
        cmd, _, arg = line.partition(" ")
        cmd, arg = cmd.lower(), arg.strip()
        if cmd in ("/quit", "/q", "/exit"):
            return False
        if cmd == "/new":
            await self.open(arg or _new_cid(self.agent))
        elif cmd == "/id":
            if arg:
                await self.open(arg)
            else:
                _out(f"{self.cid}  (store: {self.scoped})\n")
        elif cmd == "/history":
            n = int(arg) if arg.isdigit() else None
            assert self.session is not None
            print_history(self.session.history, n)
            _out("\n")
        elif cmd == "/plan":
            await self.refresh_status()
            if not self.plan:
                _out(f"{DIM}no active plan{RESET}\n")
            for item in self.plan:
                mark = {
                    "completed": "●",
                    "in_progress": "◐",
                    "cancelled": "x",
                    "blocked": "⊘",
                }.get(_status(item), "○")
                _out(f"  {mark} {item.content}\n")
        elif cmd == "/tasks":
            await self.refresh_status()
            if not self.procs:
                _out(f"{DIM}no background processes{RESET}\n")
            for label, status, detail in self.procs.values():
                color = RED if status == "error" else CYAN if status in _ACTIVE else DIM
                _out(f"  {color}{status:<8}{RESET} {label}  {DIM}{detail}{RESET}\n")
        elif cmd == "/thinking":
            self.show_thinking = not self.show_thinking
            _out(f"thinking {'shown' if self.show_thinking else 'hidden'}\n")
        elif cmd == "/markdown":
            self.markdown = not self.markdown
            _out(f"markdown {'rendered' if self.markdown else 'raw'}\n")
        elif cmd == "/memory":
            _show_memory()
        elif cmd == "/debug":
            self._show_debug()
        else:
            _out(_HELP)
        return True

    def _show_debug(self) -> None:
        s = self.session
        assert s is not None
        rows = {
            "agent": s.agent_name,
            "profile": getattr(s.profile, "name", s.profile),
            "conversation": f"{self.cid}  (store: {self.scoped})",
            "model": getattr(s.agent, "model", None),
            "resume_window": s.resume_window,
            "restored": s.restored_count,
            "history msgs": len(s.history),
            "spec": PROJECT_ROOT / "specs" / f"{s.agent_name}.yaml",
        }
        for k, v in rows.items():
            _out(f"  {CYAN}{k:<14}{RESET}{v}\n")


_HELP = """\
  /new [id]        start a fresh conversation (optionally with an id)
  /id [id]         show the conversation id, or switch to another
  /history [n]     reprint the last n turns (default: all loaded)
  /plan            show the model's current plan
  /tasks           show background processes
  /thinking        toggle showing reasoning
  /markdown        toggle rendering answers as markdown (vs raw text)
  /memory          show durable memory (MEMORY.md)
  /debug           session plumbing: model, profile, store
  /quit            exit (or Ctrl-D)
"""


def _row_key(row_id: str) -> str:
    """Coalesce a coordinator job's idle row and its per-run rows.

    Idle rows are ``consolidation.<agent>``; runs are ``consolidation.<cid>.<n>``.
    One row per job reads right in a one-line toolbar; shell/subagent rows keep
    their own ids (each process is its own row).
    """
    ns = row_id.split(".", 1)[0]
    return ns if ns in _COORDINATOR_JOBS else row_id


def _status(item: PlanItem) -> str:
    return getattr(item.status, "value", str(item.status))


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _new_cid(agent: str) -> str:
    return f"chat-{agent}-{uuid.uuid4().hex[:8]}"


def _show_memory() -> None:
    from paf.config import settings

    root_md = PROJECT_ROOT / Path(settings.memory_dir) / "MEMORY.md"
    if not root_md.exists() or not root_md.read_text(encoding="utf-8").strip():
        _out(f"{DIM}no durable memory yet ({root_md}){RESET}\n")
        return
    _out(f"{DIM}{root_md}{RESET}\n{root_md.read_text(encoding='utf-8').strip()}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Simple inline paf chat.")
    ap.add_argument("agent", nargs="?", default=DEFAULT_AGENT, help="spec name")
    ap.add_argument("--profile", help="profile override (default: the spec's)")
    lane = ap.add_mutually_exclusive_group()
    lane.add_argument("--id", help="conversation id to open or resume")
    lane.add_argument("--new", action="store_true", help="start a fresh conversation")
    ap.add_argument("--window", type=int, default=15, help="resume window (messages)")
    args = ap.parse_args()

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    profile = None
    if args.profile:
        profile = PROFILES.get(args.profile)
        if profile is None:
            print(f"unknown profile {args.profile!r}; known: {sorted(PROFILES)}")
            return 2
    cid = args.id or (_new_cid(args.agent) if args.new else f"chat-{args.agent}")

    async def _amain() -> int:
        app = ChatApp(args.agent, profile, cid, args.window)
        try:
            await app.open(cid)
            return await app.run()
        finally:
            await app.shutdown()

    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
