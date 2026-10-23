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
    paf chat [agent] [--profile P] [--id CID | --new | --resume N|NAME] [--window N]

With no conversation flag, an interactive run reopens the conversation last
open in the CLI (per agent + profile); piped input uses the default lane.
    cat task.md | paf chat contextbench --new   # one-shot: stdin = one message

From a source checkout, ``python ui/chat.py`` is the same command.

Keys: Enter sends · Alt+Enter (or Esc, Enter) inserts a newline · Ctrl-C
during a turn cancels it (twice = hard stop) · Ctrl-D exits. /help for commands.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import shlex
import signal
import sys
import termios
import time
import tty
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from paf.approvals import ApprovalRequest, ApprovalRequested
from paf.background_processes._events import get_agent_emitter
from paf.capabilities import PROFILES
from paf.capabilities.planning import PlanItem
from paf.capabilities.planning._events import PlanEventType
from paf.capabilities.planning._registry import get_plan_emitter
from paf.capabilities.visual import Describer, UploadError, UploadStore
from paf.cli._render import (
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    PROMPT_GLYPH,
    RED,
    RESET,
    YELLOW,
    TurnPrinter,
    _out,
    _width,
    parse_approval_answer,
    print_history,
    render_approval_request,
    render_user,
)
from paf.clients import notify_message, plan_items, prompt_with_images, task_frames
from paf.config import DATA_DIR, PROJECT_ROOT
from paf.conversations import (
    ProjectExistsError,
    ProjectNotFoundError,
    default_conversation_meta_store,
    default_project_store,
    list_conversations,
)
from paf.runtime import Session, TurnHandle
from paf.specs import (
    AgentSpecNotFoundError,
    InvalidSpecError,
    MissingProfileError,
    spec_visual_config,
)

DEFAULT_AGENT = "demo_assistant"
_ACTIVE = {"queued", "running", "processing"}
_COORDINATOR_JOBS = {"consolidation", "review", "thread_summarize", "keyword_curate"}
_NOTIFY = object()  # prompt result: a background notify started a turn
_SETTLED = object()  # approval prompt result: the request was settled elsewhere

# Prompt-toolkit styling. The default bottom toolbar is reverse video (a solid
# white bar); this keeps it in the terminal's own colors, dimmed, with state
# colors only where they carry meaning.
_PT_STYLE = Style.from_dict(
    {
        "bottom-toolbar": "noreverse #808080",
        "bottom-toolbar.text": "",
        "toolbar.active": "#5fafd7",
        "toolbar.failed": "#d75f5f",
        "rule": "#444444",
        "marker": "#d787ff bold",
        "placeholder": "#626262 italic",
    }
)

# --- steering input ---------------------------------------------------------


class _KeyWatch:
    """Notice typing while a turn streams, without echoing it into the output.

    The terminal goes into cbreak mode (no echo, no line buffering; Ctrl-C
    still raises SIGINT) and stdin is watched on the event loop. The first
    keypress is posted to ``inbox`` as ``("key", text)`` and the watch stops
    reading, so anything typed after it stays in the terminal's buffer for the
    steer prompt to pick up. ``resume`` re-arms it once that prompt is done.
    """

    def __init__(self, inbox: asyncio.Queue[tuple[str, Any]]) -> None:
        self.inbox = inbox
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        self.loop = asyncio.get_running_loop()
        self.reading = False

    def resume(self) -> None:
        # TCSANOW, not the default TCSAFLUSH, which would drop typed-ahead keys.
        tty.setcbreak(self.fd, termios.TCSANOW)
        if not self.reading:
            self.loop.add_reader(self.fd, self._on_readable)
            self.reading = True

    def pause(self) -> None:
        if self.reading:
            self.loop.remove_reader(self.fd)
            self.reading = False
        termios.tcsetattr(self.fd, termios.TCSANOW, self.saved)

    def _on_readable(self) -> None:
        try:
            data = os.read(self.fd, 1024).decode("utf-8", errors="ignore")
        except OSError:
            return
        # Arrow keys and other escape sequences, and Ctrl-D, don't start a steer.
        if not data or data.startswith("\x1b") or data == "\x04":
            return
        self.pause()
        self.inbox.put_nowait(("key", data))


# --- the app ------------------------------------------------------------------


class ChatApp:
    def __init__(self, agent: str, profile: Any, cid: str, window: int) -> None:
        self.agent = agent
        self.profile = profile
        self.window = window
        self.cid = cid
        self.meta = default_conversation_meta_store(agent)
        self.projects = default_project_store()
        self.uploads = UploadStore()
        self.describer = Describer(self.uploads)
        # Numbered rows from the last /list, so `/resume 3` means what was shown.
        self._listing: list[dict[str, Any]] = []
        self.session: Session | None = None
        self.scoped = cid
        self.show_thinking = True
        self.markdown = True
        self.draft = ""
        self.notices: list[str] = []
        self.queued: list[str] = []  # steer messages that arrived after their turn
        # process row id -> (label, status, detail)
        self.procs: dict[str, tuple[str, str, str]] = {}
        self.plan: list[PlanItem] = []
        self._plan_subs: list[tuple[Any, Any]] = []
        self._turn: TurnHandle | None = None
        self._sigints = 0
        self._refresh_task: asyncio.Task[None] | None = None
        self._refresh_dirty = False

        kb = KeyBindings()

        @kb.add("enter")
        def _submit(event: Any) -> None:
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        hist = DATA_DIR / "chat" / "input-history"
        hist.parent.mkdir(parents=True, exist_ok=True)
        # No line editor for piped input (one-shot mode never prompts).
        self.ps: PromptSession | None = None
        if sys.stdin.isatty():
            self.ps = PromptSession(
                history=FileHistory(str(hist)),
                key_bindings=kb,
                multiline=True,
                prompt_continuation="  ",
                bottom_toolbar=self._toolbar,
                style=_PT_STYLE,
                # The raw input is erased on Enter and re-printed in the user
                # style (render_user), so messages stand out in scrollback.
                erase_when_done=True,
                placeholder=HTML("<placeholder>Message paf · /help</placeholder>"),
            )
            # Its own session: prompt_toolkit keeps per-call options (toolbar,
            # placeholder) on a session, so sharing one would change the main prompt.
            self.steer_ps: PromptSession = PromptSession(
                key_bindings=kb,
                multiline=True,
                prompt_continuation="  ",
                style=_PT_STYLE,
                erase_when_done=True,
                placeholder=HTML(
                    "<placeholder>sent into the running turn · Ctrl-C drops it"
                    "</placeholder>"
                ),
            )
            # Approval answers: one line, never saved to the input history.
            self.approve_ps: PromptSession = PromptSession(
                key_bindings=kb,
                multiline=False,
                style=_PT_STYLE,
                erase_when_done=False,
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
        self.scoped = self.session.scoped_id(cid) or cid
        emitter = get_plan_emitter(self.scoped)
        self._plan_subs = [(t, emitter.on(t, self._on_plan)) for t in PlanEventType]
        self.procs.clear()

        restored = await self.session.restore()
        _remember_last(self.agent, getattr(self.profile, "name", None), cid)
        title = self.meta.get(cid).get("title")
        label = f"{title} ({cid})" if title else cid
        if restored:
            print_history(self.session.history)
            _out(f"\n{CYAN}resumed {restored} message(s) · {label}{RESET}\n")
        else:
            _out(f"{GREEN}new conversation · {label}{RESET}\n")
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
        """Re-read process rows + plan, and wait until the toolbar reflects them.

        Coalesced: one read runs at a time, and any number of requests that
        arrive during it share a single follow-up read. A burst of background
        events (several jobs finishing after a turn) would otherwise start one
        full status read per event, and their combined work starves the event
        loop, so the prompt and toolbar stall.
        """
        self._request_refresh()
        assert self._refresh_task is not None
        await asyncio.shield(self._refresh_task)

    def _request_refresh(self) -> None:
        """Ask for a status re-read without waiting for it (see refresh_status)."""
        self._refresh_dirty = True
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.get_running_loop().create_task(
                self._refresh_loop()
            )

    async def _refresh_loop(self) -> None:
        while self._refresh_dirty:
            self._refresh_dirty = False
            await self._read_status()

    async def _read_status(self) -> None:
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
            self._restore_task_rows()
        try:
            self.plan = await plan_items(self.scoped)
        except Exception:  # noqa: BLE001
            self.plan = []
        self._invalidate()

    def _restore_task_rows(self) -> None:
        """Shell + subagent rows live in their own stores (same as ui/sse.py)."""
        for f in task_frames(self.agent, self.scoped):
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
            self._request_refresh()
        self._invalidate()

    async def _on_plan(self, _ev: Any) -> None:
        await self.refresh_status()

    def _invalidate(self) -> None:
        if self.ps is not None and self.ps.app.is_running:
            self.ps.app.invalidate()

    def _toolbar(self) -> HTML:
        active = [(lbl, d) for lbl, st, d in self.procs.values() if st in _ACTIVE]
        failed = [lbl for lbl, st, _ in self.procs.values() if st == "error"]
        if active:
            busy = " · ".join(f"◐ {lbl}" for lbl, _ in active)
            procs = f"<toolbar.active>{_esc(busy)}</toolbar.active>"
        else:
            procs = "background idle" if self.procs else "no background jobs"
        if failed:
            gone = "✗ " + ", ".join(failed)
            procs += f" · <toolbar.failed>{_esc(gone)}</toolbar.failed>"
        plan = ""
        if self.plan:
            done = sum(1 for i in self.plan if _status(i) == "completed")
            cur = next((i for i in self.plan if _status(i) == "in_progress"), None)
            plan = f" │ plan {done}/{len(self.plan)}"
            if cur is not None:
                plan += f": {cur.active_form or cur.content}"
        saved = self.meta.get(self.cid)
        project = saved.get("project")
        where = f"{self.agent} · {saved.get('title') or self.cid}" + (
            f" · [{project}]" if project else ""
        )
        # Leading newline: a blank line between the input and the status line.
        return HTML(f"\n {_esc(where)} │ {procs}{_esc(plan)}")

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
        started = session.notify(notify_message(run_id, detail))
        if started is None and self.ps is not None and self.ps.app.is_running:
            # Idle at the prompt: stash the draft, pop the prompt, stream it.
            self.draft = self.ps.default_buffer.text
            self.ps.default_buffer.reset()
            self.ps.app.exit(result=_NOTIFY)

    # -- turns -------------------------------------------------------------------

    async def stream(self, handle: TurnHandle) -> None:
        """Render one Session-owned turn; Ctrl-C cancels (twice = hard stop).

        Typing while it streams opens a ``steer`` line: the output pauses, and
        what you send is steered into the running turn (``Session.enqueue``).
        """
        loop = asyncio.get_running_loop()
        self._turn, self._sigints = handle, 0
        loop.add_signal_handler(signal.SIGINT, self._on_sigint)
        printer = TurnPrinter(self.show_thinking, self.markdown)
        inbox: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def pump() -> None:
            try:
                async for ev in handle.subscribe():
                    inbox.put_nowait(("event", ev))
            finally:
                inbox.put_nowait(("end", None))

        pump_task = asyncio.create_task(pump())
        keys = _KeyWatch(inbox) if self.ps is not None else None
        try:
            if keys is not None:
                keys.resume()
            while True:
                kind, item = await inbox.get()
                if kind == "end":
                    break
                if kind == "event":
                    if isinstance(item, ApprovalRequested):
                        printer.pause()
                        if keys is not None:
                            keys.pause()
                        await self._approve(item.request)
                        if keys is not None and not handle.done:
                            keys.resume()
                        continue
                    printer.feed(item)
                    continue
                printer.pause()
                await self._steer(item)
                if keys is not None and not handle.done:
                    keys.resume()
        finally:
            if keys is not None:
                keys.pause()
            pump_task.cancel()
            await asyncio.gather(pump_task, return_exceptions=True)
            loop.remove_signal_handler(signal.SIGINT)
            self._turn = None
            printer.finish()
        if handle.error is not None:
            _out(f"{RED}error: {handle.error}{RESET}\n")
        elif self._sigints:
            _out(f"{DIM}(cancelled){RESET}\n")
        _out("\n")

    async def _approve(self, req: ApprovalRequest) -> None:
        """Ask the user to approve one tool call, then settle it.

        ``y`` approves; ``n``, Enter or Ctrl-C denies, and text after ``n`` is
        passed to the agent as the reason. If the request is settled some other
        way while the prompt is open (it timed out, the turn was stopped, another
        client answered) the prompt closes by itself.
        """
        assert self.session is not None
        _out(render_approval_request(req))
        if getattr(self, "approve_ps", None) is None:
            return  # no terminal: the request times out (it should not be asked)
        session = self.session

        async def watch() -> None:
            while True:
                await asyncio.sleep(0.2)
                if all(r.id != req.id for r in session.pending_approvals()):
                    app = self.approve_ps.app
                    if app.is_running:
                        app.exit(result=_SETTLED)
                    return

        watcher = asyncio.create_task(watch())
        try:
            answer = await self.approve_ps.prompt_async(
                HTML("<marker>approve? [y/N] ❯</marker> ")
            )
        except (KeyboardInterrupt, EOFError):
            answer = ""
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        if answer is _SETTLED:
            return  # the ApprovalResolved event that follows says how
        approved, note = parse_approval_answer(answer)
        if not session.resolve_approval(req.id, approved, note=note, actor="cli"):
            _out(f"{DIM}  (already settled){RESET}\n")

    async def _steer(self, typed: str) -> None:
        """Read a steer message (starting with the keys already typed) and send it.

        A line typed in one go (a paste, or keys faster than the read) arrives
        with its Enter and is sent as-is; otherwise a ``steer`` prompt opens.
        Ctrl-C there drops the message. If the turn ended meanwhile, the
        message is sent as the next turn instead.
        """
        assert self.session is not None and self.ps is not None
        first, newline, _ = typed.replace("\r", "\n").partition("\n")
        text = first.strip()
        if not newline:
            try:
                text = await self.steer_ps.prompt_async(
                    HTML("<marker>steer ❯</marker> "), default=first
                )
            except (KeyboardInterrupt, EOFError):
                text = ""
            text = text.strip()
        if not text:
            _out(f"{DIM}(steer dropped){RESET}\n")
            return
        _out(render_user(text))
        if await self.session.enqueue(text) is not None:
            _out(f"{DIM}  ↳ steering the running turn{RESET}\n\n")
        else:
            self.queued.append(text)
            _out(f"{DIM}  ↳ the turn already finished · sending it next{RESET}\n\n")

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
            handle = self.session.current_turn
            if handle is None:
                return
            await self.stream(handle)

    async def send(self, prompt: str) -> None:
        await self.drain_detached()  # never let start_turn re-attach + drop us
        assert self.session is not None
        _out("\n")
        # A person at a terminal can answer approval requests; piped input
        # cannot, so its turns refuse anything that needs approval.
        await self.stream(
            self.session.start_turn(
                prompt, conversation_id=self.cid, interactive=self.ps is not None
            )
        )
        # A steer that missed its turn (it ended first) becomes the next message.
        while self.queued:
            await self.send(self.queued.pop(0))

    # -- main loop -----------------------------------------------------------

    async def run(self) -> int:
        if not sys.stdin.isatty():
            # Piped input (e.g. `cat task.md | chat.py contextbench --new`):
            # the whole of stdin is one message; stream the turn and exit.
            prompt = sys.stdin.read().strip()
            if prompt:
                await self.send(prompt)
            return 0
        assert self.ps is not None
        _out(
            f"{DIM}Enter sends · Alt+Enter newline · type during a turn to steer it · "
            f"Ctrl-C cancels a turn · /help{RESET}\n"
        )
        while True:
            for note in self.notices:
                _out(f"{DIM}⟲ {note}{RESET}\n")
            self.notices.clear()
            await self.drain_detached()
            try:
                text = await self.ps.prompt_async(_prompt_message, default=self.draft)
            except KeyboardInterrupt:
                self.draft = ""
                continue
            except EOFError:
                return 0
            if text is _NOTIFY:
                continue  # draft was stashed by _on_complete; re-offered next prompt
            self.draft = ""
            text = text.strip()
            if not text:
                continue
            if text.startswith("/"):
                _out(f"{DIM}{PROMPT_GLYPH} {text}{RESET}\n")
                if not await self.command(text):
                    return 0
                continue
            _out(render_user(text))
            await self.send(text)

    # -- slash commands --------------------------------------------------------

    async def command(self, line: str) -> bool:
        """Handle a local /command; False means exit."""
        cmd, _, arg = line.partition(" ")
        cmd, arg = cmd.lower(), arg.strip()
        if cmd in ("/quit", "/q", "/exit"):
            return False
        if cmd == "/new":
            # A fresh conversation stays in the current project.
            project = self.meta.project_for(self.cid)
            new_cid = arg or _new_cid(self.agent)
            if project:
                self.meta.set_project(new_cid, project)
            await self.open(new_cid)
            if project:
                _out(f"{DIM}project: {project}{RESET}\n")
        elif cmd == "/id":
            if arg:
                await self.open(arg)
            else:
                _out(f"{self.cid}  (store: {self.scoped})\n")
        elif cmd == "/list":
            await self._list_command(show_all=arg.lower() == "all")
        elif cmd == "/resume":
            await self._resume_command(arg)
        elif cmd == "/name":
            self._name_command(arg)
        elif cmd in ("/hide", "/unhide"):
            hidden = cmd == "/hide"
            self.meta.set_hidden(self.cid, hidden)
            _out(
                "hidden from /list (still in /list all)\n"
                if hidden
                else "shown in /list\n"
            )
        elif cmd == "/image":
            await self._image_command(arg)
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
        elif cmd == "/project":
            self._project_command(arg)
        elif cmd == "/projects":
            self._projects_command(arg)
        elif cmd == "/memory":
            _show_memory(self.agent)
        elif cmd == "/debug":
            self._show_debug()
        else:
            _out(_HELP)
        return True

    async def _conversations(self) -> list[dict[str, Any]]:
        # The session's profile, not self.profile: a Session resolves the
        # spec's default profile when none was passed.
        profile = self.session.profile if self.session is not None else self.profile
        return await list_conversations(
            self.agent, profile=getattr(profile, "name", None)
        )

    async def _list_command(self, *, show_all: bool) -> None:
        """``/list`` recent visible conversations · ``/list all`` incl. hidden."""
        rows = await self._conversations()
        if not show_all:
            rows = [r for r in rows if not r["hidden"]]
        shown = rows if show_all else rows[:_LIST_LIMIT]
        self._listing = shown
        if not shown:
            _out(f"{DIM}no saved conversations yet{RESET}\n")
            return
        for n, row in enumerate(shown, start=1):
            _out(_list_row(n, row, current=row["conversation_id"] == self.cid))
        if len(rows) > len(shown):
            _out(f"{DIM}  … {len(rows) - len(shown)} more (/list all){RESET}\n")
        _out(f"{DIM}/resume <n | name | id>{RESET}\n")

    async def _resume_command(self, arg: str) -> None:
        """``/resume <n | name | id>`` — reopen a saved conversation."""
        if not arg:
            await self._list_command(show_all=False)
            return
        rows = self._listing if arg.isdigit() and self._listing else None
        if rows is None:
            rows = await self._conversations()
            if arg.isdigit():  # no prior /list: number the default view
                rows = [r for r in rows if not r["hidden"]][:_LIST_LIMIT]
        cid, matches = match_conversation(rows, arg)
        if cid is None:
            if matches:
                _out(f"{YELLOW}{len(matches)} conversations match {arg!r}:{RESET}\n")
                for n, row in enumerate(matches, start=1):
                    _out(_list_row(n, row, current=row["conversation_id"] == self.cid))
                self._listing = matches
                _out(f"{DIM}/resume <n> to pick one{RESET}\n")
            else:
                _out(f"{RED}no conversation matches {arg!r}{RESET} (/list)\n")
            return
        if cid == self.cid:
            _out(f"{DIM}already in {cid}{RESET}\n")
            return
        await self.open(cid)

    def _name_command(self, arg: str) -> None:
        """``/name`` show · ``/name <title>`` set · ``/name -`` clear."""
        if not arg:
            title = self.meta.get(self.cid).get("title")
            _out(f"{title}\n" if title else f"{DIM}unnamed · /name <title>{RESET}\n")
            return
        if arg == "-":
            self.meta.set_title(self.cid, None)
            _out("name cleared\n")
            return
        self.meta.set_title(self.cid, arg)
        _out(f"named: {arg}\n")

    def _project_command(self, arg: str) -> None:
        """``/project`` show · ``/project <slug>`` assign · ``/project none`` unassign."""
        if not arg:
            project = self.meta.project_for(self.cid)
            _out(f"{project}\n" if project else f"{DIM}none{RESET}\n")
            return
        if arg.lower() == "none":
            self.meta.set_project(self.cid, None)
            _out("project cleared\n")
            return
        entry = self.projects.get(arg)
        if entry is None or entry.get("archived"):
            known = ", ".join(p["slug"] for p in self.projects.list()) or "none yet"
            _out(f"{RED}unknown project {arg!r}{RESET} (known: {known})\n")
            return
        self.meta.set_project(self.cid, arg)
        _out(f"project: {arg}\n")

    def _projects_command(self, arg: str) -> None:
        """``/projects`` list · ``add <name> [-- description]`` · ``rename <slug> <name>``
        · ``archive <slug>`` · ``unarchive <slug>``."""
        sub, _, rest = arg.partition(" ")
        sub, rest = sub.lower(), rest.strip()
        if sub in ("rename", "archive", "unarchive"):
            slug, _, name = rest.partition(" ")
            if not slug or (sub == "rename" and not name.strip()):
                usage = (
                    "rename <slug> <new name>" if sub == "rename" else f"{sub} <slug>"
                )
                _out(f"{RED}usage: /projects {usage}{RESET}\n")
                return
            try:
                if sub == "rename":
                    entry = self.projects.update(slug, name=name)
                    _out(f"renamed {slug} → {entry['name']}\n")
                else:
                    self.projects.update(slug, archived=sub == "archive")
                    _out(f"{sub}d {slug}\n")
            except ProjectNotFoundError:
                _out(f"{RED}unknown project {slug!r}{RESET}\n")
            except ValueError as exc:
                _out(f"{RED}{exc}{RESET}\n")
            return
        if sub == "add":
            name, _, description = rest.partition("--")
            try:
                entry = self.projects.create(name, description)
            except ProjectExistsError as exc:
                _out(f"{RED}project {exc.args[0]!r} already exists{RESET}\n")
                return
            except ValueError as exc:
                _out(f"{RED}{exc}{RESET}\n")
                return
            _out(f"created {entry['slug']} — assign with /project {entry['slug']}\n")
            return
        current = self.meta.project_for(self.cid)
        rows = self.projects.list()
        if not rows:
            _out(
                f"{DIM}no projects yet — /projects add <name> [-- description]{RESET}\n"
            )
        for p in rows:
            mark = "●" if p["slug"] == current else " "
            desc = f"  {DIM}{p['description']}{RESET}" if p.get("description") else ""
            _out(f"  {mark} {CYAN}{p['slug']}{RESET}{desc}\n")

    async def _image_command(self, arg: str) -> None:
        """``/image <path> [message]`` — send an image (described by the vision model)."""
        path, message = _split_path_arg(arg)
        if path is None:
            _out(f"{RED}usage: /image <path> [message]{RESET}\n")
            return
        try:
            data = path.read_bytes()
        except OSError as exc:
            _out(f"{RED}can't read {path}: {exc.strerror or exc}{RESET}\n")
            return
        try:
            meta = self.uploads.save(data, path.name)
        except UploadError as exc:
            _out(f"{RED}{exc}{RESET}\n")
            return
        visual = spec_visual_config(self.agent)
        if visual["model"]:
            _out(f"{DIM}describing {path.name} with {visual['model']}…{RESET}\n")
        else:
            _out(
                f"{YELLOW}{self.agent} has no vision model (visual.model); "
                f"sending without a description{RESET}\n"
            )
        prompt = await prompt_with_images(
            message,
            [meta.id],
            model=visual["model"],
            model_settings=visual.get("model_settings"),
            uploads=self.uploads,
            describer=self.describer,
        )
        _out(render_user(f"{message}\n[image: {path.name}]".strip()))
        await self.send(prompt)

    def _show_debug(self) -> None:
        s = self.session
        assert s is not None
        rows = {
            "agent": s.agent_name,
            "profile": getattr(s.profile, "name", s.profile),
            "conversation": f"{self.cid}  (store: {self.scoped})",
            "project": self.meta.project_for(self.cid) or "none",
            "model": _model_label(getattr(s.agent, "model", None)),
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
  /list [all]      saved conversations, newest first ('all' includes hidden)
  /resume <n|name> reopen one: number from /list, part of its name, or an id
  /name [title]    show or set this conversation's name ('-' clears it)
  /hide · /unhide  hide this conversation from /list, or show it again
  /project [slug]  show this conversation's project, assign one, or 'none'
  /projects        list projects · /projects add <name> [-- description]
                   · rename <slug> <name> · archive <slug> · unarchive <slug>
  /image <path> [message]  send an image; the agent's vision model describes it
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


_LIST_LIMIT = 20


def match_conversation(
    rows: list[dict[str, Any]], arg: str
) -> tuple[str | None, list[dict[str, Any]]]:
    """Resolve ``/resume``'s argument against listed rows.

    A number picks that row (1-based); otherwise an exact id wins, then a
    case-insensitive substring of the name or id. Returns ``(cid, [])`` on a
    unique hit, else ``(None, candidates)`` — several matches, or none.
    """
    arg = arg.strip()
    if arg.isdigit():
        n = int(arg)
        return (
            (rows[n - 1]["conversation_id"], []) if 1 <= n <= len(rows) else (None, [])
        )
    for row in rows:
        if row["conversation_id"] == arg:
            return row["conversation_id"], []
    needle = arg.lower()
    hits = [
        r
        for r in rows
        if needle in (r.get("title") or "").lower()
        or needle in r["conversation_id"].lower()
    ]
    if len(hits) == 1:
        return hits[0]["conversation_id"], []
    # Prefer an exact (case-insensitive) name when a substring is ambiguous.
    exact = [r for r in hits if (r.get("title") or "").lower() == needle]
    if len(exact) == 1:
        return exact[0]["conversation_id"], []
    return None, hits


def _ago(epoch: float | None) -> str:
    if not epoch:
        return ""
    s = max(0.0, time.time() - epoch)
    if s < 3600:
        return f"{int(s // 60)}m ago"
    if s < 86400:
        return f"{int(s // 3600)}h ago"
    return f"{int(s // 86400)}d ago"


def _list_row(n: int, row: dict[str, Any], *, current: bool) -> str:
    title = row.get("title")
    name = title if title else f"{DIM}{row['conversation_id']}{RESET}"
    project = f"{CYAN}[{row['project']}]{RESET} " if row.get("project") else ""
    hidden = f"{DIM}(hidden) {RESET}" if row.get("hidden") else ""
    mark = f"{MAGENTA}●{RESET}" if current else " "
    return f"  {n:>2} {mark} {name}  {project}{hidden}{DIM}{_ago(row.get('updated_at'))}{RESET}\n"


def _status(item: PlanItem) -> str:
    return getattr(item.status, "value", str(item.status))


def _prompt_message() -> HTML:
    """A dim rule across the terminal, then the input marker.

    A callable so the rule re-measures on resize. The rule stays in scrollback
    after Enter, separating turns.
    """
    rule = "─" * max(10, _width() - 1)
    return HTML(f"<rule>{rule}</rule>\n<marker>{PROMPT_GLYPH}</marker> ")


def _model_label(model: Any) -> str | None:
    """``provider:model-name`` (e.g. ``openrouter:deepseek/deepseek-v4-flash-0731``).

    pydantic-ai model objects don't name themselves in their repr; ``system`` is
    the provider and ``model_name`` the model id. A bare string passes through.
    """
    if model is None or isinstance(model, str):
        return model
    name = getattr(model, "model_name", None)
    system = getattr(model, "system", None)
    if name and system:
        return f"{system}:{name}"
    return name or repr(model)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _last_path(agent: str, profile: str | None) -> Path:
    return DATA_DIR / "chat" / f"last-{agent}{'-' + profile if profile else ''}"


def _remember_last(agent: str, profile: str | None, cid: str) -> None:
    """Record the conversation just opened so a bare `chat.py` can reopen it."""
    try:
        path = _last_path(agent, profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(cid, encoding="utf-8")
    except OSError:
        pass  # a convenience — never block opening a conversation


def _last_conversation(agent: str, profile: str | None) -> str | None:
    try:
        cid = _last_path(agent, profile).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return cid or None


def _new_cid(agent: str) -> str:
    return f"chat-{agent}-{uuid.uuid4().hex[:8]}"


def _split_path_arg(arg: str) -> tuple[Path | None, str]:
    """``<path> [rest]`` → (path, rest). The path may be quoted; ``~`` expands."""
    stream = io.StringIO(arg)
    lex = shlex.shlex(stream, posix=True)
    lex.whitespace_split = True
    try:
        token = lex.get_token()
    except ValueError:  # unbalanced quote
        return None, ""
    if not token:
        return None, ""
    return Path(token).expanduser(), stream.read().strip()


def _show_memory(agent: str) -> None:
    from paf.capabilities.memory._window import memory_dir_for

    root_md = Path(memory_dir_for(agent)) / "MEMORY.md"
    if not root_md.exists() or not root_md.read_text(encoding="utf-8").strip():
        _out(f"{DIM}no durable memory yet ({root_md}){RESET}\n")
        return
    _out(f"{DIM}{root_md}{RESET}\n{root_md.read_text(encoding='utf-8').strip()}\n")


def main(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description="Chat with a paf agent.")
    ap.add_argument("agent", nargs="?", default=DEFAULT_AGENT, help="spec name")
    ap.add_argument("--profile", help="profile override (default: the spec's)")
    lane = ap.add_mutually_exclusive_group()
    lane.add_argument("--id", help="conversation id to open or resume")
    lane.add_argument("--new", action="store_true", help="start a fresh conversation")
    lane.add_argument(
        "--resume", metavar="N|NAME", help="reopen a saved conversation (see /list)"
    )
    ap.add_argument("--window", type=int, default=15, help="resume window (messages)")
    ap.add_argument("--project", help="assign this conversation to a project (slug)")
    args = ap.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    profile = None
    if args.profile:
        profile = PROFILES.get(args.profile)
        if profile is None:
            print(f"unknown profile {args.profile!r}; known: {sorted(PROFILES)}")
            return 2
    cid = args.id or (_new_cid(args.agent) if args.new else f"chat-{args.agent}")
    # A bare interactive `chat.py` reopens the conversation last open in the CLI
    # (piped one-shot runs keep the fixed default lane, so scripts stay stable).
    auto_resumed = False
    if not (args.id or args.new or args.resume) and sys.stdin.isatty():
        last = _last_conversation(args.agent, getattr(profile, "name", None))
        if last:
            cid, auto_resumed = last, True
    if args.project:
        entry = default_project_store().get(args.project)
        if entry is None or entry.get("archived"):
            known = sorted(p["slug"] for p in default_project_store().list())
            print(f"unknown project {args.project!r}; known: {known}")
            return 2

    async def _amain() -> int:
        nonlocal cid
        if args.resume:
            rows = await list_conversations(
                args.agent, profile=getattr(profile, "name", None)
            )
            visible = [r for r in rows if not r["hidden"]][:_LIST_LIMIT]
            found, matches = match_conversation(
                visible if args.resume.isdigit() else rows, args.resume
            )
            if found is None:
                print(f"no unique conversation for {args.resume!r}")
                for n, row in enumerate(matches, start=1):
                    print(_list_row(n, row, current=False), end="")
                return 2
            cid = found
        if args.project:
            default_conversation_meta_store(args.agent).set_project(cid, args.project)
        app = ChatApp(args.agent, profile, cid, args.window)
        try:
            await app.open(cid)
            if auto_resumed:
                _out(
                    f"{DIM}reopened your last conversation · /new for a fresh one{RESET}\n"
                )
            return await app.run()
        finally:
            await app.shutdown()

    try:
        return asyncio.run(_amain())
    except (AgentSpecNotFoundError, InvalidSpecError, MissingProfileError) as exc:
        # A spec problem the user fixes by editing a file: one line, no traceback.
        print(f"paf chat: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
