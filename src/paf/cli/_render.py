"""Terminal rendering for the paf chat CLI.

Turns stream into normal scrollback: dim thinking, one line per tool call,
then the answer, with each finished markdown block swapped in place for its
rendered form. Pure output code — no Session, no prompt — so it is testable
by capturing stdout.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
from typing import Any, ClassVar

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

from paf.approvals import ApprovalRequest, ApprovalResolved
from paf.runtime import StreamEvent, ToolError

_DATE_STAMP = re.compile(r"^\[date: [^\]]*\]\s*")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# --- terminal styling -------------------------------------------------------

_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _sgr(code: str) -> str:
    return f"\x1b[{code}m" if _COLOR else ""


DIM, RED, GREEN, YELLOW, CYAN, MAGENTA, RESET = (
    _sgr(c) for c in ("2", "31", "32", "33", "36", "35", "0")
)
# Thinking text: an explicit mid-gray. Terminals render the faint attribute
# (DIM) inconsistently — often barely distinct from normal text.
THINK = _sgr("38;5;244")
# User messages: their own color + a bolder marker, so they stand apart from
# the agent's output when scrolling back.
USER_FG = _sgr("38;5;183")  # light purple
USER_MARK = _sgr("1;38;5;177")  # bold, brighter purple
PROMPT_GLYPH = "❯"


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


def _open_fence(text: str) -> str | None:
    """The opening line of a code fence ``text`` leaves unclosed, if any."""
    opener: str | None = None
    for line in text.splitlines():
        s = line.strip()
        if opener is None:
            if s.startswith(("```", "~~~")):
                opener = line
        elif s.startswith(opener.strip()[:3]):
            opener = None
    return opener


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
        self.hidden = 0  # leading chars of ``shown`` never printed (a carried fence)

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
        if self.enabled and self.tail[self.hidden :].strip():
            self._emit(self.tail, last=True)
        elif not self.col0:
            _out("\n")
        self.tail = self.shown = ""
        self.hidden = 0

    def break_block(self) -> None:
        """End the current block early (to make room for input); cursor at column 0.

        A code fence left open is closed for this rendering and re-opened,
        unseen, for the text that follows, so the rest still renders as code.
        """
        if not self.enabled or not self.tail[self.hidden :].strip():
            if not self.col0:
                _out("\n")
            self.tail = self.shown = self.tail[: self.hidden]
            return
        fence = _open_fence(self.tail)
        block = self.tail
        if fence is not None:
            block = f"{block.rstrip(chr(10))}\n{fence.strip()[:3]}\n"
        self._emit(block, last=True)
        carry = f"{fence}\n" if fence is not None else ""
        self.tail = self.shown = carry
        self.hidden = len(carry)

    def _emit(self, block: str, *, last: bool) -> None:
        seen = self.shown[self.hidden :]
        self.hidden = 0
        up = _rows(seen, _width()) if seen else 0
        if up >= shutil.get_terminal_size((100, 24)).lines - 1:
            # Its top has scrolled off: keep the raw text, just finish printing it.
            rest = block[len(self.shown) :]
            _out(rest)
            if not (rest or self.shown).endswith("\n"):
                _out("\n")
        else:
            if seen:
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
    Calls and results are matched by ``step_id``. A failure arrives as a
    ``ToolError`` right after its ``tool_result``, so a result is held until
    the next event shows whether one follows.
    """

    def __init__(self, show_thinking: bool, markdown: bool) -> None:
        self.show_thinking = show_thinking
        self.md = MarkdownStream(markdown)
        self.mode: str | None = None  # "thinking" | "text" | "tool"
        self.col0 = True  # cursor at column 0
        self.open_step: str | None = None  # tool whose line is still open
        self.pending: dict[str, str] = {}  # step_id -> tool name (awaiting result)
        self.resolving: str | None = None  # step whose result arrived, outcome unknown
        # step_id -> why it did not run (an approval that was not granted). The
        # tool still "returns" (the denial message), so without this the line
        # would end in a misleading ✓.
        self.not_run: dict[str, str] = {}

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
        if error is None and step in self.not_run:
            error = self.not_run.pop(step)
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
        if isinstance(ev, ApprovalResolved):
            self._newline()
            self._write(render_approval_resolved(ev))
            if ev.outcome != "approved":
                self.not_run[ev.tool_call_id] = _NOT_RUN[ev.outcome]
            return
        if isinstance(ev, ToolError):
            if self.resolving == ev.step_id:
                self._flush_result(ev.error)
            elif ev.step_id in self.pending:
                self._settle(ev.step_id, ev.error)
            return
        self._flush_result()
        if not isinstance(ev, StreamEvent):
            return
        if ev.kind == "thinking" and ev.text:
            if not self.show_thinking:
                return
            if self.mode != "thinking":
                self._newline()
                if self.mode is not None:  # space it off tool calls / answer text
                    self._write("\n")
                self._write(f"{CYAN}thinking:{RESET} ")
                self.mode = "thinking"
            self._write(f"{THINK}{_safe(ev.text)}{RESET}")
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
            self.pending[ev.step_id] = name
            self.open_step = ev.step_id
        elif ev.kind == "tool_result" and ev.step_id in self.pending:
            self.resolving = ev.step_id

    def pause(self) -> None:
        """Put the cursor at column 0 so input can be drawn; the turn resumes after.

        An answer mid-block is rendered as far as it got. Thinking gets its
        label again when it resumes.
        """
        if self.mode == "text":
            self.md.break_block()
            self.col0 = True
            return
        self.open_step = None
        if not self.col0:
            self._write("\n")
        if self.mode == "thinking":
            self.mode = None

    def finish(self) -> None:
        self._flush_result()
        self._newline()


# --- approvals ------------------------------------------------------------------

_NOT_RUN = {
    "denied": "denied, not run",
    "timeout": "no answer, not run",
    "cancelled": "cancelled, not run",
    "unattended": "needs approval, not run",
}
_OUTCOME = {
    "approved": f"{GREEN}✓ approved{RESET}",
    "denied": f"{RED}✗ denied{RESET}",
    "timeout": f"{RED}✗ no answer in time, denied{RESET}",
    "cancelled": f"{RED}✗ cancelled{RESET}",
    "unattended": f"{RED}✗ nobody to ask, denied{RESET}",
}


def _pretty_args(args: str) -> list[str]:
    """Arguments as readable lines: one ``key: value`` per argument when JSON."""
    import json

    try:
        data = json.loads(args)
    except (TypeError, ValueError):
        return [args]
    if not isinstance(data, dict):
        return [json.dumps(data, ensure_ascii=False)]
    lines = []
    for key, value in data.items():
        text = (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )
        lines.append(f"{key}: {text}")
    return lines


def render_approval_request(req: ApprovalRequest, *, now: float | None = None) -> str:
    """The panel shown when a tool call is waiting for the user."""
    import time

    width = max(40, _width() - 4)
    minutes = max(1, round((req.expires_at - (now or time.time())) / 60))
    lines = [f"\n{YELLOW}⚠ approval needed{RESET}  {req.tool_name}"]
    for line in _pretty_args(req.args):
        for chunk in textwrap.wrap(_safe(line), width - 4) or [""]:
            lines.append(f"    {chunk}")
    lines.append(f"  {DIM}why: {_safe(req.reason)}{RESET}")
    if req.preview:
        for key, value in req.preview.items():
            lines.append(f"  {DIM}{key}: {_safe(str(value))}{RESET}")
    lines.append(
        f"  {DIM}y approve · n deny (text after n is sent to the agent) · "
        f"Enter or Ctrl-C deny · expires in {minutes} min{RESET}"
    )
    return "\n".join(lines) + "\n"


def render_approval_resolved(ev: ApprovalResolved) -> str:
    who = f" by {ev.actor}" if ev.actor and ev.actor not in ("system", "cli") else ""
    note = (
        f" {DIM}({_safe(ev.note)}){RESET}" if ev.note and ev.outcome == "denied" else ""
    )
    return f"  {_OUTCOME.get(ev.outcome, ev.outcome)}{who}{note}  {DIM}{ev.tool_name}{RESET}\n"


def parse_approval_answer(text: str | None) -> tuple[bool, str]:
    """``y``/``yes`` approves. Anything else denies; text after ``n``/``no``
    (or any other text) becomes the note the agent sees."""
    t = (text or "").strip()
    if not t:
        return False, ""
    first, _, rest = t.partition(" ")
    if first.lower() in ("y", "yes"):
        return True, ""
    if first.lower() in ("n", "no"):
        return False, rest.strip()
    return False, t


# --- history rendering --------------------------------------------------------


def _user_text(msg: Any) -> str | None:
    if not isinstance(msg, ModelRequest):
        return None
    for part in msg.parts:
        if isinstance(part, UserPromptPart) and isinstance(part.content, str):
            return _DATE_STAMP.sub("", part.content)
    return None


def render_user(text: str) -> str:
    """A user message: a bold purple ``❯`` and light-purple text, wrapped.

    Without color (NO_COLOR / not a tty) it is a plain ``❯ text``.
    """
    if not _COLOR:
        return f"\n{PROMPT_GLYPH} {text}\n"
    body: list[str] = []
    for para in text.splitlines() or [""]:
        body += textwrap.wrap(para, max(10, _width() - 3)) or [""]
    rows = [
        (f"{USER_MARK}{PROMPT_GLYPH}{RESET} " if i == 0 else "  ")
        + f"{USER_FG}{line}{RESET}"
        for i, line in enumerate(body)
    ]
    return "\n" + "\n".join(rows) + "\n"


def print_history(history: list[Any], turns: int | None = None) -> None:
    """Print prior turns compactly: prompt, tool one-liners, answer text."""
    starts = [i for i, m in enumerate(history) if _user_text(m) is not None]
    if turns is not None and starts:
        history = history[starts[max(0, len(starts) - turns)] :]
    for msg in history:
        text = _user_text(msg)
        if text is not None:
            _out(render_user(text))
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    args = _clip(
                        str(part.args or ""), _width() - len(part.tool_name) - 10
                    )
                    _out(f"  {YELLOW}●{RESET} {part.tool_name}{DIM}({args}){RESET}\n")
                elif isinstance(part, TextPart) and part.content.strip():
                    _out(f"\n{render_markdown(_safe(part.content.strip()))}")
