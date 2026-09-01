"""Same REPL as ui/cli.py, but streamed inside Rich panels (mirrors the
coding_agent CLI's Live layout): tool activity in a fit-width "tools" box above
the Markdown-rendered assistant answer, both redrawn per token via `Live`.

This is a drop-in alternative until it's liked, then it becomes cli.py. Resume
is surfaced as a banner (no viewing of old sessions) — use `/history`,
`/memory`, `/debug` to inspect the session and durable memory at runtime.

Usage:
    python ui/cli2.py [agent_name]
    # default agent_name "demo_assistant" (specs/demo_assistant.yaml) — the
    # lean demo profile; pass "assistant" (specs/assistant.yaml) explicitly to
    # run the full M1 profile instead.
    # conversation_id defaults to cli-{agent_name}. Slash commands (/help) are
    # handled locally and never sent to the model.

Needs OPENAI_API_KEY (live model). Run twice with the same conversation_id to
see the "resumed N prior messages" banner.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from paf.capabilities import PROFILES
from paf.config import PROJECT_ROOT
from paf.runtime import Session, StreamEvent
from paf.specs import resolve_profile_name

DEFAULT_AGENT = "demo_assistant"

# Conversation-lane epoch. The CLI is a single-lane resume (one conversation id
# per agent) — bumping this number starts a BRAND-NEW empty lane, forgetting
# every prior run's restored history for that agent. Keeps each benchmark run /
# experiment arm on its own clean conversation (no resume of prior turns).
CONV_EPOCH = 15

# conversation_id derived per agent + epoch so each agent keeps its own history
# lane (cli3-assistant-e1, cli3-demo_assistant-e1, cli3-contextbench-e1, ...).

console = Console()


def show_prompt_box(prompt: str, n_lines: int = 1) -> None:
    """Render the user's typed line as a Rich `Panel` matching the assistant.

    Same engine and borders as the `assistant` panel, full-width, so the user
    turn and reply read as one boxed conversation. The raw input row(s) are
    erased first so the text appears only inside the box — no dupe. ``n_lines``
    is the number of raw ``you >``/``...`` rows that were read for a multi-line
    message; when it's > 1 we clear from the first input row down to the end of
    the screen, then draw the box below the last-painted frame.
    """
    if n_lines > 1:
        # Jump back to the first input row, then clear everything from there
        # to the end of the screen (\x1b[0J), erasing every raw prompt row.
        _write(f"\r\x1b[{n_lines - 1}A\x1b[0J")
    else:
        # Single line: \r returns to column 0, \x1b[K clears to end of line.
        _write("\r\x1b[K")
    console.print(Panel(prompt, title="you", border_style="magenta"))


def _write(s: str) -> None:
    """Write raw bytes to the terminal, bypassing Rich's buffered console.

    Used for control sequences like cursor movement that must reach the tty
    before Rich prints the next frame.
    """
    sys.stdout.write(s)
    sys.stdout.flush()


def _safe(s: str) -> str:
    """Make a model-produced string safe to print under utf-8.

    ``FunctionToolCallEvent.args`` can carry lone UTF-16 surrogate code units
    (the model streaming a partially-decoded arg), which ``print`` then blows
    up on with ``UnicodeEncodeError``. Drop unencodable chars rather than raise.
    """
    return s.encode("utf-8", errors="replace").decode("utf-8")


@dataclass
class ThinkingBlock:
    text: str


@dataclass
class ToolBlock:
    name: str
    args: str
    status: str = "\u2026"  # …


@dataclass
class TurnStream:
    """Accumulates the live turn so the panels can be redrawn as it streams.

    ``blocks`` preserves interleaved order so each model step can render as its
    own box — e.g. thinking, then a tool, then more thinking, then a tool —
    matching ``thinking -> tool -> thinking -> tool -> response``. The final
    ``response`` is kept separate so the assistant answer always renders last.
    """

    blocks: list = field(default_factory=list)  # ThinkingBlock | ToolBlock
    response: list[str] = field(default_factory=list)

    def add_thinking(self, text: str) -> None:
        """Append a thinking chunk, coalescing consecutive thinking events."""
        if self.blocks and isinstance(self.blocks[-1], ThinkingBlock):
            self.blocks[-1].text += text
        else:
            self.blocks.append(ThinkingBlock(text))

    def add_tool(self, name: str, args: str) -> None:
        self.blocks.append(ToolBlock(name, args))

    def finish_latest_tool(self) -> None:
        for block in reversed(self.blocks):
            if isinstance(block, ToolBlock) and block.status != "done":
                block.status = "done"
                return


def _tool_line(block: ToolBlock) -> Text:
    """A labeled line for one tool call, including completion state."""
    done = block.status == "done"
    marker = "✓" if done else "●"
    style = "green" if done else "yellow"
    return Text.assemble(
        (f"{marker} tool: ", style),
        f"{block.name}({block.args})",
    )


def _thinking_line(text: str) -> Text:
    """A labeled line for reasoning (no box): `thinking: <text>`.

    Only the ``thinking:`` label is colored; the reasoning text stays default.
    """
    return Text.assemble(
        ("thinking: ", "cyan"),
        text,
    )


def _activity_strip(out: TurnStream) -> list[Text]:
    """Box-free labeled activity lines for thinking/tools in the live view.

    Each thinking block renders as ``thinking: <text>`` and each tool call as
    ``tool: name(args)``, in stream order, full text (no truncation).
    """
    lines: list[Text] = []
    for block in out.blocks:
        if isinstance(block, ToolBlock):
            lines.append(_tool_line(block))
        elif isinstance(block, ThinkingBlock) and block.text:
            lines.append(_thinking_line(block.text))
    return lines




def _clip_inline(text: str, limit: int = 240) -> str:
    """Collapse whitespace and clip noisy live-only activity text."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _live_activity_strip(out: TurnStream, max_blocks: int = 5) -> list[Text]:
    """Compact tail of activity for the live viewport only.

    Each line fits on a single terminal row so the activity block has a stable
    width: tool args and thinking text are clipped to the available columns
    (``thinking: `` label eats into the thinking budget), and only the newest
    ``max_blocks`` blocks are shown. The final committed transcript still uses
    `_activity_strip`, so nothing is lost after the turn finishes.
    """
    lines: list[Text] = []
    for block in out.blocks[-max_blocks:]:
        if isinstance(block, ToolBlock):
            # One visual row: leave room for "✓ tool: name(" + ")".
            args_limit = max(10, console.size.width - len("\u2713 tool: ") - len(block.name) - 2)
            compact = ToolBlock(block.name, _clip_inline(block.args, args_limit), block.status)
            lines.append(_tool_line(compact))
        elif isinstance(block, ThinkingBlock) and block.text:
            lines.append(_thinking_line(_clip_inline(block.text, _one_line_limit("thinking: "))))
    return lines


def _one_line_limit(label: str, min_limit: int = 40) -> int:
    """Columns available for a row after the ``label``, so it fits on one line."""
    return max(min_limit, console.size.width - len(label))



def _tail_renderable(renderable: RenderableType, *, width: int, max_lines: int) -> RenderableType:
    """Render to visual terminal rows and return the NEWEST rows as a re-renderable.

    This is the key to chat-style auto-follow: instead of letting `Live` crop
    an oversized renderable from the bottom, we explicitly give it a viewport
    containing the tail of the already-wrapped/rendered answer.

    We do NOT return the raw row segments re-wrapped in a Group: Rich does not
    re-layout pre-wrapped ``Segments`` inside a Panel, so the answer would
    collapse to a single line showing the head of the text. Instead we fold
    each trailing row's segments into a plain ``Text`` (one visual line per
    row, joined by newlines). A ``Text`` already sized to the inner panel width
    is not re-wrapped, so it renders as exactly ``max_lines`` rows of the
    newest content -- clean bottom-following with no per-frame markdown
    re-parse (which is what makes cli2's live view jumpy).
    """
    if max_lines <= 0:
        return Text("")
    options = console.options.update(width=max(1, width))
    rows = console.render_lines(renderable, options=options, pad=False)
    rows = rows[-max_lines:]
    if not rows:
        return Text("")
    text = Text()
    for ri, row in enumerate(rows):
        for seg in row:
            text.append(seg.text, style=seg.style)
        if ri != len(rows) - 1:
            text.append("\n")
    return text


def _status_line(word: str, color: str) -> Group:
    """One live status line: colored ``<phase>…`` label beside the spinner.

    The label carries its own style so only the phase word is colored (matching
    the ``thinking:`` / ``tool:`` label-only coloring elsewhere); the spinner
    glyph stays default. ``Spinner`` accepts a `RenderableType` text and uses it
    as-is (no re-parsing), so the per-phase color survives.
    """
    return Group(Text(""), Spinner("dots", text=Text(f"{word}…", style=color)), Text(""))


def _phase(out: TurnStream) -> tuple[str, str]:
    """Map the accumulated turn state to a (label word, color) status line.

    * ``writing``   once the first answer token has streamed — this is the
                    ``full response has started`` signal and dominates.
    * ``using {tool}`` / ``using tools``  while tool calls are in flight (or
                    were the last thing seen). Names the most recent tool when it
                    is still the active tail block (``using read_file``); falls
                    back to the generic ``using tools`` while reasoning about a
                    result, since naming a finished tool then would be stale.
    * ``thinking``  while only reasoning has arrived.
    * ``generating`` otherwise (no events yet — the initial fallback).
    """
    if out.response:
        return "writing", "green"
    last = out.blocks[-1] if out.blocks else None
    if isinstance(last, ToolBlock):
        return f"using {last.name}", "yellow"
    if any(isinstance(b, ToolBlock) for b in out.blocks):
        return "using tools", "yellow"
    if out.blocks:
        return "thinking", "cyan"
    return "generating", "magenta"


def _answer_panel(out: TurnStream, *, max_body_lines: int | None = None) -> Panel:
    """Render the assistant answer, optionally as a bottom-following viewport."""
    body: RenderableType = (
        Markdown(_safe("".join(out.response)))
        if out.response
        else Text("")
    )
    if max_body_lines is not None and out.response:
        # Panel consumes two border columns plus padding on both sides.
        inner_width = max(20, console.size.width - 4)
        body = _tail_renderable(body, width=inner_width, max_lines=max_body_lines)
    return Panel(body, title="assistant", border_style="green")


def render(event: StreamEvent, out: TurnStream) -> None:
    """Fold one StreamEvent into ``out`` (the Live loop drives redraws)."""
    if event.kind == "thinking" and event.text:
        out.add_thinking(event.text)
    elif event.kind == "text" and event.text:
        out.response.append(event.text)
    elif event.kind == "tool_call":
        out.add_tool(event.tool_name or "<unknown>", _safe(event.args))
    elif event.kind == "tool_result":
        out.finish_latest_tool()


async def _run_streamed(session: Session, prompt: str) -> None:
    """Stream one turn in an alternate-screen, bottom-following viewport.

    `screen=True` is retained so redraws never pollute normal scrollback. While
    streaming, only a compact tail of activity and the newest visual rows of the
    answer are rendered. When the turn completes, the full activity + full answer
    are committed exactly once to the main terminal buffer.
    """

    def build(out: TurnStream, streaming: bool = False) -> Group:
        if not streaming:
            parts: list[RenderableType] = []
            activity = _activity_strip(out)
            if activity:
                parts.extend([*activity, Text("")])
            parts.append(_answer_panel(out))
            return Group(*parts)

        # Keep only the newest few activity events in the live view. The complete
        # versions are still printed after streaming finishes.
        activity = _live_activity_strip(out, max_blocks=5)

        # Measure the activity at the current terminal width so wrapped tool args
        # don't accidentally steal the answer viewport.
        activity_rows = 0
        if activity:
            activity_rows = len(
                console.render_lines(
                    Group(*activity),
                    options=console.options.update(width=console.size.width),
                    pad=False,
                )
            )

        # Budget terminal rows. The panel itself costs two rows (top/bottom), the
        # status group costs three, and spacing costs another row or two. A small
        # floor keeps the answer useful in short terminals.
        reserved = activity_rows + 7
        answer_rows = max(3, console.size.height - reserved)

        parts: list[RenderableType] = []
        if activity:
            parts.extend([*activity, Text("")])
        parts.append(_answer_panel(out, max_body_lines=answer_rows))

        # Status belongs at the bottom in a chat UI: the newest content and current
        # phase stay together near the user's eye line.
        word, color = _phase(out)
        parts.append(_status_line(word, color))
        return Group(*parts)

    out = TurnStream()

    with Live(
        build(out, streaming=True),
        console=console,
        refresh_per_second=12,
        vertical_overflow="crop",
        screen=True,
    ) as live:
        try:
            async for event in session.stream(prompt, conversation_id=session.conversation_id):
                render(event, out)
                live.update(build(out, streaming=True))
        finally:
            # Clear the transient status before leaving the alternate buffer.
            live.update(build(out, streaming=False))

    # Commit the full turn once to normal terminal scrollback.
    if out.blocks:
        for b in out.blocks:
            if isinstance(b, ToolBlock):
                console.print(_tool_line(b))
            elif isinstance(b, ThinkingBlock) and b.text:
                console.print(_thinking_line(b.text))
        console.print()
    console.print(_answer_panel(out))


def _startup_banner(session: Session) -> None:
    """Load durable history and import it into the visible conversation.

    ``session.restore()`` seeds ``session.history`` with the prior turns for
    this conversation id. We then render those prior ``you`` / ``assistant``
    panels to the main scrollback so a fresh process opens showing the actual
    conversation, not just a count. The banner confirms what was loaded.
    """
    restored = 0
    try:
        restored = asyncio.run(session.restore())
    except Exception as exc:  # noqa: BLE001 - persistence is optional sugar here
        console.print(f"(persistence/resume skipped: {exc})")
        return

    if restored:
        _render_prior_history(session.history)
        console.print(f"[cyan]resumed {restored} prior message(s)[/cyan]\n")
    else:
        console.print(f"[green]new conversation: {session.conversation_id}[/green]\n")


def _render_prior_history(history: list) -> None:
    """Print prior turns (``Session.history``) to the main scrollback.

    Keeps it to the chat skeleton -- the user's prompts and the assistant's
    final text per turn -- mirroring how the live turn renders, so the opened
    conversation reads as a continuous transcript.
    """
    from pydantic_ai.messages import TextPart, UserPromptPart

    for msg in history:
        for part in getattr(msg, "parts", []):
            if isinstance(part, UserPromptPart):
                content = part.content
                if isinstance(content, str):
                    console.print(
                        Panel(content, title="you", border_style="magenta")
                    )
            elif isinstance(part, TextPart):
                console.print(
                    Panel(
                        Markdown(_safe(part.content)),
                        title="assistant",
                        border_style="green",
                    )
                )


# ---------------------------------------------------------------------------
# Slash commands
#
# Command lines starting with `/` are intercepted here and handled locally --
# they are never sent to the model. Everything in this set is read-only:
# it inspects the session / durable memory but writes nothing and makes no
# model or tool calls, so it is safe to run at any point in a conversation.
# ---------------------------------------------------------------------------


def _memory_dir() -> Path:
    """Resolve the durable-memory directory (config-anchored, not CWD)."""
    from paf.config import settings  # env-driven; resolves memory_dir

    return PROJECT_ROOT / Path(settings.memory_dir)


def _show_history(session: Session, n: int | None) -> None:
    """Print the last ``n`` persisted turns (or all when ``n`` is None)."""
    history = session.history or []
    if not history:
        console.print("[dim]no persisted history for this conversation[/dim]")
        return
    if n is not None:
        history = history[-n:]
    console.print(f"[dim]showing {len(history)} persisted turn(s)[/dim]")
    # Reuse the chat-skeleton renderer (users + final assistant text).
    _render_prior_history(history)


def _show_memory() -> None:
    """Show what durable memory exists on disk and what's injected."""
    memdir = _memory_dir()
    root_md = memdir / "MEMORY.md"
    journal = memdir / "memory-journal.jsonl"

    console.print(f"[cyan]memory dir:[/cyan] {memdir}")

    def _file_status(path: Path) -> str:
        if not path.exists():
            return "absent"
        lines = path.read_text(encoding="utf-8", errors="replace").count("\n")
        return f"{path.stat().st_size} bytes, {lines} lines"

    console.print(f"  MEMORY.md             {_file_status(root_md)}")
    console.print(f"  memory-journal.jsonl  {_file_status(journal)}")

    if root_md.exists():
        body = root_md.read_text(encoding="utf-8", errors="replace").strip()
        if body:
            # Show the rendered journal inline so you can see what the agent
            # actually has in context right now.
            console.print("\n[cyan]--- rendered MEMORY.md (what's injected) ---[/cyan]")
            console.print(body)
        else:
            console.print("[dim]  (MEMORY.md is empty)[/dim]")
    else:
        console.print(
            "[yellow]No MEMORY.md yet — durable memory is empty. Facts only land "
            "here after the consolidation owner folds them in.[/yellow]"
        )


def _show_debug(session: Session) -> None:
    """Print session plumbing: ids, spec, model, resume, store paths."""
    table = Table(title="debug", show_header=False, box=None)
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("value", style="default")

    agent = getattr(session, "_agent", None)
    model = getattr(agent, "model", None) if agent is not None else None

    table.add_row("agent_name", str(session.agent_name))
    table.add_row(
        "profile",
        getattr(session.profile, "name", str(session.profile))
        if session.profile is not None
        else "?",
    )
    table.add_row("conversation_id", str(session.conversation_id))
    table.add_row("resume_window", str(session.resume_window))
    table.add_row("model", str(model))
    table.add_row("spec", str(PROJECT_ROOT / "specs" / f"{session.agent_name}.yaml"))
    table.add_row("memory_dir", str(_memory_dir()))
    table.add_row("restored", str(session.restored_count))

    store = getattr(session, "_store", None)
    store_dir = getattr(store, "directory", None) or getattr(store, "root", None)
    table.add_row("store", str(store_dir) if store_dir else str(store))

    console.print(table)


def _slash_help() -> None:
    rows = [
        ("/help", "this list"),
        ("/history [n]", "show last n persisted turns (default: all)"),
        ("/memory", "show durable-memory dir, files, and injected render"),
        ("/debug", "show session plumbing: ids, model, spec, store"),
        ("/quit", "exit (alias /q, exit)"),
    ]
    console.print("[cyan]multi-line input:[/cyan] paste freely; an empty line sends the message "
                  "(single-line messages: just hit Enter).")
    table = Table(title="slash commands (read-only)", box=None)
    table.add_column("cmd", style="cyan", no_wrap=True)
    table.add_column("description", style="default")
    for cmd, desc in rows:
        table.add_row(cmd, desc)
    console.print(table)


def _handle_slash(session: Session, line: str) -> bool:
    """Dispatch a ``/command`` line locally. Returns False when the REPL should exit.

    Commands never reach the model. Unknown commands print the help list.
    """
    cmd, _, arg = line.partition(" ")
    cmd = cmd.strip().lower()
    arg = arg.strip()

    if cmd in {"/quit", "/q", "/exit"}:
        console.print("bye")
        return False
    if cmd in {"/help", "/?"}:
        _slash_help()
        return True
    if cmd == "/history":
        n = None
        if arg:
            try:
                n = int(arg)
            except ValueError:
                console.print("[yellow]/history takes an optional count[/yellow]")
        _show_history(session, n)
        return True
    if cmd == "/memory":
        _show_memory()
        return True
    if cmd == "/debug":
        _show_debug(session)
        return True

    console.print(f"[red]unknown command:[/red] {line}")
    _slash_help()
    return True


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("OPENAI_API_KEY not set. Provide it to call the live model:")
        console.print("  export OPENAI_API_KEY=sk-...")
        return 2

    # Usage: python ui/cli2.py [agent_name] [profile]
    #   agent_name — spec base name (default "assistant" → specs/assistant.yaml);
    #                "demo_assistant" → specs/demo_assistant.yaml (lean demo).
    #   profile    — profile name (default: the agent's own profile by convention;
    #                demo_assistant → "demo"). The spec name and profile name
    #                differ, so they're resolved independently here.
    argv = sys.argv[1:]
    agent_name = argv[0] if argv else DEFAULT_AGENT
    profile_name = argv[1] if len(argv) > 1 else None
    if profile_name is None:
        # Default the profile from the spec's declared `profile:` key (so a
        # one-arg `python ui/cli3.py contextbench` gets the contextbench
        # profile, not a hardcoded fallback that silently changes capabilities).
        # Only specs that don't declare `profile:` fall back to the legacy
        # agent-name mapping.
        profile_name = resolve_profile_name(agent_name) or (
            "demo" if agent_name == "demo_assistant" else "assistant"
        )
    profile = PROFILES.get(profile_name)
    if profile is None:
        console.print(f"unknown profile {profile_name!r}; known: {sorted(PROFILES)}")
        return 2
    conversation_id = f"cli3-{agent_name}-e{CONV_EPOCH}"

    with Session(agent_name, profile=profile, conversation_id=conversation_id, resume_window=15, memory_consolidation=True) as session:
        _startup_banner(session)
        console.print("type a message (blank line sends; paste multi-line freely), or \"quit\" to exit")

        while True:
            try:
                # First line. A blank line here is just a newline (no message);
                # it does NOT submit an empty prompt.
                first = input("you > ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nbye")
                return 0

            # Slash commands are always single-line: never accumulate, never
            # sent to the model. Returns False when the REPL should exit.
            if first.startswith("/"):
                if not _handle_slash(session, first):
                    return 0
                continue
            # Bare quit/exit words (legacy).
            if first.lower() in {"quit", "exit"}:
                console.print("bye")
                return 0
            if not first:
                console.print()
                continue

            # Blank-line-terminated multi-line: keep reading `... ` lines until
            # the user pastes/drops an empty line, accumulating them into ONE
            # prompt (joined with \n). A bare Enter right after the first line
            # sends a single-line message, exactly as before.
            lines = [first]
            while True:
                try:
                    more = input("... ").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print("\nbye")
                    return 0
                if not more:
                    break
                lines.append(more)

            prompt = "\n".join(lines)
            # Render the user's turn as a `you` Panel, then stream the reply.
            show_prompt_box(prompt, n_lines=len(lines))
            console.print()

            asyncio.run(_run_streamed(session, prompt))
            console.print()  # blank line before the next prompt
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
