"""Same REPL as ui/cli.py, but streamed inside Rich panels (mirrors the
coding_agent CLI's Live layout): tool activity in a fit-width "tools" box above
the Markdown-rendered assistant answer, both redrawn per token via `Live`.

This is a drop-in alternative until it's liked, then it becomes cli.py. Resume
is surfaced as a banner (no viewing of old sessions) — use `/history`,
`/memory`, `/debug` to inspect the session and durable memory at runtime.

Usage:
    python ui/cli2.py [agent_name]
    # default agent_name "assistant" (specs/assistant.yaml); use "demo_assistant"
    # (specs/demo_assistant.yaml) to poke at the lean demo profile's capabilities.
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

DEFAULT_AGENT = "assistant"

# conversation_id derived per agent so each agent keeps its own history lane
# (cli-assistant, cli-demo_assistant, ...).

console = Console()


def show_prompt_box(prompt: str) -> None:
    """Render the user's typed line as a Rich `Panel` matching the assistant.

    Same engine and borders as the `assistant` panel, full-width, so the user
    turn and reply read as one boxed conversation. The raw ``you > `` input
    line is erased first so the text appears only inside the box — no dupe.
    """
    # Erase the `you > <text>` input row, then redraw it boxed as the `you`
    # panel. \r returns to column 0, \x1b[K clears to end of line.
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
    """A labeled line for one tool call (no box): `✓ tool: name(args)`.

    Only the ``tool:`` label is colored; the call itself stays default so the
    content reads clearly. The status marker reflects the block's live state:
    ``✓`` (green) once the tool result has returned, ``●`` (yellow) while it is
    still in flight — so ``finish_latest_tool()`` has a visible effect.
    """
    done = block.status == "done"
    marker = "✓" if done else "●"
    return Text.assemble(
        (f"{marker} tool: ", "green" if done else "yellow"),
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


def _activity_strip(out: TurnStream, max_blocks: int | None = None) -> list[Text]:
    """Box-free labeled activity lines for thinking/tools in the live view.

    Each thinking block renders as ``thinking: <text>`` and each tool call as
    ``tool: name(args)``, in stream order. When ``max_blocks`` is given, only the
    *last* ``max_blocks`` blocks are shown (live viewport — keeps old activity
    from consuming the screen). The final committed transcript calls it with no
    cap so the full activity is retained.
    """
    blocks = out.blocks[-max_blocks:] if max_blocks else out.blocks
    lines: list[Text] = []
    for block in blocks:
        if isinstance(block, ToolBlock):
            lines.append(_tool_line(block))
        elif isinstance(block, ThinkingBlock) and block.text:
            lines.append(_thinking_line(block.text))
    return lines


def _tail_text_lines(text: str, max_lines: int) -> str:
    """Keep only the last ``max_lines`` visual lines (auto-follow viewport).

    Older lines scroll off the top; the newest tokens stay pinned at the bottom
    edge, so as new content streams in the view shifts upward naturally. Guards
    against starting mid-code-fence so the kept slice re-renders cleanly as
    Markdown (no giant orphaned code block at the top).
    """
    if max_lines < 1:
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    kept = lines[len(lines) - max_lines :]
    n = sum(1 for ln in kept if ln.strip().startswith("```"))
    if n % 2 == 1:
        # cut inside a fence opened above the viewport: drop content through the
        # first fence line to exit the orphaned block (best-effort; the final
        # commit re-renders the full answer exactly).
        for i, ln in enumerate(kept):
            if ln.strip().startswith("```"):
                kept = kept[i + 1 :]
                break
    return "\n".join(kept)


def _status_line(word: str, color: str) -> Group:
    """One live status line: colored ``<phase>…`` label beside the spinner.

    Sits at the BOTTOM of the live frame (the chat-eye anchor) while streaming.
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


def _answer_panel(out: TurnStream) -> Panel:
    """Render the (possibly still streaming) assistant answer panel."""
    body: RenderableType = (
        Markdown(_safe("".join(out.response)))
        if out.response
        else Text("")
    )
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
    """Run one turn, streaming thinking/tools/answer inside `Live` (alt screen).

    Commit-on-complete, alt-screen style: the whole turn is streamed and
    redrawn in place inside a `Live(screen=True)` alternate-screen buffer, so
    every repaint happens off the terminal's main scrollback -- a long answer
    can never leave piles of duplicated panels behind. When the turn settles,
    we exit the alt buffer and commit the final conversation ONCE to the main
    console (one clean copy of each thinking/tool panel plus the full answer).

    **Auto-follow viewport.** While streaming, the answer is NOT rendered in
    full: ``build()`` keeps only the last ``_available_answer_lines`` visual
    lines and re-renders that tail as Markdown. Because the newest tokens stay
    pinned at the bottom edge, older lines scroll off the top of the live view
    as the answer grows — genuine auto-follow. The phase status line sits at
    the BOTTOM (the chat-eye anchor), below the trimmed answer, and stays
    visible because the frame is trimmed to fit. Live activity is capped to the
    last few blocks so old thinking/tool lines can't consume the viewport.

    This replaces the earlier `vertical_overflow="visible"` approach, where the
    frame could exceed the terminal and Rich emitted every overflowing line
    into the scrollback on each repaint -- the duplicated `read_file`/`tools`
    boxes you saw when scrolling back through history. It also replaces an
    interim top-pinned status line; trimming the answer to fit is what keeps
    the bottom status (and the newest tokens) visible without relying on the
    top-edge preservation that ``vertical_overflow=\"ellipsis\"`` provides.
    """

    def _available_answer_lines(out: TurnStream) -> int:
        """Rows the answer panel body may occupy so the whole frame fits.

        Reserves rows for live activity (capped), a blank spacer, the panel
        borders + padding, and the bottom status line, then hands the rest to
        the answer. A small safety margin absorbs Markdown reflow wrapping so
        the frame essentially never exceeds the terminal (it must not: with
        ``vertical_overflow="ellipsis"`` Rich keeps the TOP and would chop the
        newest tokens otherwise). The final commit below re-renders the full
        answer, so this is purely the live-preview budget.
        """
        h = console.size.height
        activity = _activity_strip(out, max_blocks=3)
        # 1 blank spacer above the panel + 2 borders + 2 padding + bottom status
        # + breathing room (~2): conservative so wraps don't blow the budget.
        reserved = max(4, len(activity) + 1 + 2 + 2 + 1 + 2)
        return max(1, h - reserved)

    def build(out: TurnStream, streaming: bool = False) -> Group:
        parts: list = []
        # Live activity first (capped while streaming so old thinking/tool lines
        # don't consume the viewport). Order is top-down here.
        activity = _activity_strip(out, max_blocks=3 if streaming else None)
        if activity:
            parts.extend([*activity, Text("")])
        # The answer is rendered as a manageable viewport: only the LAST
        # `_available_answer_lines` visual lines are kept, so the newest tokens
        # stay pinned at the bottom edge and older lines scroll off the top —
        # genuine auto-follow. The status line sits at the very BOTTOM (the
        # chat-eye anchor), below the answer, and stays visible because the
        # frame is trimmed to fit.
        body = "\n".join(out.response)
        answer = Panel(
            Markdown(_tail_text_lines(body, _available_answer_lines(out))),
            title="assistant",
            border_style="green",
        )
        if streaming:
            word, color = _phase(out)
            status = _status_line(word, color)
            parts.append(Group(answer, Text(""), status))
        else:
            parts.append(answer)
        return Group(*parts)

    out = TurnStream()
    # Rich's Spinner advances its frame by real elapsed time
    # (`render(console.get_time())`), so even though `_status_line` rebuilds the
    # Spinner each repaint (needed to swap the phase label), it still animates
    # at the correct real speed and doesn't blink/freeze. A blank line above and
    # below gives the status line breathing room (not flush against content).

    with Live(
        build(out, streaming=True),
        console=console,
        refresh_per_second=12,
        vertical_overflow="ellipsis",
        screen=True,
    ) as live:
        # Live auto-refreshes at `refresh_per_second` (12Hz) via its own
        # background thread, so the bottom status-line spinner keeps animating
        # even while a long prose answer is mid-stream (no new tokens arriving)
        # or the model is slow to start -- no manual heartbeat task needed.
        try:
            async for event in session.stream(prompt, conversation_id=session.conversation_id):
                render(event, out)
                live.update(build(out, streaming=True))
        finally:
            # Final repaint with the status line removed so the settled view has
            # no stray "generating…" line.
            live.update(build(out, streaming=False))

    # Turn settled: commit the conversation once to the main scrollback --
    # plain labeled (colored) lines for thinking and tools, a blank spacer,
    # then the green answer box. Thinking is committed verbatim so it doesn't
    # vanish once streaming ends.
    if out.blocks:
        for b in out.blocks:
            if isinstance(b, ToolBlock):
                console.print(_tool_line(b))
            elif isinstance(b, ThinkingBlock) and b.text:
                console.print(_thinking_line(b.text))
        console.print()  # full empty line spacer before the answer
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
        profile_name = "demo" if agent_name == "demo_assistant" else "assistant"
    profile = PROFILES.get(profile_name)
    if profile is None:
        console.print(f"unknown profile {profile_name!r}; known: {sorted(PROFILES)}")
        return 2
    conversation_id = f"cli2-{agent_name}"

    with Session(agent_name, profile=profile, conversation_id=conversation_id, resume_window=15, memory_consolidation=True) as session:
        _startup_banner(session)
        console.print("type a message, or \"quit\" to exit")

        while True:
            try:
                # Plain line read (no fake box / prompt_toolkit fragments).
                # The `you` Panel is printed with the typed text once submitted,
                # so the input box appears exactly like the assistant box does.
                prompt = input("you > ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nbye")
                return 0

            if not prompt:
                console.print()
                continue
            if not prompt.startswith("/"):
                # Bare quit/exit words (legacy). Slash commands are handled
                # below via `_handle_slash`.
                if prompt.lower() in {"quit", "exit"}:
                    console.print("bye")
                    return 0

                # Render the user's turn as a `you` Panel, then stream the reply.
                show_prompt_box(prompt)
                console.print()

                asyncio.run(_run_streamed(session, prompt))
                console.print()  # blank line before the next prompt
                continue

            # Slash command: handled locally, never sent to the model. Returns
            # False when the REPL should exit (e.g. /quit).
            if not _handle_slash(session, prompt):
                return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
