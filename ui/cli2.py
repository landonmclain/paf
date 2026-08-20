"""Same REPL as ui/cli.py, but streamed inside Rich panels (mirrors the
coding_agent CLI's Live layout): tool activity in a fit-width "tools" box above
the Markdown-rendered assistant answer, both redrawn per token via `Live`.

This is a drop-in alternative until it's liked, then it becomes cli.py. No
slash commands. Resume is surfaced as a banner (no viewing of old sessions).

Usage:
    python ui/cli2.py [agent_name]
    # default agent_name "assistant" (specs/assistant.yaml); use "demo_assistant"
    # (specs/demo_assistant.yaml) to poke at the lean demo profile's capabilities.
    # conversation_id defaults to cli-{agent_name}.

Needs OPENAI_API_KEY (live model). Run twice with the same conversation_id to
see the "resumed N prior messages" banner.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from paf.capabilities import PROFILES
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
    """A labeled line for one tool call (no box): `tool: name(args)`.

    Only the ``tool:`` label is colored; the call itself stays default so the
    content reads clearly.
    """
    return Text.assemble(
        ("tool: ", "yellow"),
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


def _answer_panel(out: TurnStream, streaming: bool = False) -> Panel:
    """Render the (possibly still streaming) assistant answer panel.

    When ``streaming`` is true, a live ``⠋ generating…`` spinner is shown as a
    footer *inside* the green panel so a long prose answer (or a slow model)
    always reads as alive rather than stalled. The green box itself is what the
    model wrote so far; the footer is pure UI and never part of the answer.
    """
    body: RenderableType = (
        Markdown(_safe("".join(out.response)))
        if out.response
        else Text("")
    )
    if streaming:
        body = Group(body, Text(""), Spinner("dots", text="generating…"))
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

    This replaces the earlier `vertical_overflow="visible"` approach, where the
    frame could exceed the terminal and Rich emitted every overflowing line
    into the scrollback on each repaint -- the duplicated `read_file`/`tools`
    boxes you saw when scrolling back through history.
    """

    def build(out: TurnStream, streaming: bool = False) -> Group:
        # Blank line spacer: activity (tool:/thinking:) sits apart from the
        # boxed answer (only when there is activity to separate).
        activity = _activity_strip(out)
        parts: list = [*activity, Text("")] if activity else []
        return Group(*parts, _answer_panel(out, streaming))

    out = TurnStream()
    with Live(
        _answer_panel(out, streaming=True),
        console=console,
        refresh_per_second=12,
        vertical_overflow="ellipsis",
        screen=True,
    ) as live:
        # Heartbeat: repaint the whole frame ~10x/sec for the ENTIRE turn so
        # the "generating…" spinner in the answer panel footer keeps animating
        # even while a long prose answer is mid-stream (no new tokens arriving)
        # or the model is slow to start. The stream loop mutates `out` and
        # repaints on each event; the heartbeat guarantees continuous motion
        # in between. Cancelled once the turn settles.
        running = True

        async def _heartbeat() -> None:
            while running:
                # Refresh never drops below what Live needs to keep the footer
                # spinner visibly spinning between stream events.
                live.update(build(out, streaming=True), refresh=True)
                await asyncio.sleep(0.1)

        heartbeat_task = asyncio.create_task(_heartbeat())
        try:
            async for event in session.stream(prompt, conversation_id=session.conversation_id):
                render(event, out)
                # Repaint immediately on each event; live.update is cheap after
                # the heartbeat has already been driving frames.
                live.update(build(out, streaming=True))
        finally:
            running = False
            heartbeat_task.cancel()
            # Final repaint with the spinner removed so the settled panel has
            # no stray "generating…" footer.
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
    conversation_id = f"cli-{agent_name}"

    with Session(agent_name, profile=profile, conversation_id=conversation_id) as session:
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
            if prompt.lower() in {"quit", "exit", "/q"}:
                console.print("bye")
                return 0

            # Render the user's turn as a `you` Panel, then stream the reply.
            show_prompt_box(prompt)
            console.print()

            asyncio.run(_run_streamed(session, prompt))
            console.print()  # blank line before the next prompt

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
