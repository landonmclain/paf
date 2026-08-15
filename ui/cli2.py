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

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
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


def _build_layout(out: TurnStream) -> Group:
    """Build the live-turn panels, one box per model step.

    Each thinking/tool segment renders as its own full-width panel in the
    order it occurred (``thinking -> tool -> thinking -> tool``), then the
    final ``assistant`` answer as a full-width Markdown panel.
    """
    # One full-width panel per interleaved thinking/tool step.
    step_panels = []
    for block in out.blocks:
        if isinstance(block, ThinkingBlock):
            step_panels.append(
                Panel(Text(block.text, style="dim"), title="thinking", border_style="cyan")
            )
        elif isinstance(block, ToolBlock):
            mark = "\u2713" if block.status == "done" else "\u2026"
            step_panels.append(
                Panel(
                    Text(f"  {mark} {block.name}({block.args})", style="dim"),
                    title="tools",
                    border_style="blue",
                )
            )
    if out.response:
        answer = Markdown(_safe("".join(out.response)))
    else:
        answer = Text("")
    return Group(*step_panels, Panel(answer, title="assistant", border_style="green"))


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
    """Run one turn, streaming BOTH panels via `Live`.

    Live is non-transient by default, so the last frame (the finished panels)
    stays on screen after the `with` exits — no vanishing like the naive
    coding_agent version; no duplicate like a separate inline stream + box.
    """
    out = TurnStream()
    with Live(
        _build_layout(out),
        console=console,
        refresh_per_second=12,
        vertical_overflow="visible",
    ) as live:
        async for event in session.stream(prompt, conversation_id=session.conversation_id):
            render(event, out)
            live.update(_build_layout(out))


def _startup_banner(session: Session) -> None:
    """Show whether this conversation resumes durable history."""
    try:
        restored = asyncio.run(session.restore())
    except Exception as exc:  # noqa: BLE001 - persistence is optional sugar here
        console.print(f"(persistence/resume skipped: {exc})")
        return
    if restored:
        console.print(f"[cyan]resumed {restored} prior message(s)[/cyan]")
    else:
        console.print(f"[green]new conversation: {session.conversation_id}[/green]")


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
