"""Full-screen Textual REPL for paf — the designed-for-this replacement for
``ui/cli2.py``'s Rich ``Live`` streaming.

Why Textual over Rich ``Live``:
  * Rich ``Live`` re-renders one growing frame. When the frame outgrows the
    terminal it either pollutes scrollback (``vertical_overflow="visible"``)
    or clips the stream out of view (alt-screen) — you can't have both a
    clean, non-repeating transcript AND watch the answer stream down. Rich was
    never built to solve that.
  * Textual owns the whole screen. History lives in a scrollable pane, the
    streaming answer is a single widget updated in place, and the input box is
    pinned at the bottom — so you can follow the stream AND scroll back, with
    zero terminal-scrollback pollution.

This is a drop-in sibling to ``ui/cli2.py`` (same agent/profile args), not a
replacement of it — cli2.py stays until this is liked.

Usage:
    python ui/cli_textual.py [agent_name] [profile]
    # default agent "assistant"; "demo_assistant" → demo profile (lean demo).
    # Needs OPENAI_API_KEY / .env (live model).
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field

from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog, Static

from paf.capabilities import PROFILES
from paf.runtime import Session, StreamEvent

DEFAULT_AGENT = "assistant"


# ---------------------------------------------------------------------------
# Turn state — the same model cli2.py uses, kept local so the two UIs stay
# independent. `blocks` preserves interleaved thinking→tool order; `response`
# is the streaming assistant answer rendered last.
# ---------------------------------------------------------------------------
@dataclass
class ThinkingBlock:
    text: str


@dataclass
class ToolBlock:
    name: str
    args: str
    status: str = "\u2026"  # …

    @property
    def done(self) -> bool:
        return self.status == "done"


@dataclass
class TurnStream:
    blocks: list = field(default_factory=list)  # ThinkingBlock | ToolBlock
    response: list[str] = field(default_factory=list)

    def add_thinking(self, text: str) -> None:
        if self.blocks and isinstance(self.blocks[-1], ThinkingBlock):
            self.blocks[-1].text += text
        else:
            self.blocks.append(ThinkingBlock(text))

    def add_tool(self, name: str, args: str) -> None:
        self.blocks.append(ToolBlock(name, args))

    def finish_latest_tool(self) -> None:
        for block in reversed(self.blocks):
            if isinstance(block, ToolBlock) and not block.done:
                block.status = "done"
                return


def _safe(s: str) -> str:
    """Make a model-produced string safe to print under utf-8.

    Streaming tool args / text can carry lone UTF-16 surrogate code units that
    break Rich rendering; drop unencodable chars rather than raise.
    """
    return s.encode("utf-8", errors="replace").decode("utf-8")


def _block_panel(block: ThinkingBlock | ToolBlock) -> Panel:
    """One thinking or tool step as a full-width panel (same look as cli2)."""
    if isinstance(block, ThinkingBlock):
        return Panel(
            Text(block.text, style="dim"), title="thinking", border_style="cyan"
        )
    mark = "\u2713" if block.done else "\u2026"
    return Panel(
        Text(f"  {mark} {block.name}({block.args})", style="dim"),
        title="tools",
        border_style="blue",
    )


def _answer_panel(out: TurnStream) -> Panel:
    """The (possibly still streaming) assistant answer panel."""
    answer = (
        Markdown(_safe("".join(out.response)))
        if out.response
        else Text("")
    )
    return Panel(answer, title="assistant", border_style="green")


def render(event: StreamEvent, out: TurnStream) -> None:
    """Fold one StreamEvent into ``out`` (the worker drives pane updates)."""
    if event.kind == "thinking" and event.text:
        out.add_thinking(event.text)
    elif event.kind == "text" and event.text:
        out.response.append(event.text)
    elif event.kind == "tool_call":
        out.add_tool(event.tool_name or "<unknown>", _safe(event.args))
    elif event.kind == "tool_result":
        out.finish_latest_tool()


def _render_prior_history(history: list, pane: RichLog) -> int:
    """Render prior turns (``Session.history``) into the history pane.

    Keeps it to the chat skeleton -- the user's prompts and the assistant's
    final text per turn -- so the pane is a readable transcript rather than a
    dump of every tool call. Returns how many messages were rendered.
    """
    from pydantic_ai.messages import TextPart, UserPromptPart

    n = 0
    for msg in history:
        for part in getattr(msg, "parts", []):
            if isinstance(part, UserPromptPart):
                content = part.content
                if isinstance(content, str):
                    pane.write(Panel(content, title="you", border_style="magenta"))
                    n += 1
            elif isinstance(part, TextPart):
                pane.write(Panel(part.content, title="assistant", border_style="green"))
                n += 1
    return n


class PafCliApp(App):
    """Full-screen REPL: scrollable history pane + pinned input line."""

    TITLE = "paf"
    SUB_TITLE = "pydantic-agent-framework"
    CSS = """
    Screen {
        layout: vertical;
    }
    RichLog {
        height: 1fr;
        border: round $primary;
        padding: 1;
    }
    #answer {
        height: auto;
        max-height: 40%;
        border: round $success;
        padding: 1;
        margin-top: 1;
        overflow-y: auto;
    }
    Input {
        margin-top: 1;
        margin-bottom: 1;
    }
    """

    def __init__(self, agent_name: str, profile_name: str) -> None:
        super().__init__()
        self._agent_name = agent_name
        self._profile_name = profile_name
        # A running turn is exclusive: we ignore new submits until it drains.
        self._busy = asyncio.Event()
        self._out = TurnStream()
        # Number of blocks already flushed to the RichLog (append-only).
        self._committed = 0
        self._history: RichLog | None = None
        self._answer: Static | None = None
        # Printed prior-turn history once (on the first submit) so a fresh app
        # opens showing the persistent conversation, not a blank pane.
        self._history_loaded = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        self._history = RichLog(highlight=True, markup=True, auto_scroll=True)
        self._history.border_title = self._agent_name
        yield self._history
        self._answer = Static("", id="answer")
        yield self._answer
        yield Input(placeholder="type a message, or \"quit\" to exit  (submit ⏎)")
        yield Footer()

    def on_mount(self) -> None:
        self._busy.set()  # start idle
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.run_worker(self._on_submit(event.value), exclusive=True)
        event.input.clear()

    # -- rendering ----------------------------------------------------------
    def _flush_completed(self) -> None:
        """Move finished blocks from the live view into the committed RichLog.

        A block is finished once a *later* block has started, or when it is a
        tool whose result has arrived. The trailing still-growing block (an
        in-flight thinking or a pending tool) stays in the live view so it can
        update in place. The RichLog is append-only -- never cleared -- so the
        `you` panel and every completed block persist exactly once.
        """
        n = len(self._out.blocks)
        for i in range(self._committed, n):
            is_last = i == n - 1
            b = self._out.blocks[i]
            finished = not is_last or (
                isinstance(b, ToolBlock) and b.done
            ) or (isinstance(b, ThinkingBlock) and self._out.response)
            if not finished:
                break
            self._history.write(_block_panel(b))  # type: ignore[union-attr]
            self._committed = i + 1
        self._history.scroll_end(animate=False)  # type: ignore[union-attr]

    def _live(self) -> Group:
        """The live view: the still-growing trailing block + streaming answer."""
        panels = []
        if self._committed < len(self._out.blocks):
            panels.append(_block_panel(self._out.blocks[-1]))
        panels.append(_answer_panel(self._out))
        return Group(*panels)

    def _follow_answer(self) -> None:
        """Keep the streaming answer scrolled to its bottom, unless the user is
        reading earlier content (then let them stay put)."""
        a = self._answer
        if a is not None and a.is_vertical_scroll_end:
            a.scroll_end(animate=False)

    async def _on_submit(self, prompt: str) -> None:
        prompt = prompt.strip()
        if not prompt:
            return
        if prompt.lower() == "quit":
            self.exit()
            return

        await self._busy.wait()  # wait for any in-flight turn to finish
        self._busy.clear()

        # Fresh turn: reset state, echo the user's line into the history log.
        self._out = TurnStream()
        self._committed = 0
        self._history.write(Panel(prompt, title="you", border_style="magenta"))  # type: ignore[union-attr]
        assert self._answer is not None  # compose() assigns it before mount completes
        self._answer.update(self._live())

        try:
            with Session(
                self._agent_name,
                profile=PROFILES[self._profile_name],
                conversation_id=f"cli-{self._agent_name}",
            ) as session:
                # First turn of a fresh app: seed the pane with the persistent
                # conversation so the user sees history, not a blank screen.
                if not self._history_loaded:
                    try:
                        await session.restore(conversation_id=session.conversation_id)
                        _render_prior_history(session.history, self._history)  # type: ignore[arg-type]
                    except Exception:  # noqa: BLE001 - history display is sugar
                        pass
                    self._history_loaded = True
                    self._history.scroll_end(animate=False)  # type: ignore[union-attr]
                async for event in session.stream(
                    prompt, conversation_id=session.conversation_id
                ):
                    render(event, self._out)
                    self._flush_completed()
                    self._answer.update(self._live())
                    self._follow_answer()
        finally:
            # Flush any trailing block, then land the final answer permanently.
            self._flush_completed()
            assert self._answer is not None
            self._answer.update(_answer_panel(self._out))
            self._follow_answer()
            self._busy.set()
            self.query_one(Input).focus()


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Provide it to call the live model:")
        print("  export OPENAI_API_KEY=sk-...")
        return 2

    argv = sys.argv[1:]
    agent_name = argv[0] if argv else DEFAULT_AGENT
    profile_name = argv[1] if len(argv) > 1 else None
    if profile_name is None:
        profile_name = "demo" if agent_name == "demo_assistant" else "assistant"
    if profile_name not in PROFILES:
        print(f"unknown profile {profile_name!r}; known: {sorted(PROFILES)}")
        return 2

    PafCliApp(agent_name, profile_name).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
