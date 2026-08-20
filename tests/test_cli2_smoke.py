"""Smoke-test ui/cli2 `render` (fold events into TurnStream) and `_build_layout`.

The REPL streams both panels inside a `Live` region; here we test the two pure
pieces: event folding accumulates the right text, and `_build_layout` renders a
Markdown assistant box above a tool-status box (mirrors the coding_agent CLI).
"""
import io
import re
import sys

sys.path.insert(0, ".")

import rich.console

import ui.cli2 as c


class Cap(rich.console.Console):
    def __init__(self):
        self.buf = io.StringIO()
        super().__init__(file=self.buf, force_terminal=False, width=80)


def _visible(raw: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", raw)


def ev(*, kind, text="", tool_name="", args="", tool_text=""):
    if kind == "tool_result":
        return c.StreamEvent(kind=kind, tool_name=tool_name, text=tool_text)
    return c.StreamEvent(kind=kind, text=text, tool_name=tool_name, args=args)


def fold(events):
    out = c.TurnStream()
    for e in events:
        c.render(e, out)
    return out


def render_layout(out):
    cap = Cap()
    c.console = cap
    # Rebuild the committed layout: activity strip (dim tool lines) + answer box.
    for b in out.blocks:
        if isinstance(b, c.ToolBlock):
            cap.print(c._tool_line(b))
    cap.print(c._answer_panel(out))
    return _visible(cap.buf.getvalue())


def test_fold_accumulates_thinking_response_tools():
    out = fold([
        ev(kind="thinking", text="Let me think"),
        ev(kind="thinking", text=" hard."),
        ev(kind="text", text="Here is "),
        ev(kind="text", text="the answer."),
        ev(kind="tool_call", tool_name="search", args="{\"q\": \"x\"}"),
        ev(kind="tool_result", tool_name="search", tool_text="results"),
        ev(kind="text", text=" Done."),
    ])
    # Thinking coalesces into one block; response accumulates; tool marked done.
    think_blk = next(b for b in out.blocks if isinstance(b, c.ThinkingBlock))
    assert think_blk.text == "Let me think hard.", out.blocks
    assert "".join(out.response) == "Here is the answer. Done.", out.response
    tool_blk = next(b for b in out.blocks if isinstance(b, c.ToolBlock))
    assert tool_blk.status == "done", out.blocks  # tool marked done after result
    print("test_fold_accumulates_thinking_response_tools OK")


def test_layout_renders_markdown_and_tools():
    out = fold([
        ev(kind="text", text="**Bold** answer and *italics*."),
        ev(kind="tool_call", tool_name="search", args="{\"q\": \"x\"}"),
        ev(kind="tool_result", tool_name="search", tool_text="results"),
    ])
    raw = render_layout(out)
    # Assistant stays boxed (titled panel); answer is Markdown-rendered.
    assert "assistant" in raw, raw
    assert "**" not in raw, raw
    assert "Bold answer and italics." in raw.replace("\n", " "), raw
    # Tool is a compact box-free dim line (no "tools" box title), with done mark.
    assert "tools" not in raw, raw
    assert "search" in raw, raw
    assert "\u2713" in raw, raw
    print("test_layout_renders_markdown_and_tools OK")


def test_layout_thinking_is_absorbed_tools_interleave():
    # thinking -> tool -> thinking -> tool(done) -> response
    out = fold([
        ev(kind="thinking", text="check calendar"),
        ev(kind="tool_call", tool_name="get_calendar", args="{}"),
        ev(kind="thinking", text="found slot"),
        ev(kind="tool_call", tool_name="create_event", args="{\"t\":\"x\"}"),
        ev(kind="tool_result", tool_name="create_event"),
        ev(kind="text", text="Created it."),
    ])
    raw = render_layout(out)
    # Redesign: thinking is absorbed (no "thinking" box), tools are dim lines.
    assert "thinking" not in raw, raw
    # Both tool calls present as compact lines, no "tools" box title.
    assert "tools" not in raw, raw
    assert "get_calendar" in raw, raw
    assert "create_event" in raw, raw
    # Only the last (done) tool carries the check mark in this stream.
    assert "\u2713" in raw, raw
    assert "assistant" in raw, raw
    print("test_layout_thinking_is_absorbed_tools_interleave OK")


def test_layout_thinking_placeholder():
    out = fold([ev(kind="text", text="plain answer")])
    raw = render_layout(out)
    assert "plain answer" in raw, raw
    # No thinking box when there was no reasoning in the turn.
    assert "thinking" not in raw, raw
    print("test_layout_thinking_placeholder OK")


if __name__ == "__main__":
    test_fold_accumulates_thinking_response_tools()
    test_layout_renders_markdown_and_tools()
    test_layout_thinking_is_absorbed_tools_interleave()
    test_layout_thinking_placeholder()
    print("ALL OK")
