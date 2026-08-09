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
    cap.print(c._build_layout(out))
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
    # Two titled panels present.
    assert "assistant" in raw, raw
    assert "tools" in raw, raw
    # Answer is Markdown-rendered: bold consumed, no literal **.
    assert "**" not in raw, raw
    assert "Bold answer and italics." in raw.replace("\n", " "), raw
    # Tool shows done mark.
    assert "\u2713" in raw, raw
    print("test_layout_renders_markdown_and_tools OK")


def test_layout_thinking_is_own_box_before_assistant_and_tools_interleave():
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
    # Thinking renders as its own titled panel, not inline.
    assert "thinking" in raw, raw
    # Box order preserved: first thinking appears before the assistant box.
    assert raw.index("thinking") < raw.index("assistant"), raw
    # Interleaved second thinking block also present, tools between them.
    assert raw.count("thinking") >= 2, raw
    assert raw.count("tools") >= 2, raw
    print("test_layout_thinking_is_own_box_before_assistant_and_tools_interleave OK")


def test_layout_thinking_placeholder():
    out = fold([ev(kind="text", text="plain answer")])
    raw = render_layout(out)
    assert "plain answer" in raw, raw
    print("test_layout_thinking_placeholder OK")


if __name__ == "__main__":
    test_fold_accumulates_thinking_response_tools()
    test_layout_renders_markdown_and_tools()
    test_layout_thinking_is_own_box_before_assistant_and_tools_interleave()
    test_layout_thinking_placeholder()
    print("ALL OK")