"""Terminal rendering for the chat CLI (paf.cli._render)."""

from __future__ import annotations

import os

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from paf.cli import _render
from paf.cli._render import (
    MarkdownStream,
    TurnPrinter,
    _block_end,
    _rows,
    print_history,
)
from paf.runtime import StreamEvent, ToolError


def _call(name: str, step: str, args: str = "{}") -> StreamEvent:
    return StreamEvent(kind="tool_call", tool_name=name, args=args, step_id=step)


def _result(name: str, step: str) -> StreamEvent:
    return StreamEvent(kind="tool_result", tool_name=name, step_id=step)


# --- block detection ---------------------------------------------------------


def test_block_end_waits_for_a_complete_line() -> None:
    assert _block_end("hello wor") is None
    assert _block_end("hello\n") is None  # paragraph still open


def test_block_end_at_blank_line() -> None:
    text = "para one\nstill one\n\nnext"
    assert text[: _block_end(text)] == "para one\nstill one\n"


def test_block_end_keeps_blank_lines_inside_a_fence() -> None:
    text = "```py\na = 1\n\nb = 2\n```\nafter\n"
    assert text[: _block_end(text)] == "```py\na = 1\n\nb = 2\n```\n"


def test_text_before_a_fence_is_its_own_block() -> None:
    text = "intro\n```\ncode\n```\n"
    assert text[: _block_end(text)] == "intro\n"


def test_rows_counts_wrapped_lines() -> None:
    assert _rows("abc", 10) == 0  # cursor still on the first row
    assert _rows("abc\n", 10) == 1
    assert _rows("x" * 25, 10) == 2  # 3 rows, cursor on the last


# --- markdown stream ---------------------------------------------------------


def test_disabled_stream_passes_text_through(capsys: pytest.CaptureFixture) -> None:
    md = MarkdownStream(enabled=False)
    md.feed("**bold** text")
    md.finish()
    assert capsys.readouterr().out == "**bold** text\n"


def test_finished_block_is_erased_and_rendered(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(_render, "_COLOR", True)
    monkeypatch.setattr(
        _render.shutil, "get_terminal_size", lambda *_: os.terminal_size((80, 40))
    )
    md = MarkdownStream(enabled=True)
    md.feed("**bold** first\n")
    assert capsys.readouterr().out == "**bold** first\n"  # streamed raw first
    md.feed("\nsec")  # the blank line finishes the block
    out = capsys.readouterr().out
    assert out.startswith("\r\x1b[1A\x1b[J")  # erased in place (one row up)
    assert "**bold**" not in out and "bold" in out  # re-printed rendered
    assert out.endswith("sec")  # the next block streams raw
    md.finish()
    assert "sec" in capsys.readouterr().out


# --- turn printer -------------------------------------------------------------


def test_tool_success_marks_its_own_line(capsys: pytest.CaptureFixture) -> None:
    p = TurnPrinter(show_thinking=True, markdown=False)
    for ev in (_call("read_file", "s1"), _result("read_file", "s1")):
        p.feed(ev)
    p.finish()
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["  ● read_file({}) ✓"]


def test_tool_error_marks_its_own_line(capsys: pytest.CaptureFixture) -> None:
    p = TurnPrinter(show_thinking=True, markdown=False)
    for ev in (
        _call("read_file", "s1"),
        _result("read_file", "s1"),
        ToolError(step_id="s1", error="no such file"),
    ):
        p.feed(ev)
    p.finish()
    assert capsys.readouterr().out.splitlines() == ["  ● read_file({}) ✗ no such file"]


def test_parallel_calls_to_one_tool_match_by_step_id(
    capsys: pytest.CaptureFixture,
) -> None:
    """The failing call is reported, not the first call with the same name."""
    p = TurnPrinter(show_thinking=True, markdown=False)
    for ev in (
        _call("read_file", "s1", '{"path": "a"}'),
        _call("read_file", "s2", '{"path": "b"}'),
        _result("read_file", "s1"),
        _result("read_file", "s2"),
        ToolError(step_id="s2", error="boom"),
    ):
        p.feed(ev)
    p.finish()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == '  ● read_file({"path": "a"})'
    assert lines[1] == '  ● read_file({"path": "b"}) ✗ boom'


def test_thinking_can_be_hidden(capsys: pytest.CaptureFixture) -> None:
    p = TurnPrinter(show_thinking=False, markdown=False)
    p.feed(StreamEvent(kind="thinking", text="pondering"))
    p.feed(StreamEvent(kind="text", text="answer"))
    p.finish()
    out = capsys.readouterr().out
    assert "pondering" not in out
    assert out.strip() == "answer"


def test_history_strips_the_date_stamp(capsys: pytest.CaptureFixture) -> None:
    history = [
        ModelRequest(parts=[UserPromptPart(content="[date: 2026-10-07T09:00] hi")]),
        ModelResponse(parts=[TextPart(content="hello back")]),
    ]
    print_history(history)
    out = capsys.readouterr().out
    assert "❯ hi" in out
    assert "[date:" not in out
    assert "hello back" in out


# --- pausing for input (steering) --------------------------------------------


def _color_term(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_render, "_COLOR", True)
    monkeypatch.setattr(
        _render.shutil, "get_terminal_size", lambda *_: os.terminal_size((80, 40))
    )


def test_break_block_carries_an_open_fence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Text after a pause inside a code block still renders as code."""
    _color_term(monkeypatch)
    md = MarkdownStream(enabled=True)
    md.feed("```py\nx = 1\n")
    md.break_block()
    first = capsys.readouterr().out
    assert first.endswith("\n")  # cursor at column 0 for the input line
    assert "x = 1" in first and "```" not in first.split("\x1b[J", 1)[1]
    md.feed("y = 2\n")
    assert capsys.readouterr().out == "y = 2\n"  # carried opener never printed raw
    md.feed("```\n\n")  # the fence closes after the pause
    out = capsys.readouterr().out
    assert out.startswith("\r\x1b[1A\x1b[J")  # erases only the row it printed
    assert "y = " in out and "```" not in out  # rendered as code


def test_break_block_outside_a_fence_renders_what_arrived(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _color_term(monkeypatch)
    md = MarkdownStream(enabled=True)
    md.feed("a **half** written")
    md.break_block()
    out = capsys.readouterr().out
    assert "**" not in out.split("\x1b[J", 1)[1] and out.endswith("\n")
    assert md.tail == md.shown == ""


def test_pause_mid_tool_line_and_thinking(capsys: pytest.CaptureFixture) -> None:
    p = TurnPrinter(show_thinking=True, markdown=False)
    p.feed(StreamEvent(kind="thinking", text="hmm"))
    p.pause()
    p.feed(StreamEvent(kind="thinking", text="more"))
    p.feed(_call("ls", "s1"))
    p.pause()
    p.feed(_result("ls", "s1"))
    p.finish()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "thinking: hmm"
    assert "thinking: more" in lines  # relabelled after the pause
    assert "  ● ls({})" in lines  # line closed by the pause, no stray ✓ later
    assert not any("✓" in ln for ln in lines)
