"""Tests for the lean degeneration guard (`src/paf/degeneration_guard.py`).

Covers the real failure mode (the DeepSeek v4.1 "thinking forever" narration
loop), near-duplicate variation, normal-thinking negative controls, and the
streaming (delta) path used by the runtime hook.
"""

from __future__ import annotations

from paf.degeneration_guard import DegenerationGuard, _jaccard


# The exact narration loop seen from deepseek-v4.1-flash at medium reasoning
# effort (blank lines were rendered between the repeated phrases).
KNOWN_REPEAT = """OK, I'll run it now.

Let me do it.

Let me run the command.

Let me go.

OK, running.

Let me do it.

Let me run the command.

Let me go.

OK, I'll run it now.

Let me do it.

Let me run the command.

Let me go.

OK, running.

Let me do it.

Let me run the command.

Let me go.
"""


def feed_lines(guard: DegenerationGuard, text: str):
    """Feed a multi-line text through `add_line`; return the first detection (or None)."""
    out = None
    for line in text.splitlines():
        result = guard.add_line(line)
        if result is not None:
            return result
    return out


def test_known_repeat_detected():
    """The exact pasted loop must trip the guard (blank lines ignored)."""
    guard = DegenerationGuard()
    result = feed_lines(guard, KNOWN_REPEAT)
    assert result is not None
    assert result.detected
    assert result.reason == "thinking_line_repeat"
    assert result.repeat_count >= 3


def test_blank_lines_do_not_pollute_or_false_positive():
    """Empty lines are skipped, so a turn with natural paragraph gaps is safe."""
    guard = DegenerationGuard()
    result = feed_lines(
        guard,
        "Let me look at the build output.\n\n\nIt failed.\n\nLet me run the formatter.\n",
    )
    assert result is None


def test_normal_thinking_no_false_positive():
    """A real, non-looping thinking turn must NOT trip the guard."""
    guard = DegenerationGuard()
    result = feed_lines(
        guard,
        "The formatting error is in specs.py line 592.\n"
        "Let me run ruff format on it.\n"
        "It should reformat cleanly.\n"
        "Then I will run the tests.\n"
        "The tests should pass after the fix.\n"
        "Now let me verify the diff.\n",
    )
    assert result is None


def test_single_repeat_not_enough():
    """One or two occurrences of a phrase is normal; three within the window trips."""
    guard = DegenerationGuard()
    result = feed_lines(
        guard,
        "Let me check the result.\n"
        "Let me check the result.\n"  # only 2x total -> safe
        "Now reviewing the output.\n",
    )
    assert result is None


def test_near_duplicate_variation_detected():
    """Wording-varied repeats (Jaccard >= threshold) count toward the loop."""
    guard = DegenerationGuard()
    result = feed_lines(
        guard,
        "Let me run the command\n"
        "Let me run that command\n"
        "Let me run a command\n"
        "Let me run the command\n"  # 3+ near-identical -> trip
        "Let me run it now\n",
    )
    assert result is not None and result.detected


def test_stream_thinking_detects_across_delta_boundaries():
    """Runtime feeds chunks that may end mid-line; the guard must still catch the loop."""
    guard = DegenerationGuard()
    # Feed the known loop as arbitrary-sized chunks (some without newlines).
    text = KNOWN_REPEAT
    result = None
    i = 0
    while i < len(text):
        chunk = text[i : i + 7]  # tiny chunks that split lines arbitrarily
        i += 7
        result = guard.stream_thinking(chunk)
        if result is not None:
            break
    assert result is not None
    assert result.detected


def test_jaccard_basic():
    assert _jaccard("let me run the command", "let me run that command") >= 0.6
    assert _jaccard("completely unrelated thing", "let me run the command") < 0.3
