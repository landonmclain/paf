"""Tests for the lean degeneration guard (`src/paf/degeneration_guard.py`).

Covers the real failure mode (the DeepSeek v4.1 "thinking forever" narration
loop), the single-phrase false-positive regression (e.g. "hmm let me get the
exact line first then decide"), near-duplicate variation, normal-thinking
negative controls, and the streaming (delta) path used by the runtime hook.
"""

from __future__ import annotations

from paf.degeneration_guard import DegenerationGuard

# A real narration loop seen from deepseek-v4.1-flash at medium reasoning effort:
# a block of SEVERAL DISTINCT phrases cycling, with blank lines rendered between
# phrases. Each distinct phrase recurs `CYCLES` times; blank lines between.
CYCLES = 4
_LOOP_BLOCK = [
    "OK, I'll run it now.",
    "Let me do it.",
    "Let me run the command.",
    "Let me go.",
]
KNOWN_REPEAT = "\n\n".join("\n\n".join(_LOOP_BLOCK) for _ in range(CYCLES))


def feed_lines(guard: DegenerationGuard, text: str):
    """Feed a multi-line text through `add_line`; return first detection (or None)."""
    out = None
    for line in text.splitlines():
        result = guard.add_line(line)
        if result is not None:
            return result
    return out


def test_known_repeat_detected_with_defaults():
    """A several-phrase loop cycling 4x trips the DEFAULT guard config."""
    guard = DegenerationGuard()  # window=32, max_repeats=4, min_distinct_lines=3
    result = feed_lines(guard, KNOWN_REPEAT)
    assert result is not None
    assert result.detected
    assert result.reason == "thinking_loop"
    assert result.repeat_count >= 4


def test_single_common_phrase_does_not_trip():
    """Regression: 'hmm let me get the exact line first then decide' repeating a
    few times while thinking is NORMAL and must NOT trip the default config."""
    phrase = "hmm let me get the exact line first then decide"
    # The phrase cycles several times, but it is ONE family amid varied lines.
    text = "\n".join(
        [
            phrase,
            "ok now i need to pass the right args",
            phrase,
            "let me check the signature again",
            phrase,
            "then call it with the parsed values",
            phrase,
            "wrap it in try except to be safe",
            phrase,  # 5 total occurrences of ONE distinct phrase
        ]
    )
    guard = DegenerationGuard()
    assert feed_lines(guard, text) is None


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


def test_two_common_phrases_not_enough():
    """Two frequently-repeated phrases is still normal; needs 3 distinct lines."""
    guard = DegenerationGuard()  # min_distinct_lines defaults to 3
    result = feed_lines(
        guard,
        # two distinct phrases each repeating 4x, other lines vary -> only 2 lines
        "\n".join(
            [
                "let me check the result",
                "let me verify the output",
                "let me check the result",
                "let me verify the output",
                "let me check the result",
                "let me verify the output",
                "let me check the result",
                "let me verify the output",
                "then i will wrap it up",
            ]
        ),
    )
    assert result is None


def test_three_distinct_lines_exact_repeat_trips():
    """Three distinct phrases each recurring 4x *identically* is a loop (the real
    narration pattern repeats whole phrases verbatim, not with fuzzy variation)."""
    guard = DegenerationGuard()  # defaults
    text = "\n".join(
        [
            "Let me run the command",
            "Let me check the result",
            "Let me go",
            "Let me run the command",
            "Let me check the result",
            "Let me go",
            "Let me run the command",
            "Let me check the result",
            "Let me go",
            "Let me run the command",
            "Let me check the result",
            "Let me go",
        ]
    )
    result = feed_lines(guard, text)
    assert result is not None and result.detected
    assert result.reason == "thinking_loop"


def test_stream_thinking_detects_across_delta_boundaries():
    """Runtime feeds chunks that may end mid-line; the guard must still catch the loop."""
    guard = DegenerationGuard()
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
