"""
degeneration_guard.py

Lean guard for detecting repetitive / stuck LLM thinking output.

Intended placement (runtime integration):
- Feed assistant THINKING text into `guard.stream_thinking(chunk)` as it streams;
  the guard accumulates chunks and emits whole normalized lines internally.
- If `stream_thinking` returns a `DegenerationResult`, abort the current
  generation and retry (see DegenerationError in runtime).

Detects the cheap, model-agnostic signal of a thinking loop: several DISTINCT
phrases each recurring too many times within a small rolling window of recent
thinking lines — e.g.

    OK, I'll run it now.
    Let me do it.
    Let me run the command.
    Let me go.
    OK, I'll run it now.
    Let me do it.
    ...

That is a "thinking forever" narration loop (saw it from DeepSeek v4.1 at low and medium
effort). The guard is intentionally minimal and fast: exact repeated-line counting
over a rolling window, no semantic classifier, no intent heuristics, no fuzzy/
near-duplicate merging (a real loop repeats phrases verbatim). It only fires when
`min_distinct_lines` DIFFERENT lines each repeat `max_repeats` times, so a single
common phrase recurring while genuinely thinking (e.g. "hmm let me get the exact
line first then decide") never trips it. If it proves ineffective or a false-positive,
remove it — it's cheap insurance.

No external dependencies beyond the stdlib.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass


@dataclass
class DegenerationResult:
    detected: bool
    reason: str = ""
    line: str = ""
    repeat_count: int = 0


_LINE_RE = re.compile(r"[^\w\s]")  # strip punctuation for word/token comparison


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation -> {"ok running"} == {"OK, running."}."""
    text = text.lower().strip()
    text = _LINE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class DegenerationGuard:
    def __init__(
        self,
        *,
        window: int = 32,
        max_repeats: int = 4,
        min_distinct_lines: int = 3,
    ) -> None:
        """State for one generation.

        Args:
            window: how many recent thinking lines to consider.
            max_repeats: a line only counts once it recurs this many times within
                `window` recent lines. Raised from 3 because real thinking repeats
                common meta-phrases a few times.
            min_distinct_lines: require THIS many *distinct* lines to each be
                over-repeating before declaring degeneration. A real loop is a
                block of several different phrases cycling (e.g. "ok i'll run it
                now" / "let me do it" / "let me run the command" / "let me go");
                a single frequently-repeated phrase while thinking is normal.
        """
        self.window = window
        self.max_repeats = max_repeats
        self.min_distinct_lines = min_distinct_lines

        self.lines: deque[str] = deque(maxlen=window)
        self._buf: str = ""  # streaming accumulation, flushed on newline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stream_thinking(self, chunk: str) -> DegenerationResult | None:
        """Feed a streaming THINKING delta; return a result if degeneration detected.

        Call with each model-streamed thinking chunk (which may or may not end
        mid-line). Accumulates chunks and evaluates each COMPLETE line as it
        arrives (chunks can span arbitrary boundaries). Returns None while no
        degeneration is detected.
        """
        self._buf += chunk
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            result = self.add_line(line)
            if result is not None:
                return result
        return None

    def add_line(self, line: str) -> DegenerationResult | None:
        """Evaluate one complete thinking line; return a result if degeneration.

        Public for testability (feed whole lines off the hot path). Returns None
        while no degeneration is detected.

        Detection (exact-match only, deliberately simple): a real narration loop
        repeats whole phrases *identically*. We count how many DISTINCT lines each
        recur >= `max_repeats` times within the window, and only call it a loop
        when >= `min_distinct_lines` distinct lines are doing so. A single common
        phrase recurring a few times is normal thinking; a block of several
        different phrases each cycling is a loop. No fuzzy/near-duplicate merging:
        it is fragile for short phrases and the real loop repeats verbatim anyway.
        """
        normalized = _normalize(line)
        if not normalized:
            return None

        # Count occurrences of each distinct normalized line (window + this one).
        counts: Counter[str] = Counter(self.lines)
        counts[normalized] += 1

        # Distinct lines that have recurred >= max_repeats times.
        over = [ln for ln, c in counts.items() if c >= self.max_repeats]

        if len(over) >= self.min_distinct_lines:
            return DegenerationResult(
                detected=True,
                reason="thinking_loop",
                line=f"{len(over)} distinct repeating lines",
                repeat_count=max(counts[ln] for ln in over),
            )

        self.lines.append(normalized)
        return None

    def reset(self) -> None:
        """Reset all state (start of a fresh generation)."""
        self.lines.clear()
        self._buf = ""


# ----------------------------------------------------------------------
# Example integration
# ----------------------------------------------------------------------
#
# from paf.degeneration_guard import DegenerationGuard
#
# guard = DegenerationGuard()
#
# async for event in stream:
#     if isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
#         result = guard.stream_thinking(event.delta.content_delta or "")
#         if result is not None:
#             raise DegenerationError(result)   # -> runtime's mid-stream handler
#
# Recommended starting values:
#
#     DegenerationGuard(window=32, max_repeats=4, min_distinct_lines=3)
#
# Rare by design: a single common phrase repeating a few times is normal
# thinking; only when SEVERAL DISTINCT phrases each cycle >= max_repeats times do
# we call it a loop. A real narration loop is a block of different phrases
# cycling (ok i'll run it now / let me do it / let me run the command / let me
# go). Cost is negligible (tiny-line set ops over a 32-entry window per thinking
# line); if it false-positives or under-fires, tune thresholds or remove it.
