"""
degeneration_guard.py

Lean guard for detecting repetitive / stuck LLM thinking output.

Intended placement (runtime integration):
- Feed assistant THINKING text into `guard.stream_thinking(chunk)` as it streams;
  the guard accumulates chunks and emits whole normalized lines internally.
- If `stream_thinking` returns a `DegenerationResult`, abort the current
  generation and retry (see DegenerationError in runtime).

Detects the cheap, model-agnostic signal of a thinking loop: the same (or
near-identical) line recurring too many times within a small rolling window of
recent thinking lines — e.g.

    OK, I'll run it now.
    Let me do it.
    Let me run the command.
    Let me go.
    OK, I'll run it now.
    Let me do it.
    ...

That is a "thinking forever" narration loop (saw it from DeepSeek v4.1 at low and medium
effort). The guard is intentionally minimal and fast: rolling repeated-/-near-
duplicate-line detection, no semantic classifier, no intent heuristics. If it
proves ineffective or a false-positive, remove it — it's cheap insurance.

No external dependencies beyond the stdlib.
"""

from __future__ import annotations

import re
from collections import deque
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


def _jaccard(a: str, b: str) -> float:
    """Token-Jaccard similarity — cheap near-duplicate check.

    "let me run the command" vs "let me run that command" score highly without
    embeddings. Operates on tiny (few-token) thinking lines, so the set work is
    negligible.
    """
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class DegenerationGuard:
    def __init__(
        self,
        *,
        window: int = 16,
        max_repeats: int = 3,
        similarity_threshold: float = 0.6,
    ) -> None:
        """NaN guard state for one generation.

        Args:
            window: how many recent thinking lines to compare a new line against.
            max_repeats: trip when a line recurs (exact or near-dup) this many
                times within `window` recent lines.
            similarity_threshold: Jaccard at/above which two lines count as the
                same (near-duplicate). 0.6 catches wording-varied narration like
                "let me run the command" vs "let me run that command" (0.67)
                while still separating genuinely different lines (<0.3).
        """
        self.window = window
        self.max_repeats = max_repeats
        self.similarity_threshold = similarity_threshold

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
        """
        normalized = _normalize(line)
        if not normalized:
            return None

        # Count how many recent lines this one duplicates (exact or near-dup).
        match = 1  # itself
        for prev in self.lines:
            if _jaccard(normalized, prev) >= self.similarity_threshold:
                match += 1
                if match >= self.max_repeats:
                    return DegenerationResult(
                        detected=True,
                        reason="thinking_line_repeat",
                        line=normalized,
                        repeat_count=match,
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
#     DegenerationGuard(window=16, max_repeats=3, similarity_threshold=0.6)
#
# Rare by design: a normal turn may repeat a phrase once or twice in thinking,
# but the same line (or near-dup) recurring 3x within 16 lines is a genuine loop.
# Cost is negligible (tiny-line set ops over a 16-entry window per thinking line);
# if it false-positives or under-fires, tune thresholds or remove the guard.
