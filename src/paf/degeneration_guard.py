"""
degeneration_guard.py

Lean guard for detecting repetitive / stuck LLM thinking output.

Two guards live here:
- :class:`DegenerationGuard` — the original thinking-narration loop detector.
  Watches assistant THINKING text only.
- :class:`ToolCallGuard` — a repeat-tool-call detector (the same tool called
  with the same UI-identically-rendered args N times). Watches completed tool
  calls; complements the thinking guard because a stuck loop often shows up as
  the model re-issuing the same tool call rather than narrating in a loop.

DegenerationGuard detects the cheap, model-agnostic signal of a thinking loop:
several DISTINCT phrases each recurring too many times within a small rolling
window of recent thinking lines — e.g.

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

Only external dependency: `pydantic_ai.exceptions.UnexpectedModelBehavior` (the
base of :class:`DegenerationError`, so the runtime's existing mid-stream failure
handler still catches it). The detectors themselves use only the stdlib.
"""

from __future__ import annotations

import json
import re
from collections import Counter, deque
from dataclasses import dataclass

from pydantic_ai.exceptions import UnexpectedModelBehavior


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
def _scalar(v):
    """Render one dict value to a stable string, matching how a UI row would show it."""
    return (
        v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)
    )


def _humanize_args(args) -> str:
    """Canonical humanized form of a tool-call args payload.

    Mirrors the paf-ui `ToolBlock` display identity (`humanizeArgs`, PafChat.tsx)
    so a "repeat" here means "the UI would show N identical tool rows". One
    deliberate difference from the frontend: object keys are SORTED, whereas the
    UI preserves insertion order — so two semantically-identical calls whose args
    keys arrive in a different order still collapse to ONE key server-side (the
    user would still read them as the same call).
    """
    if args is None:
        return ""
    obj = args
    if isinstance(args, str):
        try:
            obj = json.loads(args)
        except Exception:
            return args  # not JSON — show as-is, like the UI
    if isinstance(obj, list):
        return ", ".join(map(str, obj))
    if isinstance(obj, dict):
        return ", ".join(f"{k}: {_scalar(v)}" for k, v in sorted(obj.items()))
    return str(obj)


def _tool_key(tool_name: str, args) -> str:
    """Canonical identity of a tool call = `toolName(humanized_args)`.

    This is exactly the string identity the paf-ui `ToolBlock` trigger renders
    (`WrenchIcon` + `tool.toolName` + `(humanizeArgs(args))`), so server-side
    detection and the UI's visible grouping agree by construction.
    """
    return f"{tool_name}({_humanize_args(args)})"


class ToolCallGuard:
    """Detect a tool-call repeat loop: the same tool called with the same (UI
    identically-rendered) args ``max_repeats`` times CONSECUTIVELY.

    Complements :class:`DegenerationGuard` (which watches only THINKING text):
    a stuck loop very often manifests as the *model re-issuing the same completed
    tool call* back-to-back (e.g. the same `search_files(path=…, pattern=svg)`
    ~10x) rather than a thinking-narration loop — and the thinking guard never
    sees tool calls.

    Consecutive-only by design: the count resets to 1 the moment the
    `(tool_name, humanized-args)` identity changes, so a call spread across a
    turn among other calls never accumulates. We count "N identical calls in a
    row", which is exactly the stuck-loop signature and avoids false positives
    from legitimately re-using the same call later in a turn.
    """

    def __init__(self, *, max_repeats: int = 5) -> None:
        """State for one generation (a fresh guard per turn).

        Args:
            max_repeats: the SAME tool+args must be issued this many times
                consecutively before we call it a loop. Default 5 (the observed
                real loop hit ~10x; a legitimate turn almost never issues the
                *identical* tool call 5 times back-to-back).
        """
        self.max_repeats = max_repeats
        self._last: str | None = None
        self._count = 0

    def add_tool_call(self, tool_name: str, args) -> DegenerationResult | None:
        """Evaluate one COMPLETED tool call; return a result if it's a loop.

        Feed each completed `ToolCallPart` (from a `CallToolsNode.model_response`,
        not the streaming fragments). The consecutive-run count increments when
        the call matches the previous identity, else resets to 1. Returns None
        while below threshold.
        """
        key = _tool_key(tool_name, args)
        if key == self._last:
            self._count += 1
        else:
            self._last = key
            self._count = 1
        if self._count >= self.max_repeats:
            return DegenerationResult(
                detected=True,
                reason="repeated_tool_call",
                line=key,
                repeat_count=self._count,
            )
        return None

    def reset(self) -> None:
        """Reset all state (start of a fresh generation)."""
        self._last = None
        self._count = 0


class DegenerationError(UnexpectedModelBehavior):
    """Raised when a degeneration guard detects a loop mid-generation.

    Subclasses :class:`~pydantic_ai.exceptions.UnexpectedModelBehavior` so the
    main runtime's existing mid-stream failure handler
    (``except (ModelHTTPError, UnexpectedModelBehavior)``) preserves the partial
    messages and surfaces a clean error+retry. For background/direct
    ``agent.run`` call sites it simply propagates as a per-job failure.
    """

    def __init__(self, result) -> None:
        self.result = result
        super().__init__(
            f"degeneration guard: {result.reason!r} repeated {result.repeat_count}x "
            f"({result.line!r})"
        )


async def run_guarded(
    agent,
    prompt,
    *,
    thinking_guard: DegenerationGuard | None = None,
    tool_guard: ToolCallGuard | None = None,
):
    """Run a pydantic-ai agent under the degeneration guards, mirroring the main
    runtime's `Session.stream` wiring for background/direct `agent.run` call
    sites (thread-summarizer, memory owner, vision) that never stream.

    Streams the agent via `agent.iter(prompt)` (the same graph execution
    `agent.run` uses internally, just exposed) so we can watch thinking deltas
    and completed tool calls, and returns the COMPLETE `AgentRunResult` — so
    callers can keep using `result.output` and `result.all_messages()` exactly
    as they would after `await agent.run(prompt)`.

    Guard policy (per call site, both optional):
    - Thinking guard is ON by default — it only fires on a pure thinking
      narration loop and is safe on every background agent.
    - Tool-call guard is OFF by default — the tool-using background agents
      (the memory owner) deliberately repeat their own tool calls as the job
      (the owner reads `read_memory`/`search_memory` in a loop with its own
      `_guard_repeat` guard). Apply it only where a stuck repeat-tool loop is the actual risk.

    On detection raises :class:`DegenerationError`. Background callers already handle
    per-job exceptions (e.g. the thread summarizer's `on_complete(False, ...)` +
    re-raise), so a guarded abort surfaces as a job failure and the block is
    retried next cadence — it never burns the token budget on the runaway.
    """
    from pydantic_ai.agent import CallToolsNode, ModelRequestNode
    from pydantic_ai.messages import (
        PartDeltaEvent,
        ThinkingPartDelta,
        ToolCallPart,
    )

    think = thinking_guard or DegenerationGuard()
    tool_g = tool_guard or ToolCallGuard()
    async with agent.iter(prompt) as run:
        async for node in run:
            # Only request/tool nodes stream parts; other graph nodes
            # (e.g. UserPromptNode) have no .stream() — mirror runtime.py.
            if not isinstance(node, (ModelRequestNode, CallToolsNode)):
                continue
            # Completed tool calls -> tool-loop guard (opt-in).
            if isinstance(node, CallToolsNode):
                for part in node.model_response.parts:
                    if isinstance(part, ToolCallPart):
                        tg = tool_g.add_tool_call(part.tool_name, part.args)
                        if tg is not None:
                            raise DegenerationError(tg)
            # Streaming thinking deltas -> thinking-loop guard (default on).
            async with node.stream(run.ctx) as stream:
                async for event in stream:
                    if isinstance(event, PartDeltaEvent) and isinstance(
                        event.delta, ThinkingPartDelta
                    ):
                        rg = think.stream_thinking(event.delta.content_delta or "")
                        if rg is not None:
                            raise DegenerationError(rg)
    return run.result


__all__ = [
    "DegenerationGuard",
    "ToolCallGuard",
    "DegenerationResult",
    "DegenerationError",
    "run_guarded",
]
