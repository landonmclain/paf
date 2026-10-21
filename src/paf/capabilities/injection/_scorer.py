"""Scorers: turn one tool-result value into a :class:`Score`.

The policy never talks to a classifier directly, only to this interface, so the
classifier can be replaced (another model, an LLM adjudicator, a cascade) without
touching what paf does with the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import anyio.to_thread
from pydantic_core import to_jsonable_python

# Risk levels that mean the scorer escalated beyond stackone's 'medium' baseline.
_ESCALATED = ("high", "critical")


@dataclass(frozen=True)
class Score:
    """A scorer's opinion of one value."""

    hit: bool
    """The value crosses the scorer's decision line (what ``block`` acts on)."""
    noteworthy: bool
    """Worth tracing even if not a hit (pattern findings, escalated risk)."""
    risk_level: str
    score: float | None
    """The semantic classifier's score, if it ran."""
    detections: tuple[str, ...] = ()
    trigger: str | None = None
    """The chunk that scored highest, when the scorer can say."""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    """Plain-JSON details for spans and ``ToolReturn.metadata``."""


class InjectionScorer(Protocol):
    """Anything that can score a tool-result value."""

    @property
    def name(self) -> str: ...

    @property
    def threshold(self) -> float | None:
        """The effective decision threshold, if the scorer has one."""
        ...

    def warmup(self) -> None:
        """Load models eagerly (called off the event loop at run start)."""
        ...

    async def score(self, value: object, tool_name: str) -> Score: ...

    async def localize(
        self, text: str, tool_name: str
    ) -> list[tuple[str, float]] | None:
        """Per-segment scores for ``text``: verbatim substrings and their scores.

        Used by the ``redact`` action. ``None`` means the scorer cannot say where
        in the text the problem is, and the caller falls back to ``block``.
        """
        ...


def _payload(value: object) -> object:
    """Put a bare string under stackone's standard ``content`` field.

    Matches what the harness defender did, so verdicts are unchanged by the
    rehome. Tier 1 pattern detection inspects known risky fields.
    """
    if isinstance(value, str):
        return {"content": value}
    return to_jsonable_python(value, fallback=str)


class StackoneScorer:
    """``stackone-defender``: Tier 1 patterns plus the local MiniLM classifier."""

    name = "stackone"

    def __init__(
        self, *, semantic: bool = True, threshold: float | None = None
    ) -> None:
        from stackone_defender import PromptDefense

        tier2_config = (
            {"high_risk_threshold": threshold} if threshold is not None else None
        )
        self._semantic = semantic
        self._defense = PromptDefense(
            # Scan large arrays in full; the default samples them, letting a tail
            # injection through (same setting the harness used).
            config={"traversal": {"skip_large_arrays": False}},
            block_high_risk=True,
            enable_tier2=semantic,
            tier2_config=tier2_config,
        )

    @property
    def threshold(self) -> float | None:
        if not self._semantic:
            return None
        return float(self._defense._tier2.get_config()["high_risk_threshold"])  # noqa: SLF001

    def warmup(self) -> None:
        if self._semantic:
            self._defense.warmup_tier2()

    def score_sync(self, value: object, tool_name: str) -> Score:
        """Synchronous scoring (the eval uses this; the capability awaits ``score``)."""
        return self._to_score(
            self._defense.defend_tool_result(_payload(value), tool_name)
        )

    async def score(self, value: object, tool_name: str) -> Score:
        verdict = await self._defense.defend_tool_result_async(
            _payload(value), tool_name
        )
        return self._to_score(verdict)

    def localize_sync(self, text: str) -> list[tuple[str, float]] | None:
        """Sentence-level scores from the same classifier (``None`` without it)."""
        if not self._semantic:
            return None
        result = self._defense._tier2.classify_by_sentence(text)  # noqa: SLF001
        if result.get("skipped"):
            return None
        return [
            (s["sentence"], float(s["score"]))
            for s in result.get("sentence_scores", [])
        ]

    async def localize(
        self, text: str, tool_name: str
    ) -> list[tuple[str, float]] | None:
        return await anyio.to_thread.run_sync(self.localize_sync, text)

    @staticmethod
    def _to_score(verdict: Any) -> Score:
        findings = bool(verdict.detections) or bool(verdict.fields_sanitized)
        hit = not verdict.allowed
        return Score(
            hit=hit,
            noteworthy=hit or findings or verdict.risk_level in _ESCALATED,
            risk_level=verdict.risk_level,
            score=verdict.tier2_score,
            detections=tuple(verdict.detections),
            trigger=verdict.max_sentence,
            diagnostics={
                "blocked": hit,
                "risk_level": verdict.risk_level,
                "detections": list(verdict.detections),
                "fields_sanitized": list(verdict.fields_sanitized),
                "tier2_score": verdict.tier2_score,
                "latency_ms": verdict.latency_ms,
            },
        )
