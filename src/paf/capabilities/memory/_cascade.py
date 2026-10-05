"""Cascade judge for the pairwise reviewer: Jev screens, DeepSeek confirms.

Jev (``typesafe/jev-*``, a System One decision model on OpenRouter) labels every
(new fact, older fact) candidate in ONE ``/api/v1/systemone`` call. Only the
candidates it calls ``same``/``update`` -- or labels ``different`` with low
confidence -- go on to the production judge (``deepseek_judge``), which sees just
that subset. Candidates Jev is confident are ``different`` produce no judgement,
exactly as a ``different`` verdict produces no action. If the screen call fails,
every candidate is escalated, so an outage costs money, never a missed fact.

Models and the confidence floor come from the ``memory_pair_reviewer`` spec
(``screen_model`` / ``screen_min_confidence``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

from . import _pairwise
from ._pairwise import JUDGE_SPEC, Fact, PairJudgement

logger = logging.getLogger(__name__)

SYSTEMONE_URL = "https://openrouter.ai/api/v1/systemone"
SCREEN_TIMEOUT_SECONDS = 60.0
DEFAULT_SCREEN_MODEL = "typesafe/jev-1.13"
DEFAULT_MIN_CONFIDENCE = 0.9

CRITERIA = {
    "same": (
        "new_fact and {old} are fully covered by one another: they say the same "
        "thing, or one says everything the other does plus more"
    ),
    "update": (
        "the same specific attribute of the same specific thing, but the value "
        "changed (new place, number, date, version, status, or a reversed "
        "decision), and new_fact replaces {old}'s entire claim"
    ),
    "different": (
        "anything else: same topic but each says something the other doesn't, or "
        "a different topic. The thing a fact is about is specific (a project, "
        "tool, machine, place, person) -- never just 'the user'"
    ),
}


@dataclass
class CascadeStats:
    """Per-process tally, so a benchmark can print what the screen saved."""

    calls: int = 0
    failed_calls: int = 0
    candidates: int = 0
    escalated: int = 0
    cost: float = 0.0

    def summary(self) -> str:
        pct = f"{self.escalated / self.candidates:.0%}" if self.candidates else "n/a"
        return (
            f"cascade: {self.calls} screen call(s) ({self.failed_calls} failed), "
            f"{self.escalated}/{self.candidates} candidates escalated ({pct}), "
            f"screen cost ${self.cost:.5f}"
        )


STATS = CascadeStats()
JUDGE_NAME = "jev+deepseek"


def screen_enabled() -> bool:
    """Spec ``screen`` (default on); env ``PAF_PAIR_SCREEN=0|1`` overrides it, so a
    benchmark can A/B the cascade against DeepSeek-only without editing the spec."""
    env = os.environ.get("PAF_PAIR_SCREEN")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "off", "")
    from paf.specs import resolve_background_spec

    _spec, raw = resolve_background_spec(JUDGE_SPEC)
    return bool(raw.get("screen", True))


def default_judge():
    """``(judge, name)`` the reviewer uses when none is injected."""
    if screen_enabled() and os.environ.get("OPENROUTER_API_KEY"):
        return cascade_judge, JUDGE_NAME
    return _pairwise.deepseek_judge, JUDGE_SPEC


def _screen_config() -> tuple[str, float]:
    from paf.specs import resolve_background_spec

    _spec, raw = resolve_background_spec(JUDGE_SPEC)
    return (
        str(raw.get("screen_model") or DEFAULT_SCREEN_MODEL),
        float(raw.get("screen_min_confidence") or DEFAULT_MIN_CONFIDENCE),
    )


def screen_request(model: str, new: Fact, older: list[Fact]) -> dict:
    state = {"new_fact": new.record.fact} | {
        f"older_{i}": f.record.fact for i, f in enumerate(older, 1)
    }
    questions: dict = {}
    for i in range(1, len(older) + 1):
        old = f"older_{i}"
        questions[f"relation_{i}"] = {
            "type": "choice",
            "instructions": f"How does new_fact relate to {old}?",
            "criteria": {k: v.format(old=old) for k, v in CRITERIA.items()},
        }
    return {"model": model, "state": state, "questions": questions}


async def _post_screen(body: dict) -> dict:
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    async with httpx.AsyncClient(timeout=SCREEN_TIMEOUT_SECONDS) as client:
        resp = await client.post(SYSTEMONE_URL, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()


def needs_judge(rel: dict, min_confidence: float) -> bool:
    """Escalate unless Jev is confidently ``different``."""
    if rel["choice"] != "different":
        return True
    return float(rel.get("confidence") or 0.0) < min_confidence


async def cascade_judge(new: Fact, older: list[Fact]) -> list[PairJudgement]:
    """``Judge``-compatible: screen all candidates, escalate the flagged subset."""
    model, min_conf = _screen_config()
    STATS.candidates += len(older)
    STATS.calls += 1
    try:
        resp = await _post_screen(screen_request(model, new, older))
        answers = resp["answers"]
        STATS.cost += float((resp.get("usage") or {}).get("cost") or 0.0)
        flagged = [
            i
            for i in range(1, len(older) + 1)
            if needs_judge(answers[f"relation_{i}"], min_conf)
        ]
    except Exception:  # noqa: BLE001 - a screen outage must not skip facts
        logger.exception("jev screen failed on #%s; escalating all", new.line)
        STATS.failed_calls += 1
        flagged = list(range(1, len(older) + 1))
    STATS.escalated += len(flagged)
    if not flagged:
        return []
    judgements = await _pairwise.deepseek_judge(new, [older[i - 1] for i in flagged])
    out = []
    for j in judgements:
        if 1 <= j.index <= len(flagged):
            out.append(j.model_copy(update={"index": flagged[j.index - 1]}))
    return out
