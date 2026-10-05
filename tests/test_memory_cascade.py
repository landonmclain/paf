"""Cascade judge: Jev screen -> DeepSeek on the flagged subset only."""

from __future__ import annotations

import asyncio

import pytest

from paf.capabilities.memory import _cascade, _pairwise
from paf.capabilities.memory._journal import MemoryRecord
from paf.capabilities.memory._pairwise import Fact, PairJudgement


def _fact(line: int, text: str) -> Fact:
    rec = MemoryRecord.__new__(MemoryRecord)
    object.__setattr__(rec, "fact", text)
    object.__setattr__(rec, "born_on", "2026-01-01")
    return Fact(line, rec)


def _rel(choice: str, conf: float = 0.99) -> dict:
    return {"type": "choice", "choice": choice, "confidence": conf}


@pytest.fixture
def patched(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(_cascade, "_screen_config", lambda: ("m", 0.9))
    _cascade.STATS.__init__()

    async def fake_deepseek(new, older):
        seen["older"] = [f.line for f in older]
        return [PairJudgement(index=1, reason="r", label="update")]

    monkeypatch.setattr(_pairwise, "deepseek_judge", fake_deepseek)
    return seen, monkeypatch


def test_only_flagged_candidates_escalate_and_index_maps_back(patched):
    seen, mp = patched

    async def post(body):
        return {
            "answers": {
                "relation_1": _rel("different"),
                "relation_2": _rel("update", 0.8),
                "relation_3": _rel("different", 0.5),  # unsure -> escalate
            },
            "usage": {"cost": 0.001},
        }

    mp.setattr(_cascade, "_post_screen", post)
    older = [_fact(1, "a"), _fact(2, "b"), _fact(3, "c")]
    out = asyncio.run(_cascade.cascade_judge(_fact(9, "new"), older))
    assert seen["older"] == [2, 3]
    assert [j.index for j in out] == [2]  # DeepSeek's #1 is original #2
    assert _cascade.STATS.escalated == 2 and _cascade.STATS.candidates == 3


def test_all_confident_different_skips_deepseek(patched):
    seen, mp = patched

    async def post(body):
        return {"answers": {"relation_1": _rel("different")}, "usage": {}}

    mp.setattr(_cascade, "_post_screen", post)
    assert asyncio.run(_cascade.cascade_judge(_fact(9, "n"), [_fact(1, "a")])) == []
    assert "older" not in seen


def test_screen_failure_escalates_everything(patched):
    seen, mp = patched

    async def post(body):
        raise RuntimeError("boom")

    mp.setattr(_cascade, "_post_screen", post)
    asyncio.run(_cascade.cascade_judge(_fact(9, "n"), [_fact(1, "a"), _fact(2, "b")]))
    assert seen["older"] == [1, 2]
    assert _cascade.STATS.failed_calls == 1


def test_screen_env_override(monkeypatch):
    monkeypatch.setenv("PAF_PAIR_SCREEN", "0")
    assert not _cascade.screen_enabled()
    monkeypatch.setenv("PAF_PAIR_SCREEN", "1")
    assert _cascade.screen_enabled()
