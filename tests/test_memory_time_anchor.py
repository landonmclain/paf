"""Cross-run, time-anchored resolution tests for ``resolve_unseen_span``.

The watermark is now **time-anchored** (``consumed_ts`` = the ``started_at`` of
the run holding the last folded turn). The critical behavior these tests pin is
the cross-run case the old per-run ``turn_id`` ordinal math got wrong: when a
session forks/continues into a NEW run, the new run's turns must still be
folded even though its local ``turn_id`` is small (it is NOT a global ordinal).
"""

from __future__ import annotations

import asyncio

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from paf.capabilities.memory._background import resolve_unseen_span
from paf.capabilities.memory._watermark import MemoryWatermark
from tests.test_memory_background_real_store import _store


def _turns(*labels: str) -> list[ModelMessage]:
    """Messages for turns labeled ``t0..tn`` — user prompt + assistant reply."""
    msgs: list[ModelMessage] = []
    for lab in labels:
        msgs.append(ModelRequest(parts=[UserPromptPart(content=lab)]))
        msgs.append(ModelResponse(parts=[TextPart(f"reply:{lab}")]))
    return msgs


def _resolve(span_source, wm, tip_run, tip_turn):
    return resolve_unseen_span(span_source, wm, tip_run=tip_run, tip_turn=tip_turn)


def _user_texts(messages) -> set[str]:
    out = set()
    for m in messages:
        for p in m.parts:
            if isinstance(p, UserPromptPart):
                c = p.content
                out.add(c if isinstance(c, str) else str(c))
    return out


def test_folds_new_run_after_watermark_in_old_run(tmp_path) -> None:
    """A fork into a NEW run is folded even though its local turn_id restarts."""
    histories = {
        "run-a": _turns("a0", "a1", "a2"),  # 2 complete turns (turn 2 is last)
        "run-b": _turns("b0", "b1"),        # fork; local turn ids restart at 0
    }

    store = _store(tmp_path, {"run-a": histories["run-a"], "run-b": histories["run-b"]})
    from pydantic_ai_harness.conversation_search import SnapshotHistorySource

    src = SnapshotHistorySource(store)
    # Watermark consumed run-a up to turn 1 (time-anchored at run-a's started_at).
    wm = MemoryWatermark(run_id="run-a", turn_id=1)

    span = asyncio.run(
        _resolve(src, wm, tip_run="run-b", tip_turn=1)  # in-flight tip = run-b turn 1
    )

    texts = _user_texts(span.messages)
    # run-a's turn 2 (a2) plus the forked run-b's turn 0 (b0) must be folded.
    assert "a2" in texts
    assert "b0" in texts, "the forked run's turn 0 must be folded (old code dropped it)"
    # the in-flight tip (run-b turn 1 = b1) is excluded
    assert "b1" not in texts
    # watermark advanced into run-b at its time anchor
    assert span.watermark.run_id == "run-b"
    assert span.watermark.turn_id == 0
    # the TIME anchor advanced to run-b's started_at (not None) — the old
    # resolver advanced only the {run_id, turn_id} ref and left consumed_ts
    # unset, so the next cadence had no time basis to gauge progress.
    assert span.watermark.consumed_ts is not None


def test_single_turn_transcript_folds_nothing(tmp_path) -> None:
    """One run, one turn: it is the in-flight tip -> nothing folds."""
    store = _store(tmp_path, {"run-a": _turns("a0")})
    from pydantic_ai_harness.conversation_search import SnapshotHistorySource

    src = SnapshotHistorySource(store)
    wm = MemoryWatermark()
    span = asyncio.run(_resolve(src, wm, tip_run="run-a", tip_turn=0))
    assert span.messages == []
    # nothing folded -> watermark untouched
    assert span.watermark.run_id == ""


def test_time_anchor_makes_progress_across_cadence_after_fork(tmp_path) -> None:
    """Sustained loop: each resolve advances the time anchor across a fork.

    Cadence 1 folds run-a 0..2. After a fork spawns run-b, cadence 2 folds
    run-b 0 (tip 1 in-flight), then cadence 3 folds run-b 1. The old
    {run_id, turn_id}-only watermark stalled here because run-b's local turn
    ids restarted and there was no global (time) anchor to gauge progress.
    """
    store = _store(
        tmp_path,
        {"run-a": _turns("a0", "a1", "a2"), "run-b": _turns("b0", "b1")},
    )
    from pydantic_ai_harness.conversation_search import SnapshotHistorySource

    src = SnapshotHistorySource(store)

    # Cadence 1: fold run-a 0..2, tip = run-a turn 3 (in flight).
    wm = MemoryWatermark()
    s1 = asyncio.run(_resolve(src, wm, tip_run="run-a", tip_turn=3))
    assert _user_texts(s1.messages) == {"a0", "a1", "a2"}
    assert s1.watermark.consumed_ts is not None

    # Cadence 2: continue from s1's watermark; run-b spawned, fold run-b 0.
    s2 = asyncio.run(_resolve(src, s1.watermark, tip_run="run-b", tip_turn=1))
    assert _user_texts(s2.messages) == {"b0"}
    assert s2.watermark.run_id == "run-b"
    assert s2.watermark.turn_id == 0
    assert s2.watermark.consumed_ts is not None

    # Cadence 3: continue from s2's watermark; tip moved to run-b turn 2.
    s3 = asyncio.run(_resolve(src, s2.watermark, tip_run="run-b", tip_turn=2))
    assert _user_texts(s3.messages) == {"b1"}  # only the newly-settled turn
    assert s3.watermark.turn_id == 1
