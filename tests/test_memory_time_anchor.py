"""Cross-run, time-anchored resolution tests for ``resolve_unseen_span``.

The watermark is now **time-anchored** (``consumed_ts`` = the ``started_at`` of
the run holding the last folded turn). The critical behavior these tests pin is
the cross-run case the old per-run ``turn_id`` ordinal math got wrong: when a
session forks/continues into a NEW run, the new run's turns must still be
folded even though its local ``turn_id`` is small (it is NOT a global ordinal).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

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


def test_process_job_does_not_advance_watermark_on_zero_fold(tmp_path) -> None:
    """The core stall fix: a no-op fold (0 records) must NOT advance the watermark.

    The old worker advanced the watermark to the resolver's scan position
    unconditionally, so when the owner folded nothing the cadence kept firing on
    false progress and the owner stopped writing. Now a 0-fold leaves the
    durable watermark untouched so the next cadence retries the same material.
    """
    store = _store(tmp_path, {"run-a": _turns("a0", "a1", "a2", "a3")})
    memdir = tmp_path / "mem"
    memdir.mkdir(exist_ok=True)
    from pydantic_ai_harness.conversation_search import SnapshotHistorySource

    from paf.capabilities.memory._background import (
        ConsolidationJob,
        MemoryConsolidationQueue,
    )

    src = SnapshotHistorySource(store)
    seed = MemoryWatermark(run_id="run-a", turn_id=1, consumed_ts=datetime(2026, 1, 1, tzinfo=UTC))
    seed.save(str(memdir))

    async def span_source(run_id: str, up_to_turn_id: int):
        wm = MemoryWatermark.load(str(memdir), run_id=run_id)
        return await resolve_unseen_span(
            src, wm, tip_run=run_id, tip_turn=up_to_turn_id
        )

    async def zero_fold(messages, directory, watermark) -> int:
        return 0  # owner folded nothing

    async def scenario():
        q = MemoryConsolidationQueue(
            memory_directory=str(memdir),
            agent_name="assistant",
            span_source=span_source,
            consolidate_span=zero_fold,
        )
        q.enqueue("assistant", ConsolidationJob(run_id="run-a", turn_id=2))
        await q.drain("assistant")

    asyncio.run(scenario())
    # The durable watermark must be UNCHANGED after a 0-fold (not advanced to
    # the resolver's position turn 2).
    after = MemoryWatermark.load(str(memdir))
    assert after.run_id == "run-a"
    assert after.turn_id == 1
    assert after.consumed_ts == seed.consumed_ts


def test_process_job_advances_watermark_on_real_fold(tmp_path) -> None:
    """A real fold (>=1 record) advances the watermark to the folded position."""
    store = _store(tmp_path, {"run-a": _turns("a0", "a1", "a2", "a3")})
    memdir = tmp_path / "mem"
    memdir.mkdir(exist_ok=True)
    from pydantic_ai_harness.conversation_search import SnapshotHistorySource

    from paf.capabilities.memory._background import (
        ConsolidationJob,
        MemoryConsolidationQueue,
    )

    src = SnapshotHistorySource(store)
    MemoryWatermark(run_id="run-a", turn_id=1, consumed_ts=datetime(2026, 1, 1, tzinfo=UTC)).save(str(memdir))

    async def span_source(run_id: str, up_to_turn_id: int):
        wm = MemoryWatermark.load(str(memdir), run_id=run_id)
        return await resolve_unseen_span(
            src, wm, tip_run=run_id, tip_turn=up_to_turn_id
        )

    async def real_fold(messages, directory, watermark) -> int:
        return 1  # folded 1 fact

    async def scenario():
        q = MemoryConsolidationQueue(
            memory_directory=str(memdir), agent_name="assistant",
            span_source=span_source, consolidate_span=real_fold,
        )
        q.enqueue("assistant", ConsolidationJob(run_id="run-a", turn_id=3))
        await q.drain("assistant")

    asyncio.run(scenario())
    after = MemoryWatermark.load(str(memdir))
    assert after.turn_id == 2  # advanced to the last complete turn folded


# ---------------------------------------------------------------------------
# Fire-time watermark advance + always-run owner (the live-stall fixes)
# ---------------------------------------------------------------------------


def test_owner_trigger_advances_watermark_at_fire_time(tmp_path) -> None:
    """Fix A: the trigger advances the owner watermark at fire-time like the
    reviewer, so a legacy no-``consumed_ts`` watermark is upgraded to a
    time-anchored one the moment it next fires — it can no longer stay stuck
    in the old format that makes ``should_fire`` return True every turn."""
    from tests.test_memory_background_real_store import _msg

    msgs: list[ModelMessage] = []
    for i in range(12):
        msgs.extend(_msg(user=f"t{i}", assistant=f"a{i}"))
    store = _store(tmp_path, {"run-a": msgs})
    memdir = tmp_path / "mem"
    memdir.mkdir(exist_ok=True)
    from paf.capabilities.memory._journal import JournalStore

    JournalStore(str(memdir))
    # Legacy (pre-fix) owner watermark: only {run_id, turn_id}, NO consumed_ts.
    MemoryWatermark(run_id="run-a", turn_id=1).save(str(memdir))

    from paf.runtime import Session

    async def noop_fold(messages, directory, wm) -> int:
        return 0

    with Session(
        "assistant",
        trace=False,
        resume=False,
        store=store,
        memory_directory=str(memdir),
        consolidate_span=noop_fold,
    ) as session:
        for i in range(12):
            session.history.extend(_msg(user=f"t{i}", assistant=f"a{i}"))
        asyncio.run(session._maybe_trigger_consolidation("c1"))

    # After the trigger (even with a no-op fold), the owner watermark must be
    # upgraded out of the legacy format: fresh run, a turn, and a consumed_ts.
    wm = MemoryWatermark.load(str(memdir))
    assert wm.consumed_ts is not None, "trigger must write a time-anchored watermark"
    assert wm.run_id == "run-a"
    assert wm.turn_id is not None


def test_consolidate_always_runs_owner_without_durable_text(monkeypatch) -> None:
    """Fix B: ``consolidate`` no longer skips when the span has no durable
    user/assistant text — the owner is always invoked rather than silently
    folded-0 (which froze the owner watermark and starved memory)."""
    import asyncio as _asyncio

    from paf.capabilities.memory import _consolidate as m

    calls = []

    class _FakeOwner:
        async def run(self, prompt):  # noqa: A002
            calls.append(prompt)
            from types import SimpleNamespace

            return SimpleNamespace(output="ok")

    def _fake_build_owner(*args, **kwargs):
        return _FakeOwner(), None

    monkeypatch.setattr(m, "_build_owner", _fake_build_owner)

    # A span with NO durable user/assistant text (pure tool/ModelResponse).
    no_text = [
        ModelRequest(parts=[UserPromptPart(content="")]),
        ModelResponse(parts=[TextPart(content="")]),
    ]

    from paf.capabilities.memory._journal import JournalStore

    async def go():
        import tempfile

        d = tempfile.mkdtemp()
        await m.consolidate(
            no_text,
            store=None,
            agent_name="assistant",
            namespace="",
            keep_tail=0,
            journal=JournalStore(d),
        )

    _asyncio.run(go())
    assert calls, "owner must be invoked even with no durable text (gate removed)"


def test_owner_escapes_legacy_watermark_and_folds_real_store(tmp_path) -> None:
    """End to end (Fix A + B): a legacy no-``consumed_ts`` owner watermark,
    driven through the real trigger + worker against a real store, is upgraded
    to a time-anchored watermark AND the fold advances past the old position
    with journal records — the exact live scenario that was stalled."""
    from tests.test_memory_background_real_store import _msg

    msgs: list[ModelMessage] = []
    for _t in ["coffee", "remote work", "dark mode", "python", "gardening",
               "hiking", "cooking", "photography", "reading", "cycling",
               "chess", "bread"]:
        msgs.extend(_msg(
            user=f"The user enjoys {_t} as a hobby and prefers it on weekends",
            assistant=f"Recorded the {_t} preference.",
        ))
    store = _store(tmp_path, {"run-a": msgs})
    memdir = tmp_path / "mem"
    memdir.mkdir(exist_ok=True)
    from paf.capabilities.memory._journal import JournalStore

    JournalStore(str(memdir))
    # Legacy owner watermark (the stuck old-format file), consumed up to turn 1.
    MemoryWatermark(run_id="run-a", turn_id=1).save(str(memdir))

    from paf.runtime import Session

    async def real_owner_fold(messages, directory, wm) -> int:
        from paf.capabilities.memory._consolidate import consolidate
        from paf.capabilities.memory._view import JournalViewStore

        j = JournalStore(directory)
        before = j.count()
        await consolidate(
            messages,
            store=JournalViewStore(directory),
            agent_name="assistant",
            namespace="",
            keep_tail=0,
            journal=j,
        )
        return j.count() - before

    with Session(
        "assistant",
        trace=False,
        resume=False,
        store=store,
        memory_directory=str(memdir),
        consolidate_span=real_owner_fold,
    ) as session:
        topics = ["coffee", "remote work", "dark mode", "python", "gardening",
                  "hiking", "cooking", "photography", "reading", "cycling",
                  "chess", "bread"]
        for t in topics:
            session.history.extend(_msg(
                user=f"The user enjoys {t} as a hobby and prefers doing it on weekends",
                assistant=f"Recorded the {t} preference.",
            ))
        async def scenario():
            await session._maybe_trigger_consolidation("c1")
            q = session.memory_queue
            assert q is not None
            await q.drain("assistant")

        asyncio.run(scenario())

    wm = MemoryWatermark.load(str(memdir))
    assert wm.consumed_ts is not None, "owner watermark must be time-anchored now"
    assert wm.turn_id is not None and wm.turn_id > 1, "must advance past legacy turn 1"
    journal = JournalStore(str(memdir))
    rows = journal.read_all()
    assert len(rows) > 0, "owner must have folded real facts into the journal"
