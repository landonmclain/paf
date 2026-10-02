"""Real-store integration tests for the background memory-consolidation worker.

These exercise the three load-bearing behaviors the synthetic-seam unit tests
(:mod:`test_memory_background_trigger`) deliberately do NOT:
  1. **Persistence pull** — the worker resolves the unseen persisted span from a
     real ``FileStepStore`` via ``SnapshotHistorySource`` (list_runs ascending +
     run_history), honoring the ref watermark.
  2. **Watermark off it** — ``{run_id, turn_id}`` advances against the naturally
     sorted store: skip already-consumed turns within a run, include later runs.
  3. **Background queue trigger + fold** — a real store drives the worker's
     fold, advances the watermark, and renders, off the critical path.

The store fixtures mirror ``tests/test_search.py::_mk_store`` (register_run +
save_snapshot) and ``tests/test_runtime.py`` seeding.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    UserPromptPart,
)
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    RunRecord,
)

from paf.capabilities.memory._background import (
    ConsolidationJob,
    MemoryConsolidationQueue,
    ResolvedSpan,
    resolve_unseen_span,
)
from paf.capabilities.memory._journal import JournalStore
from paf.capabilities.memory._watermark import MemoryWatermark
from paf.capabilities.persistence import build_history_source
from tests.framework import TEST_PROFILE


def _msg(*, user: str, assistant: str) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content=user)]),
        ModelResponse(parts=[TextPart(assistant)]),
    ]


def _store(
    tmp_path: Path, runs: dict[str, list[ModelMessage]], conversation_id: str = "c1"
) -> FileStepStore:
    """Write a FileStepStore with one run per key, in the given dict order."""
    store = FileStepStore(directory=tmp_path)
    for i, (run_id, msgs) in enumerate(runs.items()):
        asyncio.run(
            store.register_run(
                RunRecord(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    started_at=datetime(2026, 1, 1 + i, tzinfo=UTC),
                )
            )
        )
        asyncio.run(
            store.save_snapshot(
                ContinuableSnapshot(
                    run_id=run_id,
                    step_index=0,
                    conversation_id=conversation_id,
                    messages=msgs,
                )
            )
        )
    return store


def _turns(messages: list[ModelMessage]) -> list[str]:
    """User-prompt texts in the message list, in order (the turn anchors)."""
    out = []
    for m in messages:
        for p in m.parts:
            if isinstance(p, UserPromptPart):
                content = p.content
                text = content if isinstance(content, str) else str(content)
                out.append(text)
    return out


# ---------------------------------------------------------------------------
# 1. Persistence pull — resolve_unseen_span over a real store
# ---------------------------------------------------------------------------


def test_resolve_unseen_span_fresh_watermark_returns_all_complete_turns(
    tmp_path: Path,
) -> None:
    """Fresh watermark (nothing consumed) -> every complete turn, in order.

    ``third`` is the in-flight tip (tip=run-b:1), so it is excluded — the fold
    stops on the second-to-last turn to guarantee only settled turns fold.
    """
    store = _store(
        tmp_path,
        {
            "run-a": [*_msg(user="first", assistant="a1")],
            "run-b": [
                *_msg(user="second", assistant="b1"),
                *_msg(user="third", assistant="b2"),
            ],
        },
    )
    source = SnapshotHistorySource(store)
    wm = MemoryWatermark()  # fresh

    async def go():
        span = await resolve_unseen_span(source, wm, tip_run="run-b", tip_turn=1)
        # "third" is the tip -> excluded; only complete turns fold.
        assert _turns(span.messages) == ["first", "second"]
        # watermark advances to the last COMPLETE turn folded.
        assert span.watermark.run_id == "run-b"
        assert span.watermark.turn_id == 0

    asyncio.run(go())


def test_resolve_unseen_span_skips_consumed_tail_past_watermark(tmp_path: Path) -> None:
    """Watermark consumed run-a turn 0 -> run-a turn 1 + run-b complete turns."""
    store = _store(
        tmp_path,
        {
            "run-a": [
                _msg(user="consumed", assistant="a1")[0],
                _msg(user="mid", assistant="a2")[0],
            ],
            "run-b": [
                _msg(user="new-1", assistant="b1")[0],
                _msg(user="new-2", assistant="b2")[0],
            ],
        },
    )
    source = SnapshotHistorySource(store)
    wm = MemoryWatermark(run_id="run-a", turn_id=0)

    async def go():
        span = await resolve_unseen_span(source, wm, tip_run="run-b", tip_turn=1)
        # consumed(run-a:0) skipped; mid(run-a:1) + new-1(run-b:0) fold;
        # new-2(run-b:1) is the tip -> excluded.
        assert _turns(span.messages) == ["mid", "new-1"]
        assert span.watermark.run_id == "run-b"
        assert span.watermark.turn_id == 0

    asyncio.run(go())


def test_resolve_unseen_span_cross_run_cumulative_no_duplication(
    tmp_path: Path,
) -> None:
    """A resumed/steered cumulative run must not re-collect the prior run's prefix.

    Regression for the resume-duplication bug: run-b is a ``continue_run`` whose
    snapshot carries run-a's FULL history forward as a prefix (re-instantiated
    messages). The cross-run overlap reconciliation must strip that prefix, so
    fresh-watermark resolution yields each turn ONCE. It relies on the
    content-stable ``_content_hash`` — the harness ``message_hash`` hashes
    pydantic's per-instantiation ``timestamp``, so byte-identical re-instantiated
    messages hash differently and the overlap fires 0 (see the history-dup bug).
    """
    # run-b = full copy of run-a + 2 new turns (cumulative, like a resume).
    run_a = [*_msg(user="q1", assistant="a1"), *_msg(user="q2", assistant="a2")]
    run_b = [
        *_msg(user="q1", assistant="a1"),
        *_msg(user="q2", assistant="a2"),
        *_msg(user="q3", assistant="a3"),
        *_msg(user="q4", assistant="a4"),
    ]
    store = _store(tmp_path, {"run-a": run_a, "run-b": run_b})
    source = SnapshotHistorySource(store)
    wm = MemoryWatermark()  # fresh

    async def go():
        # q4 is the in-flight tip (excluded); q1/q2/q3 complete.
        span = await resolve_unseen_span(source, wm, tip_run="run-b", tip_turn=3)
        # Each turn appears exactly ONCE — the cumulative prefix is not re-added.
        assert _turns(span.messages) == ["q1", "q2", "q3"]
        assert span.watermark.run_id == "run-b"
        assert span.watermark.turn_id == 2

    asyncio.run(go())


def test_resolve_unseen_span_cross_run_three_runs_no_duplication(
    tmp_path: Path,
) -> None:
    """A 3-run cumulative chain (two steers) also dedupes to one copy each."""
    run_a = [*_msg(user="q1", assistant="a1")]
    run_b = [*_msg(user="q1", assistant="a1"), *_msg(user="q2", assistant="a2")]
    run_c = [
        *_msg(user="q1", assistant="a1"),
        *_msg(user="q2", assistant="a2"),
        *_msg(user="q3", assistant="a3"),
        *_msg(user="q4", assistant="a4"),
    ]
    store = _store(tmp_path, {"run-a": run_a, "run-b": run_b, "run-c": run_c})
    source = SnapshotHistorySource(store)
    wm = MemoryWatermark()  # fresh

    async def go():
        # q4 in flight; q1/q2/q3 complete and must appear once each.
        span = await resolve_unseen_span(source, wm, tip_run="run-c", tip_turn=3)
        assert _turns(span.messages) == ["q1", "q2", "q3"]
        assert span.watermark.run_id == "run-c"
        assert span.watermark.turn_id == 2

    asyncio.run(go())


def test_resolve_unseen_span_excludes_in_flight_tip(tmp_path: Path) -> None:
    """The in-flight tip turn is never folded (stop on the second-to-last turn)."""
    store = _store(
        tmp_path,
        {
            "run-a": [*_msg(user="only", assistant="a")],
        },
    )
    source = SnapshotHistorySource(store)
    wm = MemoryWatermark()  # fresh

    async def go():
        # tip is run-a:0 — the ONLY turn, and it's in flight -> fold nothing.
        span = await resolve_unseen_span(source, wm, tip_run="run-a", tip_turn=0)
        assert span.messages == []
        # fresh watermark is untouched (nothing folded).
        assert span.watermark.run_id == ""
        assert span.watermark.turn_id is None

    asyncio.run(go())


# ---------------------------------------------------------------------------
# 3. Session hook — combustion of trigger + real-store worker (live wiring)
# ---------------------------------------------------------------------------


def test_session_hook_enqueues_job_past_cadence(tmp_path: Path) -> None:
    """The Session turn hook fires the trigger and folds against a real store.

    Drives the LIVE wiring end to end: ``_maybe_trigger_consolidation``
    computes the tip, ``should_fire`` approves, the queue (constructed lazily,
    bound to ``self._store``) receives a job, and its self-contained worker
    resolves the persisted span and folds it into the journal — all without a
    live LLM (journal-fenced consolidate seam).
    """
    msgs: list[ModelMessage] = []
    for i in range(12):
        msgs.extend(_msg(user=f"t{i}", assistant=f"a{i}"))
    store = _store(tmp_path, {"run-a": msgs})
    memdir = tmp_path / "mem"
    memdir.mkdir(exist_ok=True)
    JournalStore(str(memdir))
    # Watermark at turn 1; 12 turns -> tip=11, 11-1=10 >= 10 cadence -> fires.
    MemoryWatermark(run_id="run-a", turn_id=1).save(str(memdir))

    async def fence(
        messages: list[ModelMessage], directory: str, wm: MemoryWatermark
    ) -> int:
        j = JournalStore(directory)
        for m in messages:
            for p in m.parts:
                if isinstance(p, UserPromptPart):
                    j.append(
                        type="fact",
                        fact=str(p.content)
                        if isinstance(p.content, str)
                        else str(p.content),
                        tier="t3",
                        born_on=datetime.now(UTC),
                        source_ref="run-a:0",
                    )
        return j.count()

    from paf.background import BackgroundCoordinator
    from paf.runtime import Session

    with Session(
        "test",
        profile=TEST_PROFILE,
        trace=False,
        resume=False,
        store=store,
    ) as session:
        session.background = BackgroundCoordinator(
            store=store,
            agent_name="assistant",
            memory_directory=str(memdir),
            consolidate_span=fence,
        )
        for i in range(12):
            session.history.extend(_msg(user=f"t{i}", assistant=f"a{i}"))

        async def scenario():
            await session.background._maybe_trigger_consolidation("c1")
            q = session.background.memory_queue
            assert q is not None, "cadence reached, so the queue must be constructed"
            await q.drain("assistant")  # let the fire-and-forget worker finish

        asyncio.run(scenario())

    # The worker folded the persisted span of complete turns (turns 2..10:
    # turns 0,1 consumed; turn 11 is the in-flight tip) and advanced the
    # durable watermark to the last complete turn folded.
    wm = MemoryWatermark.load(str(memdir))
    assert wm.run_id == "run-a"
    assert wm.turn_id == 10
    journal = JournalStore(str(memdir))
    rows = journal.read_all()
    folded = sorted(str(r.fact) for _, r in rows)
    assert "t2" in folded and "t10" in folded
    # the in-flight tip was never folded
    assert not any(str(r.fact) == "t11" for _, r in rows)
    assert (memdir / "MEMORY.md").exists()


def test_turn_gate_fires_at_consolidate_turns_threshold(tmp_path: Path) -> None:
    """The trigger fires once ENOUGH COMPLETE turns have accrued (turn gate).

    Regression for the DECIDED 2026-09-14 cadence change (see
    Designs/Turn-Based-Background-Cadence): consolidation should fire once at
    least ``consolidate_turns`` (default 5, spec-configurable) *settled* turns
    have accrued since the last fold — NOT on a minute timer. This test pins
    the turn-count boundary and the spec configurability.
    """
    from paf.background import SKIPPED, BackgroundCoordinator

    # 7 user turns (indices 0..6): watermark consumed turn 0, the tip turn 6 is
    # in-flight, so the span is complete turns 1..5 = FIVE -> fires at 5.
    msgs: list[ModelMessage] = []
    for i in range(7):
        msgs.extend(_msg(user=f"t{i}", assistant=f"a{i}"))
    store = _store(tmp_path, {"run-a": msgs})
    memdir = tmp_path / "mem"
    memdir.mkdir(exist_ok=True)
    JournalStore(str(memdir))
    MemoryWatermark(run_id="run-a", turn_id=0).save(str(memdir))

    bg = BackgroundCoordinator(
        store=store,
        agent_name="assistant",
        memory_directory=str(memdir),
        consolidate_turns=5,  # explicit override, not the spec default
    )
    fired: list[str] = []
    bg.subscribe_tasks(lambda ev: fired.append(ev.status))

    async def scenario():
        await bg._maybe_trigger_consolidation("c1")

    asyncio.run(scenario())

    # The queue must have been constructed and a job enqueued (PROCESSING).
    assert bg.memory_queue is not None, "consolidate_turns reached -> queue built"
    assert fired and fired[-1] != SKIPPED


def test_turn_gate_skips_below_threshold_and_defaults(tmp_path: Path) -> None:
    """One complete turn short of the gate -> skip; turn gates resolve to defaults."""
    from paf.background import SKIPPED, BackgroundCoordinator
    from paf.specs import resolve_memory_config

    # The turn-count gate comes from the code default, not a spec knob.
    cfg = resolve_memory_config("demo_assistant")
    assert cfg["consolidate_turns"] == 5

    # 5 complete turns -> span is 4 (tip excluded) -> ONE short of 5 -> skip.
    msgs: list[ModelMessage] = []
    for i in range(5):
        msgs.extend(_msg(user=f"t{i}", assistant=f"a{i}"))
    store = _store(tmp_path, {"run-a": msgs})
    memdir = tmp_path / "mem"
    memdir.mkdir(exist_ok=True)
    JournalStore(str(memdir))
    MemoryWatermark(run_id="run-a", turn_id=0).save(str(memdir))

    bg = BackgroundCoordinator(
        store=store,
        agent_name="assistant",
        memory_directory=str(memdir),
        consolidate_turns=5,
    )
    fired: list[str] = []
    bg.subscribe_tasks(lambda ev: fired.append(ev.status))

    async def scenario():
        await bg._maybe_trigger_consolidation("c1")

    asyncio.run(scenario())

    # Below the gate -> SKIPPED and the queue is never even constructed.
    assert fired and fired[-1] == SKIPPED
    assert bg.memory_queue is None


def test_session_hook_skips_below_cadence(tmp_path: Path) -> None:
    """Below the cadence threshold the hook enqueues nothing."""
    msgs: list[ModelMessage] = []
    for i in range(5):
        msgs.extend(_msg(user=f"t{i}", assistant=f"a{i}"))
    store = _store(tmp_path, {"run-a": msgs})
    memdir = tmp_path / "mem"
    memdir.mkdir(exist_ok=True)
    JournalStore(str(memdir))
    # 5 user turns -> resolve_unseen_span yields 4 COMPLETE turns (the tip is
    # in-flight) -> one short of the consolidate_turns=5 default -> no fire.
    MemoryWatermark(run_id="run-a", consumed_ts=datetime.now(UTC)).save(str(memdir))

    from paf.background import BackgroundCoordinator
    from paf.runtime import Session

    with Session(
        "test",
        profile=TEST_PROFILE,
        trace=False,
        resume=False,
        store=store,
    ) as session:
        session.background = BackgroundCoordinator(
            store=store,
            agent_name="assistant",
            memory_directory=str(memdir),
        )
        for i in range(5):
            session.history.extend(_msg(user=f"t{i}", assistant=f"a{i}"))

        asyncio.run(session.background._maybe_trigger_consolidation("c1"))

        q = session.background.memory_queue
        # Below cadence -> the queue is never even constructed.
        assert q is None


# ---------------------------------------------------------------------------
# 2. Queue fold advances the ref watermark against a real store
# ---------------------------------------------------------------------------


def test_queue_folds_real_store_and_advances_watermark(tmp_path: Path) -> None:
    """Draining a job with a real store folds complete turns, advances, renders."""
    store = _store(
        tmp_path,
        {
            "run-a": [
                *_msg(user="hello", assistant="hey"),
                *_msg(user="tip", assistant="t"),
            ]
        },
    )
    source = SnapshotHistorySource(store)
    memdir = tmp_path / "mem"
    memdir.mkdir(exist_ok=True)
    JournalStore(str(memdir))  # create the empty journal

    q = MemoryConsolidationQueue(
        memory_directory=str(memdir),
    )

    async def span(run: str, turn: int) -> ResolvedSpan:
        wm = MemoryWatermark.load(str(memdir), run_id=run)
        return await resolve_unseen_span(source, wm, tip_run=run, tip_turn=turn)

    async def fake_consolidate(
        messages: list[ModelMessage], directory: str, watermark: MemoryWatermark
    ) -> int:
        j = JournalStore(directory)
        for m in messages:
            for p in m.parts:
                if isinstance(p, UserPromptPart):
                    j.append(
                        type="fact",
                        fact=str(p.content),
                        tier="t3",
                        born_on=datetime.now(UTC),
                        source_ref="run-a:0",
                    )
        return j.count()

    q.span_source = span
    q.consolidate_span = fake_consolidate

    async def go():
        # job position = the in-flight tip (run-a:1).
        await q.process_job("sess", ConsolidationJob(run_id="run-a", turn_id=1), None)

    asyncio.run(go())

    wm = MemoryWatermark.load(str(memdir))
    # watermark advanced to the last COMPLETE turn folded = run-a:0 (turn 1 is tip).
    assert wm.run_id == "run-a"
    assert wm.turn_id == 0
    assert (memdir / "memory-watermark.json").exists()
    assert (memdir / "MEMORY.md").exists()
    # the folded (complete) turn is in the journal (via read-back)
    j = JournalStore(str(memdir))
    rows = j.read_all()
    assert any(str(r.fact) == "hello" for _, r in rows), rows
    # the in-flight tip turn was NOT folded
    assert not any(str(r.fact) == "tip" for _, r in rows), rows


# ---------------------------------------------------------------------------
# 5. Within-run cumulative-snapshot duplication (real-live-branch regression)
# ---------------------------------------------------------------------------


def _multi_snapshot_store(tmp_path: Path, conversation_id: str = "c1") -> FileStepStore:
    """One run whose SNAPSHOTS carry cumulative history, like the live harness.

    A ``ContinuableSnapshot`` in this store is the FULL accumulating history
    re-instantiated each step, so snapshot N repeats everything snapshot N-1
    had plus one new turn (the live branch's step-persistence pattern). Each
    snapshot is written from a freshly-instantiated message list, mirroring the
    real deserialization that gives parts distinct ``timestamp`` values.
    """
    store = FileStepStore(directory=tmp_path)
    run_id = "run-multi"
    asyncio.run(
        store.register_run(
            RunRecord(
                run_id=run_id,
                conversation_id=conversation_id,
                started_at=datetime(2026, 2, 1, tzinfo=UTC),
            )
        )
    )
    turns = [("t0", "a0"), ("t1", "a1"), ("t2", "a2"), ("t3", "a3")]
    for i in range(len(turns)):
        # Build a FRESH cumulative message list (new parts -> new timestamps)
        # each snapshot, exactly as the harness does when it re-serializes the
        # accumulating history at every step.
        cumulative: list[ModelMessage] = []
        for user, assistant in turns[: i + 1]:
            cumulative.extend(_msg(user=user, assistant=assistant))
        asyncio.run(
            store.save_snapshot(
                ContinuableSnapshot(
                    run_id=run_id,
                    step_index=i,
                    conversation_id=conversation_id,
                    messages=cumulative,
                )
            )
        )
    return store


def from_harness_run_history(store: FileStepStore, run_id: str) -> bool:
    """True if the harness's own ``run_history`` duplicates turns on this store."""
    harness = SnapshotHistorySource(store)
    history = asyncio.run(harness.run_history(run_id=run_id))
    turns = _turns(history)
    return len(turns) != len(set(turns))


def test_stable_history_source_dedups_within_run_cumulative_snapshots(
    tmp_path: Path,
) -> None:
    """Each turn appears exactly once in ``run_history`` under the stable source.

    The harness's own ``SnapshotHistorySource.run_history`` reconciles the
    snapshot union with ``message_hash`` (timestamp-unstable), so every
    cumulative snapshot re-appends the whole history — the within-run
    duplication the live 18-run conversation was hit by (61+ snapshots -> each
    turn repeated dozens of times). ``build_history_source`` keys on
    content-stable hashes, so a turn carried forward by later snapshots is
    emitted once.
    """
    store = _multi_snapshot_store(tmp_path)

    # Guard: the harness source is genuinely broken on this shape (proves the
    # regression exists and the stable source is what fixes it, not something
    # already calling it). If this ever starts passing, the harness fixed its
    # hashing and StableSnapshotHistorySource may become redundant.
    harness_broken = from_harness_run_history(store, "run-multi")
    assert harness_broken, (
        "harness run_history unexpectedly deduped; StableSnapshotHistorySource "
        "may now be redundant"
    )

    stable = build_history_source(store)
    history = asyncio.run(stable.run_history(run_id="run-multi"))
    turns = _turns(history)
    assert turns == ["t0", "t1", "t2", "t3"], turns


def test_injected_marker_blocks_do_not_anchor_turns() -> None:
    """Injected `<memory>`/index blocks never count as user turns (marker SOP).

    The thread summarizer's minimum-turn gate must count *distinct real* user
    turns. Injected context (harness `TextContent` tagged with an artifact
    marker) must be stripped so it cannot inflate the floor, while real user
    prompts — whether plain-str or list-wrapped — still anchor.
    """
    from paf.capabilities._markers import injected_artifact_markers
    from paf.capabilities._turns import _is_user_turn, distinct_user_turn_texts

    marker = injected_artifact_markers()[1] + ":scope_hash"
    real_req = ModelRequest(parts=[UserPromptPart(content="real question")], run_id="r")
    # harness memory injection: UserPromptPart whose content is a list of
    # marker-tagged TextContent (system context, not user speech).
    injected_req = ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    TextContent(
                        content="\n\n<memory>\n### MEMORY.md\n...", metadata=marker
                    )
                ]
            )
        ],
        run_id="r",
    )
    # list-wrapped real prompt (the shape test/read fixtures use): still real.
    listwrapped_req = ModelRequest(
        parts=[UserPromptPart(content=["real question"])], run_id="r"
    )

    assert _is_user_turn(real_req) is True
    assert _is_user_turn(listwrapped_req) is True
    assert _is_user_turn(injected_req) is False, (
        "injected <memory> block must not anchor"
    )

    msgs = [real_req, injected_req, listwrapped_req]
    # the real prompt appears twice (clean + listwrapped) -> collapses to one
    assert distinct_user_turn_texts(msgs) == ["real question"]


def test_pairwise_review_is_on_by_default_and_spec_can_turn_it_off(
    tmp_path: Path,
) -> None:
    """The pairwise reviewer runs after each fold unless a spec sets
    `memory.pairwise_review: false`; there is no separate review process."""
    from paf.specs import _spec_memory_config, resolve_memory_config

    spec = tmp_path / "agent.yaml"
    spec.write_text("memory:\n  pairwise_review: false\n", encoding="utf-8")
    assert _spec_memory_config(spec) == {"pairwise_review": False}
    assert resolve_memory_config("demo_assistant")["pairwise_review"] is True
    assert resolve_memory_config("spec-less-agent")["pairwise_review"] is True

    from paf.background import BackgroundCoordinator
    from paf.capabilities.profiles import AgentProfile

    bg = BackgroundCoordinator(
        store=_store(tmp_path, {}),
        agent_name="spec-less-agent",
        memory_directory=str(tmp_path / "mem"),
        profile=AgentProfile(name="memory", capability_names=("Memory",)),
        lane="agent",
    )
    assert bg.pairwise_review is True
    assert [p.id_namespace for p in bg._processes] == ["consolidation"]
