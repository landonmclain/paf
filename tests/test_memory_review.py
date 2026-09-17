"""Memory reviewer surface: ``memory_review`` tool, ``ReviewerToolset``, the
``Memory``-capability-with-reviewer wiring, ``read_render``, and the background
``MemoryReviewQueue`` mechanics.

Covers the Phase-3 curation pass plumbing deterministically — no live LLM. The
real reviewer ``Agent`` construction (``_build_reviewer``) is exercised for its
capability/toolset wiring; the LLM pass itself is not run (costs tokens and is
covered by the live web loop + an optional manual run).
"""

from __future__ import annotations

import asyncio
from datetime import UTC

import pytest
from pydantic_ai import ModelRetry

from paf.capabilities.memory._index import build_index
from paf.capabilities.memory._journal import JournalStore, MemoryRecord
from paf.capabilities.memory._overrides import OverrideStore
from paf.capabilities.memory._render import read_render
from paf.capabilities.memory._toolset import OverrideToolset, ReviewerToolset


def _mk_record(
    fact, *, tier="t1", born="2026-08-17T00:00:00Z", src="run:0", typ="fact"
):
    return MemoryRecord(type=typ, fact=fact, tier=tier, born_on=born, source_ref=src)


def _seed(tmp_path, facts):
    store = JournalStore(tmp_path)
    for r in facts:
        store.append_record(r)
    build_index(tmp_path)
    return store


class _NullSpan:
    def is_recording(self):
        return False

    def set_attributes(self, attrs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return None


class _FakeCtx:
    def __init__(self):
        self.tracer = type(
            "T", (), {"start_as_current_span": staticmethod(lambda *a, **k: _NullSpan())}
        )()


# ---------------------------------------------------------------------------
# ReviewerToolset surface
# ---------------------------------------------------------------------------


def test_reviewer_toolset_exposes_only_memory_review(tmp_path):
    rt = ReviewerToolset(OverrideStore(tmp_path))
    assert set(rt.tools.keys()) == {"memory_review"}
    assert "append_memory" not in rt.tools
    assert "memory_archive_suggestion" not in rt.tools  # the reviewer, not the assistant


def test_reviewer_toolset_archive_applies_and_drops_fact(tmp_path):
    _seed(tmp_path, [_mk_record("User prefers flat-file storage")])
    rt = ReviewerToolset(OverrideStore(tmp_path))

    result = asyncio.run(
        rt.memory_review(_FakeCtx(), journal_id=1, op="archive", reason="disproven")
    )
    assert result["status"] == "applied"
    # fact dropped from the render immediately
    md = read_render(str(tmp_path))
    assert "flat-file storage" not in md


def test_reviewer_toolset_tier_change_moves_fact(tmp_path):
    _seed(tmp_path, [_mk_record("Stable identity fact", tier="t1")])
    rt = ReviewerToolset(OverrideStore(tmp_path))

    asyncio.run(
        rt.memory_review(
            _FakeCtx(), journal_id=1, op="tier_change", tier="t3", reason="deprioritized"
        )
    )
    md = read_render(str(tmp_path))
    # fact is now under t3, not t1
    t3_idx = md.find("## t3")
    assert t3_idx != -1
    # no t1 section holds it (it moved down)
    assert "Stable identity fact" in md[t3_idx:]


# ---------------------------------------------------------------------------
# Memory capability wiring: reviewer=True -> ReviewerToolset
# ---------------------------------------------------------------------------


def test_memory_capability_reviewer_wires_reviewer_toolset(tmp_path):
    from paf.capabilities.memory import Memory
    from paf.capabilities.memory._view import JournalViewStore

    m = Memory(
        store=JournalViewStore(tmp_path),
        agent_name="reviewer",
        overrides=OverrideStore(tmp_path),
        reviewer=True,
        expose_tools=True,
    )
    ts = m.get_toolset()
    assert ts is not None
    assert isinstance(ts, ReviewerToolset)
    assert set(ts.tools.keys()) == {"memory_review"}


def test_memory_capability_assistant_still_gets_override_toolset(tmp_path):
    from paf.capabilities.memory import Memory
    from paf.capabilities.memory._view import JournalViewStore

    m = Memory(
        store=JournalViewStore(tmp_path),
        agent_name="assistant",
        overrides=OverrideStore(tmp_path),
        reviewer=False,
    )
    ts = m.get_toolset()
    assert isinstance(ts, OverrideToolset)
    assert set(ts.tools.keys()) == {"memory_archive_suggestion"}


# ---------------------------------------------------------------------------
# _review module: reviewer agent wiring (no LLM run)
# ---------------------------------------------------------------------------


def test_review_builds_reviewer_agent(tmp_path):
    from paf.capabilities.memory._review import (
        _build_reviewer_memory,
        review_model,
    )
    from paf.capabilities.memory._view import JournalViewStore

    overrides = OverrideStore(tmp_path)
    mem = _build_reviewer_memory(
        JournalViewStore(tmp_path),
        overrides,
        agent_name="reviewer",
        namespace="",
    )
    assert mem.get_toolset() is not None
    assert set(mem.get_toolset().tools.keys()) == {"memory_review"}
    assert review_model()  # resolves to the owner's model string


# ---------------------------------------------------------------------------
# read_render
# ---------------------------------------------------------------------------


def test_read_render_returns_full_unbounded_view(tmp_path):
    store = _seed(tmp_path, [_mk_record("First fact"), _mk_record("Second fact")])
    md = read_render(str(tmp_path))  # no MEMORY.md written yet -> computes
    assert "First fact" in md
    assert "Second fact" in md

    # after write_render, read_render returns the on-disk file verbatim
    from paf.capabilities.memory._render import write_render

    write_render(store, str(tmp_path / "MEMORY.md"))
    md2 = read_render(str(tmp_path))
    assert md2 == (tmp_path / "MEMORY.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# MemoryReviewQueue mechanics (fake span, no LLM)
# ---------------------------------------------------------------------------


def test_reviewer_toolset_with_journal_exposes_merge(tmp_path):
    from paf.capabilities.memory._journal import JournalStore

    rt = ReviewerToolset(OverrideStore(tmp_path), journal=JournalStore(tmp_path))
    assert "memory_review" in rt.tools
    assert "memory_merge" in rt.tools
    # The reviewer still has NO standalone append path (merge is its only write).
    assert "append_memory" not in rt.tools


# ---------------------------------------------------------------------------
# memory_merge: consolidate complementary fragments into one canonical line
# ---------------------------------------------------------------------------


def _seed_identity_cluster(tmp_path):
    """Three complementary fragments of the SAME identity fact (not duplicates)."""
    store = JournalStore(tmp_path)
    store.append_record(_mk_record("The user is named Landon", typ="identity"))
    store.append_record(_mk_record("Landon is married to Melanie", typ="identity"))
    store.append_record(
        _mk_record("Landon has three kids: Caleb, Elizabeth, Benjamin",
                   typ="identity")
    )
    build_index(tmp_path)
    return store


def test_memory_merge_appends_archives_and_rerenders(tmp_path):
    _seed_identity_cluster(tmp_path)
    rt = ReviewerToolset(OverrideStore(tmp_path), agent_name="assistant",
                         journal=JournalStore(tmp_path))

    result = asyncio.run(
        rt.memory_merge(
            _FakeCtx(),
            merge_ids=[1, 2, 3],
            fact=(
                "The user is Landon, married to Melanie, with three kids: "
                "Caleb, Elizabeth, Benjamin"
            ),
            tier="t1",
            type="identity",
            reason="same identity, one canonical line",
        )
    )
    assert result["status"] == "applied"
    assert result["op"] == "merge"
    assert result["archived"] == [1, 2, 3]

    # 1) The canonical record was appended to the journal, traceable to sources.
    journal = JournalStore(tmp_path)
    rows = journal.read_all()
    merged_id, merged = rows[-1]
    assert merged_id == result["merged_id"]
    assert "Landon, married to Melanie, with three kids" in merged.fact
    assert merged.source_ref == "merge(#1,#2,#3)"  # exactly the absorbed lines
    assert merged.tier == "t1"
    assert merged.type == "identity"

    # 2) The three sources were archived through the sidecar.
    archived = [r["target_line"] for r in OverrideStore(tmp_path).read_all()
                if r["op"] == "archive"]
    assert sorted(archived) == [1, 2, 3]

    # 3) Re-render collapsed the cluster: the merged line renders, the sources
    #    no longer appear in the injected view (they are archived).
    md = read_render(str(tmp_path))
    assert "married to Melanie" in md
    assert "named Landon" not in md  # source #1 dropped from injection
    assert "Caleb, Elizabeth, Benjamin" in md  # detail preserved in the merged line


def test_memory_merge_preserves_distinct_detail_into_one_render_line(tmp_path):
    """The merged fact must appear as a SINGLE line in the render — the whole
    point is that one canonical line replaces N parallel ones."""
    _seed_identity_cluster(tmp_path)
    rt = ReviewerToolset(OverrideStore(tmp_path), journal=JournalStore(tmp_path))

    asyncio.run(
        rt.memory_merge(
            _FakeCtx(),
            merge_ids=[1, 2, 3],
            fact=(
                "The user is Landon, married to Melanie, with three kids: "
                "Caleb, Elizabeth, Benjamin"
            ),
            tier="t1",
            type="identity",
        )
    )
    md = read_render(str(tmp_path))
    # The merged fact appears exactly once (no duplicate parallel lines remain).
    assert md.count("Landon, married to Melanie") == 1


def test_memory_merge_rejects_less_than_two_ids(tmp_path):
    _seed_identity_cluster(tmp_path)
    rt = ReviewerToolset(OverrideStore(tmp_path), journal=JournalStore(tmp_path))

    with pytest.raises(ModelRetry, match="at least 2"):  # single-line is not a merge
        asyncio.run(
            rt.memory_merge(
                _FakeCtx(), merge_ids=[1], fact="solo", tier="t1", type="identity"
            )
        )
    with pytest.raises(ModelRetry, match="at least 2"):  # empty is not a merge
        asyncio.run(
            rt.memory_merge(
                _FakeCtx(), merge_ids=[], fact="solo", tier="t1", type="identity"
            )
        )


def test_memory_merge_rejects_duplicate_ids(tmp_path):
    _seed_identity_cluster(tmp_path)
    rt = ReviewerToolset(OverrideStore(tmp_path), journal=JournalStore(tmp_path))
    with pytest.raises(ModelRetry, match="distinct"):
        asyncio.run(
            rt.memory_merge(
                _FakeCtx(), merge_ids=[1, 1], fact="dup", tier="t1", type="identity"
            )
        )


def test_memory_merge_requires_tier_and_type(tmp_path):
    _seed_identity_cluster(tmp_path)
    rt = ReviewerToolset(OverrideStore(tmp_path), journal=JournalStore(tmp_path))

    # An invalid/excluded tier is refused — tier is an explicit required placement.
    with pytest.raises(ModelRetry, match="invalid memory tier"):
        asyncio.run(
            rt.memory_merge(
                _FakeCtx(),
                merge_ids=[1, 2],
                fact="merged",
                tier="t9",
                type="identity",
            )
        )
    with pytest.raises(ModelRetry, match="invalid memory type"):
        asyncio.run(
            rt.memory_merge(
                _FakeCtx(),
                merge_ids=[1, 2],
                fact="merged",
                tier="t1",
                type="bogus",
            )
        )


def test_memory_merge_without_journal_is_unavailable(tmp_path):
    """A standalone ReviewerToolset with no journal exposes no merge — there is
    nowhere to write the canonical record (and no standalone append either)."""
    rt = ReviewerToolset(OverrideStore(tmp_path))
    assert set(rt.tools.keys()) == {"memory_review"}
    assert "memory_merge" not in rt.tools
    assert "append_memory" not in rt.tools


def test_memory_capability_reviewer_with_journal_wires_merge(tmp_path):
    from paf.capabilities.memory import Memory
    from paf.capabilities.memory._journal import JournalStore
    from paf.capabilities.memory._view import JournalViewStore

    m = Memory(
        store=JournalViewStore(tmp_path),
        agent_name="assistant",
        overrides=OverrideStore(tmp_path),
        reviewer=True,
        expose_tools=True,
        journal=JournalStore(tmp_path),
    )
    ts = m.get_toolset()
    assert isinstance(ts, ReviewerToolset)
    assert "memory_review" in ts.tools
    assert "memory_merge" in ts.tools
    assert "append_memory" not in ts.tools

    from paf.capabilities.memory._background import MemoryReviewQueue, ReviewJob
    from paf.capabilities.memory._watermark import MemoryWatermark

    calls = []

    async def fake_review(directory):
        calls.append(directory)
        return 3

    q = MemoryReviewQueue(
        memory_directory=str(tmp_path),
        agent_name="assistant",
        review_span=fake_review,
    )
    from datetime import datetime

    advance = MemoryWatermark(
        run_id="r1", turn_id=3, consumed_ts=datetime(2026, 9, 6, tzinfo=UTC)
    )

    class FakeState:
        pass

    async def run():
        await q.process_job("sess", ReviewJob(run_id="r1", turn_id=3, watermark=advance), FakeState())

    asyncio.run(run())
    assert calls == [str(tmp_path)]
    # Successful pass => the review watermark advanced to the job's target.
    wm = MemoryWatermark.load(str(tmp_path), run_id="r1", filename="memory-review-watermark.json")
    assert wm.consumed_ts is not None
    assert wm.turn_id == 3


def test_review_queue_requires_directory():
    from paf.capabilities.memory._background import MemoryReviewQueue, ReviewJob

    q = MemoryReviewQueue()  # missing directory
    called = []

    async def fake_review(directory):
        called.append(directory)
        return 0

    q.review_span = fake_review

    class FakeState:
        pass

    async def run():
        # Directory missing -> process_job drops before using the job's fields,
        # so an empty job is fine here.
        await q.process_job("sess", ReviewJob(run_id="r1", turn_id=1), FakeState())

    asyncio.run(run())
    assert called == []  # dropped without a directory


# ---------------------------------------------------------------------------
# Reviewer watermark contract (mirrors the owner's worker-owns-watermark rule)
# ---------------------------------------------------------------------------


def _mk_review_wm(tmp_path, run_id="r1", turn_id=1):
    from paf.capabilities.memory._watermark import MemoryWatermark

    MemoryWatermark(run_id=run_id, turn_id=turn_id).save(
        tmp_path, filename="memory-review-watermark.json"
    )
    return tmp_path


def test_review_throw_does_not_advance_watermark(tmp_path):
    """A review pass that THROWS must NOT advance the review watermark — it is
    left behind so the pass retries next cadence instead of being silently
    skipped (the same owner-watermark contract)."""
    from paf.capabilities.memory._background import MemoryReviewQueue, ReviewJob
    from paf.capabilities.memory._watermark import MemoryWatermark

    _mk_review_wm(tmp_path)

    async def failing_review(directory):
        raise RuntimeError("review LLM failed")

    q = MemoryReviewQueue(
        memory_directory=str(tmp_path), agent_name="assistant", review_span=failing_review
    )

    class FakeState:
        pass

    async def run():
        await q.process_job("sess", ReviewJob(run_id="r1", turn_id=7), FakeState())

    with pytest.raises(RuntimeError):
        asyncio.run(run())

    wm = MemoryWatermark.load(
        str(tmp_path), run_id="r1", filename="memory-review-watermark.json"
    )
    assert wm.turn_id == 1, "failed pass must leave the review watermark untouched"


def test_review_zero_verdicts_advances_watermark(tmp_path):
    """'Nothing to curate' (0 verdicts) is a SUCCESSFUL, complete review pass —
    the review watermark must advance anyway (completion, not content)."""
    from datetime import datetime

    from paf.capabilities.memory._background import MemoryReviewQueue, ReviewJob
    from paf.capabilities.memory._watermark import MemoryWatermark

    _mk_review_wm(tmp_path)

    async def empty_review(directory):
        return 0

    q = MemoryReviewQueue(
        memory_directory=str(tmp_path), agent_name="assistant", review_span=empty_review
    )
    advance = MemoryWatermark(
        run_id="r1",
        turn_id=7,
        consumed_ts=datetime(2026, 9, 6, tzinfo=UTC),
    )

    class FakeState:
        pass

    async def run():
        await q.process_job(
            "sess", ReviewJob(run_id="r1", turn_id=7, watermark=advance), FakeState()
        )

    asyncio.run(run())

    wm = MemoryWatermark.load(
        str(tmp_path), run_id="r1", filename="memory-review-watermark.json"
    )
    assert wm.turn_id == 7, "a 0-verdict pass is a completed pass and must advance"
    assert wm.consumed_ts is not None


# ---------------------------------------------------------------------------
# Session wiring: reviewer publishes a task-row that completes (like memory)
# ---------------------------------------------------------------------------


def test_session_review_span_publishes_task_done(tmp_path):
    """The Session's review queue seam flips the in-flight ``memory-review.*``
    task to DONE and publishes it — so the reviewer shows on the background-task
    timeline / Processes panel exactly like consolidation, not running silently.
    """
    import paf.background as _bgmod
    import paf.capabilities.memory._review as _revmod

    # Bare coordinator: only the fields the review seam touches.
    s = object.__new__(_bgmod.BackgroundCoordinator)
    s._background_tasks = {}
    s._task_subscribers = []
    s._consolidation_seq = 0
    s._thread_summarizer_seq = 0
    s._review_seq = 0
    s._review_queue = None
    s._memory_directory = str(tmp_path)
    s.agent_name = "assistant"
    s._run_events = None

    # Deterministic no-LLM review span (the seam references _revmod.review).
    async def noop_review(directory, *, overrides=None, agent_name=None):
        return 2

    _saved = _revmod.review
    _revmod.review = noop_review
    try:
        q = s._ensure_review_queue()
        assert q is not None
    finally:
        _revmod.review = _saved

    # Publish the queued task row (what _maybe_trigger_review now does).
    task = _bgmod.BackgroundTask(
        id="review.demo:x.1", label="Reviewing memory (demo:x)"
    )
    s._publish_task(task)

    seen: list[tuple[str, str]] = []
    s.subscribe_tasks(lambda ev: seen.append((ev.kind, ev.status)))

    async def run():
        assert q is not None
        await q.review_span(str(tmp_path))

    asyncio.run(run())

    # The in-flight row must transition to done and be visible to subscribers.
    assert task.status == "done", f"expected done, got {task.status!r}"
    assert ("task:Reviewing memory (demo:x)", "done") in seen

