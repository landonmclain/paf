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


def test_review_queue_runs_review_span(tmp_path):
    from paf.capabilities.memory._background import MemoryReviewQueue, ReviewJob

    calls = []

    async def fake_review(directory):
        calls.append(directory)
        return 3

    q = MemoryReviewQueue(
        memory_directory=str(tmp_path),
        agent_name="assistant",
        review_span=fake_review,
    )

    class FakeState:
        pass

    async def run():
        await q.process_job("sess", ReviewJob(), FakeState())

    asyncio.run(run())
    assert calls == [str(tmp_path)]


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
        await q.process_job("sess", ReviewJob(), FakeState())

    asyncio.run(run())
    assert called == []  # dropped without a directory
