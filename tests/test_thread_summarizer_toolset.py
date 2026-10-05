"""Deterministic tests for the `list_turn_summaries` thread-beat pager tool.

Covers the chronological pager contract: oldest->newest ordering, offset/limit
paging, header + summary + beats rendering, the inert `conversation_id` (must
match the active conversation or the call is rejected), and the empty / no-active
conversation cases. No model, no network — pure read of a synthetic store.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry

from paf.capabilities.thread_summarizer._store import (
    ThreadSummaryRecord,
    ThreadSummaryStore,
)
from paf.capabilities.thread_summarizer._toolset import ThreadSummaryToolset


def _run(coro):
    return asyncio.run(coro)


def _record(
    run: str,
    turn: int,
    cid: str,
    summary: str,
    beats: list[dict] | None = None,
    started_at: str = "2026-09-08T00:00:00+00:00",
    ended_at: str = "2026-09-08T00:30:00+00:00",
) -> ThreadSummaryRecord:
    return ThreadSummaryRecord(
        conversation_id=cid,
        run_id=run,
        turn_id=turn,
        summarized_at="2026-09-08T01:00:00+00:00",
        summary=summary,
        refs=[f"{run}:{turn}"],
        beats=beats or [],
        started_at=started_at,
        ended_at=ended_at,
    )


def _seed(root, cid: str, records: list[ThreadSummaryRecord]) -> None:
    store = ThreadSummaryStore(root, cid)
    for rec in records:
        store.append_summary(rec)


def _ctx(cid: str | None = "p:c1") -> SimpleNamespace:
    return SimpleNamespace(conversation_id=cid)


def test_lists_oldest_to_newest_with_header_summary_beats(tmp_path) -> None:
    root = tmp_path / "assistant"
    _seed(
        root,
        "p:c1",
        [
            _record(
                "r1",
                0,
                "p:c1",
                "first block arc",
                beats=[{"at": "r1:0", "what": "started the work"}],
            ),
            _record(
                "r2",
                0,
                "p:c1",
                "second block arc",
                beats=[
                    {"at": "r2:0", "what": "hit a snag"},
                    {"at": "r2:2", "what": "fixed it"},
                ],
            ),
        ],
    )
    ts = ThreadSummaryToolset(root)
    out = _run(ts.list_turn_summaries(_ctx()))
    # oldest first
    assert out.index("first block arc") < out.index("second block arc")
    # header present
    assert "ago" in out
    # beats rendered with at refs
    assert "· r1:0  started the work" in out
    assert "· r2:0  hit a snag" in out
    assert "· r2:2  fixed it" in out


def test_offset_limit_paging(tmp_path) -> None:
    root = tmp_path / "assistant"
    _seed(
        root,
        "p:c1",
        [_record(f"r{i}", 0, "p:c1", f"block {i}") for i in range(5)],
    )
    ts = ThreadSummaryToolset(root)
    # page 1: blocks 0,1
    out1 = _run(ts.list_turn_summaries(_ctx(), offset=0, limit=2))
    assert "block 0" in out1 and "block 1" in out1
    assert "block 2" not in out1
    assert "3 more blocks" in out1
    # page 2: blocks 2,3
    out2 = _run(ts.list_turn_summaries(_ctx(), offset=2, limit=2))
    assert "block 2" in out2 and "block 3" in out2
    assert "block 0" not in out2
    # last page: block 4, no more pointer
    out3 = _run(ts.list_turn_summaries(_ctx(), offset=4, limit=2))
    assert "block 4" in out3
    assert "more blocks" not in out3


def test_inert_conversation_id_must_match_active(tmp_path) -> None:
    root = tmp_path / "assistant"
    _seed(root, "p:c1", [_record("r1", 0, "p:c1", "block")])
    ts = ThreadSummaryToolset(root)
    # matching conversation_id is accepted (inert)
    out = _run(ts.list_turn_summaries(_ctx("p:c1"), conversation_id="p:c1"))
    assert "block" in out
    # mismatching conversation_id is rejected, not silently read
    with pytest.raises(ModelRetry):
        _run(ts.list_turn_summaries(_ctx("p:c1"), conversation_id="p:other"))


def test_no_active_conversation(tmp_path) -> None:
    ts = ThreadSummaryToolset(tmp_path / "assistant")
    out = _run(ts.list_turn_summaries(_ctx(None)))
    assert "no active conversation" in out


def test_no_summaries_yet(tmp_path) -> None:
    ts = ThreadSummaryToolset(tmp_path / "assistant")
    out = _run(ts.list_turn_summaries(_ctx("p:c1")))
    assert "no thread summaries" in out


def test_invalid_args_raise_model_retry(tmp_path) -> None:
    ts = ThreadSummaryToolset(tmp_path / "assistant")
    with pytest.raises(ModelRetry):
        _run(ts.list_turn_summaries(_ctx(), offset=-1))
    with pytest.raises(ModelRetry):
        _run(ts.list_turn_summaries(_ctx(), limit=0))


def test_capability_exposes_toolset() -> None:
    from paf.capabilities.thread_summarizer import ThreadSummarizer

    cap = ThreadSummarizer(agent_name="assistant")
    ts = cap.get_toolset()
    assert ts is not None
    assert hasattr(ts, "list_turn_summaries")
