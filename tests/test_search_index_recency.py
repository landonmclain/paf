"""Deterministic tests for `search_index` recency nudge.

The recency feature is a conservative, opt-configurable tie-breaker: when two
turns are relevance-tied, the more recent one wins. It must be bounded so it
never overrides a *meaningful* relevance gap, and it must be disable-able
(`recency_weight=0`) for exact BM25-order regression scenarios.

These tests control `ts` explicitly (bypassing the auto `_utcnow()` in
`TurnIndexStore.append`) so recency ordering is fully deterministic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_ai_harness.conversation_search._source import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import FileStepStore

from paf.capabilities.turn_index import TurnIndexStore, TurnIndexToolset
from paf.capabilities.turn_index._store import TurnIndexEntry, _content_hash

# ISO timestamps, newest last (so an anchor-aware recency boost favors turn 1).
_OLD_TS = "2026-07-01T00:00:00+00:00"
_NEW_TS = "2026-08-01T00:00:00+00:00"  # ~30 days newer (~1 half-life @ 30d)


class _FakeCtx:
    def __init__(self, conversation_id: str | None):
        self.conversation_id = conversation_id


def _seed_entries(
    tmp_path: Path, entries: list[dict], *, recency_weight: float = 0.1,
    recency_half_life_days: float = 30.0,
) -> TurnIndexToolset:
    """Build a toolset over a tmp store with EXPLICIT `ts` on each entry."""
    idx = TurnIndexStore(tmp_path / "index")
    for kwargs in entries:
        ts = kwargs.pop("ts")
        idx.service.append(
            "turn_index", idx.scan_root, idx._row(
                TurnIndexEntry(
                    conversation_id=kwargs.get("conversation_id", "c1"),
                    run_id=kwargs.get("run_id", "r1"),
                    turn_id=kwargs["turn_id"],
                    ts=ts,
                    summary=kwargs["summary"],
                    hash=_content_hash(
                        kwargs.get("conversation_id", "c1"),
                        kwargs.get("run_id", "r1"),
                        kwargs["turn_id"],
                        kwargs["summary"],
                    ),
                    key_terms=kwargs.get("key_terms"),
                    outcome=kwargs.get("outcome"),
                )
            ),
        )
    source = SnapshotHistorySource(FileStepStore(directory=tmp_path / "steps"))
    return TurnIndexToolset(
        source, idx, recency_weight=recency_weight,
        recency_half_life_days=recency_half_life_days,
    )


def run(ts: TurnIndexToolset, conv: str | None, query: str) -> str:
    return asyncio.run(ts.search_index(_FakeCtx(conv), query))


def _refs(out: str) -> list[str]:
    """Extract refs in the order they appear in the output."""
    refs = []
    for line in out.splitlines():
        if ": " not in line and line.startswith("["):
            refs.append(line.split("]")[0][1:])
        elif "] (score" in line:
            refs.append(line[1:].split("]")[0])
    return refs


# ---------------------------------------------------------------------------
# Recency as a tie-breaker
# ---------------------------------------------------------------------------


def test_recency_breaks_relevance_tie_in_favor_of_newer(tmp_path: Path) -> None:
    """Two turns matched identically on BM25 (same rare term) -> the newer one
    outranks the older, everything else equal."""
    ts = _seed_entries(
        tmp_path,
        [
            dict(conversation_id="c1", run_id="r1", turn_id=0, ts=_OLD_TS,
                 summary="the gazebo stands the lawn"),
            dict(conversation_id="c1", run_id="r1", turn_id=1, ts=_NEW_TS,
                 summary="another gazebo in the park"),
        ],
    )
    out = run(ts, "c1", "gazebo")
    assert out.index("r1:1") < out.index("r1:0"), out


def test_recency_does_not_override_meaningful_relevance_gap(tmp_path: Path) -> None:
    """A clearly more-relevant OLD turn still outranks a fresh NEW one: the
    nudge is bounded (max ~recency_weight) and cannot win a real gap.

    Same-length docs so length-normalization is neutral; the old turn matches
    the query term 2x (higher BM25) vs the new turn's 1x (lower BM25). The
    relevance edge must exceed the recency nudge.
    """
    ts = _seed_entries(
        tmp_path,
        [
            # old turn: query term appears 2x, short equal-length doc = higher BM25
            dict(conversation_id="c1", run_id="r1", turn_id=0, ts=_OLD_TS,
                 summary="the gazebo gazebo stands majestically"),
            # new turn: query term appears 1x, same-length doc = lower BM25
            dict(conversation_id="c1", run_id="r1", turn_id=1, ts=_NEW_TS,
                 summary="a single gazebo is what we saw"),
        ],
    )
    out = run(ts, "c1", "gazebo")
    # old turn (2x term) must outrank new turn (1x term) despite its age
    assert out.index("r1:0") < out.index("r1:1"), out


def test_recency_weight_zero_restores_pure_bm25(tmp_path: Path) -> None:
    """`recency_weight=0` disables the nudge entirely: a relevance-tied NEW turn
    no longer gains, so iteration/store order (which read_all sorts by ts,
    oldest-first) determines the result."""
    ts = _seed_entries(
        tmp_path,
        [
            dict(conversation_id="c1", run_id="r1", turn_id=0, ts=_OLD_TS,
                 summary="the gazebo stands the lawn"),
            dict(conversation_id="c1", run_id="r1", turn_id=1, ts=_NEW_TS,
                 summary="another gazebo in the park"),
        ],
        recency_weight=0.0,
    )
    out = run(ts, "c1", "gazebo")
    # read_all sorts ts ascending -> old turn is index 0 in entries[], and with
    # equal BM25 both are emitted in entries order (oldest first) => r1:0 first.
    assert out.index("r1:0") < out.index("r1:1"), out


def test_recency_half_life_flattens_nearby_timestamps(tmp_path: Path) -> None:
    """A very long half-life makes a 1-day gap negligible: the boost difference
    between two fresh turns is ~0, so a small BM25 edge still wins."""
    ts = _seed_entries(
        tmp_path,
        [
            dict(conversation_id="c1", run_id="r1", turn_id=0,
                 ts="2026-08-01T00:00:00+00:00",
                 summary="gazebo A"),  # BM25: 'gazebo' once
            dict(conversation_id="c1", run_id="r1", turn_id=1,
                 ts="2026-08-02T00:00:00+00:00",
                 summary="gazebo A gazebo"),  # BM25: 'gazebo' twice (higher)
        ],
        recency_half_life_days=3650.0,  # 10y: 1-day gap ~ decay 0.998
    )
    out = run(ts, "c1", "gazebo")
    # The BM25 edge (2x term freq) dominates over the ~0 recency difference.
    assert out.index("r1:1") < out.index("r1:0"), out
