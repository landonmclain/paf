"""Deterministic tests for `search_index` BM25-over-full-record.

These assert the *ranking behavior* the tool now guarantees, with no LLM:
the tool seeds a `TurnIndexStore`, runs `search_index(query)` directly, and we
check which refs rank where. The intent is a thorough regression guard for the
rework from whole-query-`in`-substring matching to BM25 over the full record
(`summary` + `key_terms` + `outcome`).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic_ai_harness.conversation_search._source import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import FileStepStore

from paf.capabilities.turn_index import TurnIndexStore, TurnIndexToolset


class _FakeCtx:
    def __init__(self, conversation_id: str | None):
        self.conversation_id = conversation_id


def _seed(tmp_path: Path, *appends) -> tuple[SnapshotHistorySource, TurnIndexStore, TurnIndexToolset]:
    """Build an isolated toolset over a tmp store; each `append` is kwargs for `idx.append`."""
    idx = TurnIndexStore(tmp_path / "index")
    for kwargs in appends:
        idx.append(**kwargs)
    source = SnapshotHistorySource(FileStepStore(directory=tmp_path / "steps"))
    ts = TurnIndexToolset(source, idx)
    return source, idx, ts


def run(ts: TurnIndexToolset, conv: str | None, query: str, **kw) -> str:
    return asyncio.run(ts.search_index(_FakeCtx(conv), query))


# ---------------------------------------------------------------------------
# Core: token-overlap (the rework's headline fix) and field coverage
# ---------------------------------------------------------------------------


def test_multiword_phrase_now_matches_without_contiguous_substring(tmp_path: Path) -> None:
    """The regression this rework fixes: a multi-word query no longer needs to be a
    contiguous substring of summary/key_terms. 'maria workout december' — split
    across summary and key_terms — must surface the turn."""
    _, _, ts = _seed(
        tmp_path,
        dict(conversation_id="c1", run_id="r1", turn_id=0, summary="planned a workout routine", key_terms=["maria", "december"]),
        dict(conversation_id="c1", run_id="r1", turn_id=1, summary="talked about the weather"),
    )
    out = run(ts, "c1", "maria workout december")
    assert "r1:0" in out, out
    assert "planned a workout routine" in out, out


def test_whole_phrase_query_still_finds_contiguous_terms(tmp_path: Path) -> None:
    """A natural full phrase still works when its tokens actually appear."""
    _, _, ts = _seed(
        tmp_path,
        dict(conversation_id="c1", run_id="r1", turn_id=0, summary="aerial yoga class on wednesday"),
        dict(conversation_id="c1", run_id="r1", turn_id=1, summary="wrote documentation"),
    )
    out = run(ts, "c1", "aerial yoga wednesday")
    assert "r1:0" in out, out
    assert "r1:1" not in out, out


def test_key_terms_surface_abstract_summary(tmp_path: Path) -> None:
    """A term present only in `key_terms` (not the one-liner) still surfaces the turn."""
    _, _, ts = _seed(
        tmp_path,
        dict(conversation_id="c1", run_id="r1", turn_id=0, summary="migrated the database", key_terms=["postgres", "row-level security"]),
        dict(conversation_id="c1", run_id="r1", turn_id=1, summary="fixed a typo"),
    )
    out = run(ts, "c1", "postgres")
    assert "r1:0" in out, out


def test_outcome_is_searchable(tmp_path: Path) -> None:
    """`outcome` tokens rank a turn whose keyword is otherwise absent."""
    _, _, ts = _seed(
        tmp_path,
        dict(conversation_id="c1", run_id="r1", turn_id=0, summary="deployed the schema", outcome="error"),
        dict(conversation_id="c1", run_id="r1", turn_id=1, summary="updated the docs", outcome="resolved"),
    )
    out = run(ts, "c1", "error")
    assert "r1:0" in out, out
    assert "r1:1" not in out, out


def test_rare_term_outranks_common_term(tmp_path: Path) -> None:
    """IDF: the distinctive 'gazebo' (in one doc) outranks the ubiquitous 'the'."""
    _, _, ts = _seed(
        tmp_path,
        dict(conversation_id="c1", run_id="r1", turn_id=0, summary="the quick brown fox jumps the log"),
        dict(conversation_id="c1", run_id="r1", turn_id=1, summary="the gazebo stands the lawn"),
    )
    out = run(ts, "c1", "the gazebo")
    # 'gazebo' is in exactly one doc (rare, high IDF); 'the' is in both (common,
    # low IDF). The rare-term match must rank above the common-only match.
    assert out.index("r1:1") < out.index("r1:0"), out


# ---------------------------------------------------------------------------
# Anchoring: summary dominates key_terms/outcome
# ---------------------------------------------------------------------------


def test_summary_anchor_outranks_key_term_only(tmp_path: Path) -> None:
    """When a term is in one turn's summary and only another's key_terms, the
    summary-anchored turn ranks first (summary text is weighted by repetition)."""
    _, _, ts = _seed(
        tmp_path,
        # turn 0: summary carries the term
        dict(conversation_id="c1", run_id="r1", turn_id=0, summary="the postgres migration is stuck"),
        # turn 1: the term is only a keyword, sentence phrases it abstractly
        dict(conversation_id="c1", run_id="r1", turn_id=1, summary="we upgraded the data store", key_terms=["postgres"]),
    )
    out = run(ts, "c1", "postgres")
    assert out.index("r1:0") < out.index("r1:1"), out


def test_anchor_applies_within_one_turn_vs_disjoint_keyword(tmp_path: Path) -> None:
    """Even with equal IDF across the corpus, summary text outranks a keyword:
    a turn matching on its one-liner scores above one matching only on outcome."""
    _, _, ts = _seed(
        tmp_path,
        dict(conversation_id="c1", run_id="r1", turn_id=0, summary="verified the backup works", key_terms=["restore"]),
        dict(conversation_id="c1", run_id="r1", turn_id=1, summary="nothing to see here", outcome="resolved", key_terms=["verified"]),
    )
    out = run(ts, "c1", "restore verified backup")
    # the term 'verified' appears in turn 0 summary and turn 1 key_terms; turn 0's
    # summary-anchored copy must outrank turn 1's keyword-only copy for that term.
    assert out.index("r1:0") < out.index("r1:1"), out


# ---------------------------------------------------------------------------
# Capping, scoping, edge handling
# ---------------------------------------------------------------------------


def test_search_index_caps_results(tmp_path: Path) -> None:
    """Returns at most `max_matches` hits, rank-ordered."""
    appends = []
    for i in range(8):
        appends.append(dict(conversation_id="c1", run_id="r1", turn_id=i, summary=f"turn number {i} about deploy"))
    _, _, ts = _seed(tmp_path, *appends)
    out = run(ts, "c1", "deploy")
    assert out.count("r1:") == 5, out  # default max_matches


def test_search_index_respects_custom_max_matches(tmp_path: Path) -> None:
    """Constructor `max_matches` bounds the result set."""
    idx = TurnIndexStore(tmp_path / "index")
    for i in range(6):
        idx.append(conversation_id="c1", run_id="r1", turn_id=i, summary=f"everything has the word zebra {i}")
    source = SnapshotHistorySource(FileStepStore(directory=tmp_path / "steps"))
    ts = TurnIndexToolset(source, idx, max_matches=2)
    out = run(ts, "c1", "zebra")
    assert out.count("r1:") == 2, out


def test_search_index_scopes_to_conversation(tmp_path: Path) -> None:
    """Only the CURRENT conversation's entries are searched."""
    _, _, ts = _seed(
        tmp_path,
        dict(conversation_id="c1", run_id="r1", turn_id=0, summary="alpha secret", key_terms=["leak"]),
        dict(conversation_id="c2", run_id="r2", turn_id=0, summary="alpha secret", key_terms=["leak"]),
    )
    out = run(ts, "c1", "leak")
    assert out.count("alpha secret") == 1, out
    assert "r1:0" in out and "r2:0" not in out, out


def test_search_index_fails_closed_without_conversation_id(tmp_path: Path) -> None:
    """An unscoped run must NOT search every conversation's index."""
    _, _, ts = _seed(
        tmp_path,
        dict(conversation_id="c1", run_id="r1", turn_id=0, summary="alpha secret", key_terms=["leak"]),
    )
    out = run(ts, None, "leak")
    assert "conversation id" in out.lower(), out
    assert "alpha secret" not in out, out


def test_search_index_empty_index(tmp_path: Path) -> None:
    """No indexed turns -> a clear message (not a silent empty string)."""
    _, _, ts = _seed(tmp_path)
    out = run(ts, "c1", "anything")
    assert "No turn index entries" in out, out


def test_search_index_no_coverage_escalates(tmp_path: Path) -> None:
    """Zero shared tokens -> explicit escalate-to-search_conversation_history."""
    _, _, ts = _seed(
        tmp_path,
        dict(conversation_id="c1", run_id="r1", turn_id=0, summary="about gardening and soil"),
        dict(conversation_id="c1", run_id="r1", turn_id=1, summary="cooking pasta"),
    )
    out = run(ts, "c1", "quantum entanglement")
    assert "No turn index matches" in out, out
    assert "search_conversation_history" in out, out


# ---------------------------------------------------------------------------
# Output shape & validation
# ---------------------------------------------------------------------------


def test_search_index_output_is_rank_annotated(tmp_path: Path) -> None:
    """Hits carry a relevance score and the ref, so the model can decide/act."""
    _, _, ts = _seed(
        tmp_path,
        dict(conversation_id="c1", run_id="r1", turn_id=3, summary="the wifi keeps dropping"),
    )
    out = run(ts, "c1", "wifi")
    assert "BM25-ranked" in out, out
    assert "score" in out, out
    assert "r1:3" in out, out


def test_search_index_reject_bad_max_matches(tmp_path: Path) -> None:
    idx = TurnIndexStore(tmp_path / "index")
    source = SnapshotHistorySource(FileStepStore(directory=tmp_path / "steps"))
    with pytest.raises(ValueError):
        TurnIndexToolset(source, idx, max_matches=0)


def test_search_index_case_insensitive(tmp_path: Path) -> None:
    """Tokenization lowercases both sides, so case never affects matching."""
    _, _, ts = _seed(
        tmp_path,
        dict(conversation_id="c1", run_id="r1", turn_id=0, summary="OAuth token expired at noon"),
    )
    for q in ("oauth", "OAuth", "OAUTH"):
        out = run(ts, "c1", q)
        assert "r1:0" in out, f"{q!r} -> {out}"


def test_searchable_document_includes_summary_key_terms_outcome() -> None:
    """Unit-level: the flattened document carries all three fields."""
    from paf.capabilities.turn_index._toolset import _searchable_text
    from paf.capabilities.turn_index._store import TurnIndexEntry

    e = TurnIndexEntry(
        conversation_id="c1", run_id="r1", turn_id=0, ts="2026-01-01T00:00:00+00:00",
        summary="the garden bloomed", hash="h",
        key_terms=["aerial yoga", "community"], outcome="resolved",
    )
    doc = _searchable_text(e)
    # summary appears twice (anchoring)
    assert doc.count("the garden bloomed") == 2, doc
    # key_terms flattened as tokens, outcome present
    assert "aerial" in doc and "yoga" in doc, doc
    assert "community" in doc, doc
    assert "resolved" in doc, doc
