"""Deterministic derived memory index (``_index.py``): journal -> survivors.

Covers the incremental fold (exact + token-set dedup, keep-winner rule),
the watermark advance (no full re-scan on a second fold), the one-time full
build with fuzzy cluster collapse, and the on-disk roundtrip. All deterministic,
no model needed.
"""

from __future__ import annotations

import json

from paf.capabilities.memory._index import (
    MemoryIndex,
    build_index,
    fact_keys,
    update_index,
)
from paf.capabilities.memory._journal import JournalStore, MemoryRecord


def _mk_record(fact, *, tier="t1", born="2026-08-17T00:00:00Z", src="run:0", typ="fact"):
    return MemoryRecord(type=typ, fact=fact, tier=tier, born_on=born, source_ref=src)


def _seed_store(tmp_path, records) -> JournalStore:
    store = JournalStore(tmp_path)
    for r in records:
        store.append_record(r)
    return store


def _load(tmp_path) -> MemoryIndex:
    return MemoryIndex.load(tmp_path)


def _index_path(tmp_path) -> str:
    return str(tmp_path / "memory-index.json")


# --------------------------------------------------------------------------
# fact_keys (the O(1) lookup keys)
# --------------------------------------------------------------------------

def test_fact_keys_exact_normalizes_case_and_whitespace():
    k1 = fact_keys("User prefers mobile-first web design")
    k2 = fact_keys("user prefers mobile-first web design")
    assert k1["exact"] == k2["exact"]


def test_fact_keys_token_set_catches_reworded_same_tokens():
    k1 = fact_keys("prefers mobile-first design for the site")
    k2 = fact_keys("design mobile-first prefers for the site")  # re-ordered tokens
    assert k1["tokens"] == k2["tokens"]
    assert k1["exact"] != k2["exact"]  # but exact differs


def test_fact_keys_stopwords_dropped():
    k1 = fact_keys("a the of in user prefers mobile-first")
    k2 = fact_keys("user prefers mobile-first")
    # the token-set key drops stopwords, so both normalize to the same tokens
    assert k1["tokens"] == k2["tokens"]
    # the exact key is byte-true (keeps stopwords) — it is NOT the fuzzy key
    assert k1["exact"] != k2["exact"]


# --------------------------------------------------------------------------
# incremental fold
# --------------------------------------------------------------------------

def test_incremental_exact_dup_absorbed(tmp_path):
    _seed_store(tmp_path, [
        _mk_record("User prefers mobile-first design", src="run:0"),
        _mk_record("User prefers mobile-first design", src="run:1"),
    ])
    update_index(tmp_path)
    idx = _load(tmp_path)
    assert len(idx.entries) == 1
    assert idx.watermark == 2
    # identical (tier, born_on) -> tie-break makes the newer line the winner
    assert idx.entries[0]["current_line"] == 2
    assert idx.entries[0]["deduped_lines"] == [1]


def test_incremental_token_reorder_dup_absorbed(tmp_path):
    _seed_store(tmp_path, [
        _mk_record("design mobile-first prefers", src="run:0"),
        _mk_record("prefers mobile-first design", src="run:1"),
    ])
    update_index(tmp_path)
    assert len(_load(tmp_path).entries) == 1


def test_keep_winner_tier_then_recency_newer_t1_wins(tmp_path):
    # newer t1 beats the older t2 (tier first, then most-recent)
    _seed_store(tmp_path, [
        _mk_record("same fact", tier="t2", born="2026-08-17T00:00:00Z", src="run:0"),
        _mk_record("same fact", tier="t1", born="2026-08-17T00:01:00Z", src="run:1"),
    ])
    update_index(tmp_path)
    e = _load(tmp_path).entries[0]
    assert e["tier"] == "t1"
    assert e["current_line"] == 2
    assert e["deduped_lines"] == [1]


def test_keep_winner_same_tier_recency(tmp_path):
    # same tier -> more-recent born_on wins
    _seed_store(tmp_path, [
        _mk_record("same fact", tier="t1", born="2026-08-17T00:00:00Z", src="run:0"),
        _mk_record("same fact", tier="t1", born="2026-08-17T00:01:00Z", src="run:1"),
    ])
    update_index(tmp_path)
    assert _load(tmp_path).entries[0]["current_line"] == 2


def test_keep_winner_tier_wins_over_recency(tmp_path):
    # much newer lower-tier does NOT beat a stable t1
    _seed_store(tmp_path, [
        _mk_record("same fact", tier="t1", born="2026-08-17T00:00:00Z", src="run:0"),
        _mk_record("same fact", tier="t3", born="2026-09-01T00:00:00Z", src="run:1"),
    ])
    update_index(tmp_path)
    e = _load(tmp_path).entries[0]
    assert e["tier"] == "t1"
    assert e["current_line"] == 1


def test_incremental_distinct_facts_separate_winners(tmp_path):
    _seed_store(tmp_path, [
        _mk_record("User prefers mobile-first design", src="run:0"),
        _mk_record("User's site organized into four arcs", src="run:1"),
    ])
    update_index(tmp_path)
    assert len(_load(tmp_path).entries) == 2


def test_watermark_advances_incrementally_no_rescan(tmp_path):
    store = _seed_store(tmp_path, [_mk_record("fact one", src="run:0")])
    update_index(tmp_path)
    assert _load(tmp_path).watermark == 1

    # a second fold only processes the new line; prior winner survives
    store.append_record(_mk_record("fact two", src="run:1"))
    update_index(tmp_path)
    idx = _load(tmp_path)
    assert idx.watermark == 2
    assert {e["current_line"] for e in idx.entries} == {1, 2}


# --------------------------------------------------------------------------
# update_index (top-level incremental entry point)
# --------------------------------------------------------------------------

def test_update_index_builds_then_increments(tmp_path):
    store = _seed_store(tmp_path, [_mk_record("User prefers mobile-first design", src="run:0")])
    assert update_index(tmp_path) == 1  # first call -> full build

    store.append_record(_mk_record("User prefers mobile-first design", src="run:1"))  # exact dup
    assert update_index(tmp_path) == 1  # incremental fold, absorbed
    idx = _load(tmp_path)
    assert idx.watermark == 2
    assert idx.entries[0]["current_line"] == 2
    assert idx.entries[0]["deduped_lines"] == [1]


# --------------------------------------------------------------------------
# full build (one-time, journal with pre-existing history, fuzzy collapse)
# --------------------------------------------------------------------------

def test_build_index_collapses_fuzzy_cluster(tmp_path):
    # near-verbatim paraphrase of the same fact (different words, not exact)
    _seed_store(tmp_path, [
        _mk_record(
            "User prefers mobile-first design because most readers browse on their phones",
            tier="t1", src="run:0",
        ),
        _mk_record(
            "User prefers mobile-first web design because about 80 percent of readers "
            "browse and read on their phone",
            tier="t2", src="run:1",
        ),
    ])
    n = build_index(tmp_path)
    assert n == 1  # fuzzy-clustered to a single living fact (t1 wins)
    e = _load(tmp_path).entries[0]
    assert e["tier"] == "t1"
    assert sorted(e["deduped_lines"]) == [2]


def test_build_index_preserves_distinct_facts(tmp_path):
    _seed_store(tmp_path, [
        _mk_record("User prefers mobile-first design", src="run:0"),
        _mk_record("User's site organized into four arcs", src="run:1"),
    ])
    build_index(tmp_path)
    assert len(_load(tmp_path).entries) == 2


def test_build_index_idempotent_and_persists_schema(tmp_path):
    _seed_store(tmp_path, [
        _mk_record("fact one", src="run:0"),
        _mk_record("fact two", src="run:1"),
    ])
    n1 = build_index(tmp_path)
    n2 = build_index(tmp_path)
    assert n1 == n2 == 2
    raw = json.loads(open(_index_path(tmp_path)).read())
    assert raw["schema"] == 1
    assert raw["watermark"] == 2


def test_build_index_empty_journal(tmp_path):
    build_index(tmp_path)
    assert _load(tmp_path).watermark == 0
    assert _load(tmp_path).entries == []
