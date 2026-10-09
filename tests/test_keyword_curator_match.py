"""Keyword curator Phase 3a: the hot-path matcher.

Synthetic indexes, so each rule is pinned independent of any real store: terms
come from a source's OWN record, bare entity names are discounted by the size of
the entity, multi-word terms are specific, compound queries still find sources,
and unrelated messages return nothing.
"""

from __future__ import annotations

import math

from paf.capabilities.keyword_curator import (
    KeywordIndex,
    KeywordJournalStore,
    make_record,
    write_entity_map,
)
from paf.capabilities.keyword_curator._match import SKILL_BOOST, name_term, tokens


def _rec(ref, entities, keywords=()):
    return make_record(ref, ref.split(":")[0], keywords, entities)


def _idx(*recs, live=None):
    return KeywordIndex.from_records({r["source_ref"]: r for r in recs}, live)


QC = {"canonical": "quantconnect", "aliases": ["qc"]}


def _qc_family(n=6):
    return [
        _rec(f"skill:qc-{i}", [QC], [f"feature {i}", "live logs read"][:1])
        for i in range(n)
    ]


def test_tokens_lowercase_alnum_and_light_stem() -> None:
    assert tokens("Live/Logs-READ v2") == ["live", "log", "read", "v2"]
    assert tokens("symphonies") == ["symphony"]
    assert tokens("class") == ["class"]  # 'ss' is not a plural
    assert tokens("") == []


def test_name_term_uses_skill_leaf_and_note_stem() -> None:
    assert name_term("skill:trading/quantconnect-api") == "quantconnect-api"
    assert name_term("note:Designs/Keyword-Curator.md") == "Keyword-Curator"


def test_matches_only_through_terms_a_source_declared_itself() -> None:
    a = _rec(
        "skill:a", [{"canonical": "quantconnect", "aliases": ["qc deploy"]}], ["lean"]
    )
    b = _rec("skill:b", [{"canonical": "quantconnect", "aliases": ["qc research"]}])
    idx = _idx(a, b)
    hits = idx.match("help with the qc deploy step")
    assert [h.ref for h in hits] == ["skill:a"]  # b shares the entity but not the term
    assert "qc deploy" in hits[0].matched


def test_unrelated_and_stopword_messages_return_nothing() -> None:
    idx = _idx(*_qc_family())
    for msg in ["", "   ", "thanks, that looks good", "what is the weather", "ok"]:
        assert idx.match(msg) == [], msg


def test_a_specific_phrase_outranks_a_bare_name_shared_by_the_whole_entity() -> None:
    fam = _qc_family(8)
    special = _rec(
        "note:live-log-access.md",
        [{"canonical": "quantconnect", "aliases": ["qc live logs"]}],
    )
    idx = _idx(*fam, special)
    hits = idx.match("check the live logs on my QC algo")
    assert hits[0].ref == "note:live-log-access.md"


def test_bare_alias_is_discounted_by_the_size_of_its_entity() -> None:
    # 10 sources: 6 under entity "big" (only 2 spell the alias "bg"), 1 under
    # "small" (spells "sm"), 3 filler. Both aliases are declared by few sources, but
    # "bg" names a 6-source entity, so a hit on it is far less specific.
    recs = [
        _rec("skill:b0", [{"canonical": "big", "aliases": ["bg"]}]),
        _rec("skill:b1", [{"canonical": "big", "aliases": ["bg"]}]),
        *[_rec(f"skill:b{i}", [{"canonical": "big"}]) for i in range(2, 6)],
        _rec("skill:s0", [{"canonical": "small", "aliases": ["sm"]}]),
        *[_rec(f"skill:f{i}", [{"canonical": f"filler{i}"}]) for i in range(3)],
    ]
    idx = _idx(*recs)
    bg = idx.match("bg", min_score=0.0)[0].score
    sm = idx.match("sm", min_score=0.0)[0].score
    assert sm > bg
    # df = the entity's size (6), not the 2 sources that spell the alias; these are
    # skills, so the modest skill preference applies on top
    assert bg == round(3.0 * math.log(1 + 10 / 6) * SKILL_BOOST, 2)


def test_subphrase_of_a_longer_term_matches_at_reduced_weight() -> None:
    idx = _idx(_rec("note:x.md", [{"canonical": "thing"}], ["live logs read access"]))
    whole = idx.match("live logs read access", min_score=0.0)[0].score
    part = idx.match("the live logs", min_score=0.0)[0].score
    assert 0 < part < whole


def test_token_tier_needs_two_distinct_content_words() -> None:
    idx = _idx(
        _rec(
            "note:n.md", [{"canonical": "thing"}], ["deploy pipeline staging cluster"]
        ),
        _rec("note:o.md", [{"canonical": "other"}], ["unrelated words here"]),
    )
    assert idx.match("the staging cluster", min_score=0.1)[0].ref == "note:n.md"
    assert idx.match("something about cluster", min_score=0.1) == []  # one word only


def test_floor_relative_cutoff_and_top_k() -> None:
    recs = [
        _rec("skill:strong", [{"canonical": "alpha", "aliases": ["alpha beta gamma"]}]),
        _rec("skill:weak", [{"canonical": "beta"}], ["gamma"]),
        *[_rec(f"skill:f{i}", [{"canonical": f"f{i}"}]) for i in range(8)],
    ]
    idx = _idx(*recs)
    hits = idx.match("alpha beta gamma", min_score=0.1, relative=0.0)
    assert hits[0].ref == "skill:strong"
    assert idx.match("alpha beta gamma", min_score=0.1, relative=0.99)[0].ref == (
        "skill:strong"
    )
    assert len(idx.match("alpha beta gamma", min_score=0.1, relative=0.0, top_k=1)) == 1
    assert idx.match("alpha beta gamma", min_score=10_000) == []


def test_deterministic_order_ties_break_by_ref() -> None:
    idx = _idx(
        _rec("skill:b", [{"canonical": "same", "aliases": ["x"]}]),
        _rec("skill:a", [{"canonical": "same", "aliases": ["x"]}]),
    )
    assert [h.ref for h in idx.match("same", min_score=0.1)] == ["skill:a", "skill:b"]


def test_hint_carries_kind_label_and_matched_terms() -> None:
    idx = _idx(_rec("note:Dir/Foo.md", [{"canonical": "foo", "aliases": ["the foo"]}]))
    h = idx.match("tell me about foo", min_score=0.1)[0]
    assert (h.kind, h.label, h.ref) == ("note", "Dir/Foo.md", "note:Dir/Foo.md")
    assert h.matched and h.score > 0


def test_live_refs_drop_dead_sources() -> None:
    a = _rec("skill:a", [{"canonical": "alpha"}])
    b = _rec("skill:b", [{"canonical": "alpha"}])
    idx = _idx(a, b, live={"skill:a"})
    assert idx.size == 1
    assert [h.ref for h in idx.match("alpha", min_score=0.1)] == ["skill:a"]


def test_from_store_uses_the_entity_map_for_liveness(tmp_path) -> None:
    j = KeywordJournalStore(tmp_path)
    j.append(_rec("skill:a", [{"canonical": "alpha"}]))
    j.append(_rec("skill:gone", [{"canonical": "alpha"}]))
    write_entity_map(
        tmp_path,
        {"alpha": {"canonical": "alpha", "aliases": [], "sources": ["skill:a"]}},
    )
    idx = KeywordIndex.from_store(tmp_path)
    assert [h.ref for h in idx.match("alpha", min_score=0.1)] == ["skill:a"]


def test_from_store_without_a_map_keeps_every_journaled_source(tmp_path) -> None:
    KeywordJournalStore(tmp_path).append(_rec("skill:a", [{"canonical": "alpha"}]))
    assert KeywordIndex.from_store(tmp_path).size == 1
    assert KeywordIndex.from_store(tmp_path / "nope").size == 0


def test_empty_index_matches_nothing() -> None:
    assert KeywordIndex().match("anything at all") == []


def test_a_skill_wins_a_tie_against_notes_with_the_same_terms() -> None:
    same = [{"canonical": "longmem", "aliases": ["longmem benchmark"]}]
    idx = _idx(
        _rec("note:A-first.md", same),
        _rec("note:B-second.md", same),
        _rec("skill:run-longmem", same),
    )
    hits = idx.match("run the longmem benchmark", min_score=0.1)
    assert hits[0].ref == "skill:run-longmem"  # not 'note:' just for sorting first


def test_the_skill_preference_does_not_override_a_much_stronger_note() -> None:
    idx = _idx(
        _rec("skill:weak", [{"canonical": "thing", "aliases": []}]),
        _rec(
            "note:strong.md",
            [{"canonical": "thing", "aliases": ["thing deep dive notes"]}],
            ["thing deep dive notes"],
        ),
    )
    hits = idx.match("thing deep dive notes", min_score=0.1)
    assert hits[0].ref == "note:strong.md"
