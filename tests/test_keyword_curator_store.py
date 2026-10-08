"""Keyword curator Phase 1: the journal, entity-map derivation, and watermark.

No model and no capability wiring — pure storage behaviour: append-only
idempotency, torn-line tolerance, latest-wins derivation with liveness, alias
merging, and mtime+size change detection.
"""

from __future__ import annotations

import json
import threading

from paf.capabilities.keyword_curator import (
    ENTITY_MAP_FILENAME,
    JOURNAL_FILENAME,
    WATERMARK_FILENAME,
    KeywordJournalStore,
    KeywordWatermark,
    derive_entity_map,
    make_record,
    normalize_term,
    read_entity_map,
    write_entity_map,
)

QC = {"canonical": "quantconnect", "aliases": ["qc", "quantconnect-api"]}
PAF = {"canonical": "paf", "aliases": ["the framework"]}


def _rec(
    ref, entities, keywords=(), source_type="skill", at="2026-10-08T00:00:00+00:00"
):
    return make_record(ref, source_type, keywords, entities, at=at)


# --- normalize_term / make_record --------------------------------------------


def test_normalize_term_folds_case_and_separators() -> None:
    assert normalize_term("Memory-Extractor") == "memory extractor"
    assert normalize_term("  memory__extractor ") == "memory extractor"
    assert normalize_term("") == ""


def test_make_record_normalizes_and_dedupes() -> None:
    rec = make_record(
        " skill:qc ",
        "skill",
        ["QuantConnect", "quantconnect", " lean-cli ", ""],
        [
            {"canonical": "quantconnect", "aliases": ["QC", "qc", "QuantConnect"]},
            {"canonical": "  ", "aliases": ["x"]},
            "not-a-dict",
        ],
    )
    assert rec["source_ref"] == "skill:qc"
    assert rec["keywords"] == ["QuantConnect", "lean-cli"]
    # alias equal to the canonical is dropped; case-duplicates collapse
    assert rec["entities"] == [{"canonical": "quantconnect", "aliases": ["QC"]}]
    assert rec["at"]


def test_make_record_rejects_empty_ref() -> None:
    try:
        make_record("  ", "skill")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# --- journal -----------------------------------------------------------------


def test_append_and_read_all_roundtrip(tmp_path) -> None:
    store = KeywordJournalStore(tmp_path)
    assert store.read_all() == []
    assert store.append(_rec("skill:qc", [QC], ["lean-cli"])) is True
    got = store.read_all()
    assert [r["source_ref"] for r in got] == ["skill:qc"]
    assert (tmp_path / JOURNAL_FILENAME).read_text().endswith("\n")


def test_append_is_idempotent_by_content(tmp_path) -> None:
    store = KeywordJournalStore(tmp_path)
    assert store.append(_rec("skill:qc", [QC], at="2026-10-01T00:00:00+00:00"))
    # same content, different timestamp -> no-op
    assert not store.append(_rec("skill:qc", [QC], at="2026-10-08T00:00:00+00:00"))
    assert len(store.read_all()) == 1


def test_append_new_content_adds_line_and_latest_wins(tmp_path) -> None:
    store = KeywordJournalStore(tmp_path)
    store.append(_rec("skill:qc", [QC], ["old"]))
    assert store.append(_rec("skill:qc", [QC], ["new"]))
    assert len(store.read_all()) == 2  # lossless: old line kept
    assert store.latest()["skill:qc"]["keywords"] == ["new"]
    # reverting to an earlier content is a change vs. the LATEST line
    assert store.append(_rec("skill:qc", [QC], ["old"]))
    assert len(store.read_all()) == 3


def test_read_all_skips_torn_and_invalid_lines(tmp_path) -> None:
    store = KeywordJournalStore(tmp_path)
    store.append(_rec("skill:a", [PAF]))
    with (tmp_path / JOURNAL_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write('{"source_ref": "skill:torn", "keywo')  # torn, no newline
    assert [r["source_ref"] for r in store.read_all()] == ["skill:a"]
    # a later append must not glue onto the torn line
    store.append(_rec("skill:b", [QC]))
    assert [r["source_ref"] for r in store.read_all()] == ["skill:a", "skill:b"]


def test_read_all_skips_records_without_source_ref(tmp_path) -> None:
    (tmp_path / JOURNAL_FILENAME).write_text(
        json.dumps({"keywords": ["x"]})
        + "\n[1,2]\n"
        + json.dumps(_rec("skill:a", []))
        + "\n"
    )
    assert [r["source_ref"] for r in KeywordJournalStore(tmp_path).read_all()] == [
        "skill:a"
    ]


def test_keyed_refs(tmp_path) -> None:
    store = KeywordJournalStore(tmp_path)
    store.append(_rec("skill:a", [PAF]))
    store.append(_rec("note:b.md", [QC], source_type="note"))
    assert store.keyed_refs() == {"skill:a", "note:b.md"}


def test_concurrent_appends_lose_nothing(tmp_path) -> None:
    store = KeywordJournalStore(tmp_path)

    def work(i: int) -> None:
        store.append(_rec(f"skill:s{i}", [PAF], [f"k{i}"]))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.keyed_refs() == {f"skill:s{i}" for i in range(12)}


# --- derive_entity_map -------------------------------------------------------


def test_derive_merges_entities_across_sources() -> None:
    recs = [
        _rec("skill:qc", [QC]),
        _rec(
            "note:qc.md",
            [{"canonical": "QuantConnect", "aliases": ["lean", "QC"]}],
            source_type="note",
        ),
    ]
    m = derive_entity_map(recs)
    # case-variants merge; the spelling from the first source ref in sorted order
    # ("note:..." < "skill:...") is the one shown
    assert list(m) == ["QuantConnect"]
    ent = m["QuantConnect"]
    assert ent["sources"] == ["note:qc.md", "skill:qc"]
    assert ent["aliases"] == ["lean", "QC", "quantconnect-api"]  # same first-seen rule
    assert ent["canonical"] == "QuantConnect"


def test_derive_latest_line_wins_per_source() -> None:
    recs = [
        _rec("skill:a", [{"canonical": "old-name", "aliases": ["o"]}]),
        _rec("skill:a", [{"canonical": "new-name", "aliases": ["n"]}]),
    ]
    m = derive_entity_map(recs)
    assert list(m) == ["new-name"]


def test_derive_drops_non_live_refs() -> None:
    recs = [_rec("skill:a", [PAF]), _rec("skill:gone", [QC])]
    m = derive_entity_map(recs, live_refs={"skill:a"})
    assert list(m) == ["paf"]
    # without a liveness set nothing is dropped
    assert set(derive_entity_map(recs)) == {"paf", "quantconnect"}


def test_derive_entity_with_no_live_source_disappears() -> None:
    recs = [_rec("skill:gone", [QC])]
    assert derive_entity_map(recs, live_refs=set()) == {}


def test_derive_is_order_independent() -> None:
    a = _rec("skill:a", [{"canonical": "Paf", "aliases": ["x"]}])
    b = _rec("skill:b", [{"canonical": "paf", "aliases": ["y"]}])
    assert derive_entity_map([a, b]) == derive_entity_map([b, a])


def test_derive_alias_never_repeats_canonical() -> None:
    recs = [
        _rec("skill:a", [PAF]),
        _rec("skill:b", [{"canonical": "paf", "aliases": []}]),
    ]
    m = derive_entity_map(recs)
    assert "paf" not in [normalize_term(a) for a in m["paf"]["aliases"]]


# --- entity-map file ---------------------------------------------------------


def test_entity_map_roundtrip(tmp_path) -> None:
    m = derive_entity_map([_rec("skill:a", [PAF])])
    write_entity_map(tmp_path, m)
    assert (tmp_path / ENTITY_MAP_FILENAME).exists()
    assert read_entity_map(tmp_path) == m


def test_read_entity_map_absent_or_corrupt(tmp_path) -> None:
    assert read_entity_map(tmp_path) == {}
    (tmp_path / ENTITY_MAP_FILENAME).write_text("{not json")
    assert read_entity_map(tmp_path) == {}
    (tmp_path / ENTITY_MAP_FILENAME).write_text("[1, 2]")
    assert read_entity_map(tmp_path) == {}


# --- watermark ---------------------------------------------------------------


def test_watermark_fresh_when_absent_or_corrupt(tmp_path) -> None:
    assert KeywordWatermark.load(tmp_path).files == {}
    (tmp_path / WATERMARK_FILENAME).write_text("{oops")
    assert KeywordWatermark.load(tmp_path).files == {}
    (tmp_path / WATERMARK_FILENAME).write_text("[]")
    assert KeywordWatermark.load(tmp_path).files == {}


def test_watermark_roundtrip(tmp_path) -> None:
    wm = KeywordWatermark()
    wm.advance({"skill:a": (1727083000.5, 4821), "note:b.md": (1.0, 9)})
    wm.save(tmp_path)
    loaded = KeywordWatermark.load(tmp_path)
    assert loaded.files == {"skill:a": (1727083000.5, 4821), "note:b.md": (1.0, 9)}
    assert loaded.updated_at
    raw = json.loads((tmp_path / WATERMARK_FILENAME).read_text())
    assert raw["files"]["skill:a"] == {"mtime": 1727083000.5, "size": 4821}


def test_watermark_load_drops_malformed_entries(tmp_path) -> None:
    (tmp_path / WATERMARK_FILENAME).write_text(
        json.dumps(
            {
                "files": {
                    "ok": {"mtime": 2, "size": 3},
                    "bad-type": {"mtime": "x", "size": 3},
                    "bool": {"mtime": True, "size": 3},
                    "missing": {"mtime": 1.0},
                    "not-dict": 7,
                }
            }
        )
    )
    assert KeywordWatermark.load(tmp_path).files == {"ok": (2.0, 3)}


def test_changed_flags_new_and_modified_only() -> None:
    wm = KeywordWatermark()
    wm.advance({"skill:a": (1.0, 10), "skill:b": (2.0, 20)})
    current = {
        "skill:a": (1.0, 10),  # unchanged
        "skill:b": (3.0, 20),  # mtime moved
        "skill:c": (1.0, 5),  # new
    }
    assert wm.changed(current) == ["skill:b", "skill:c"]
    # same mtime but different size also counts
    assert wm.changed({"skill:a": (1.0, 11)}) == ["skill:a"]


def test_vanished_and_advance_remove() -> None:
    wm = KeywordWatermark()
    wm.advance({"skill:a": (1.0, 1), "skill:b": (1.0, 1)})
    current = {"skill:a": (1.0, 1)}
    assert wm.vanished(current) == ["skill:b"]
    wm.advance({}, remove=["skill:b", "skill:never-tracked"])
    assert wm.files == {"skill:a": (1.0, 1)}


def test_failed_pass_does_not_advance(tmp_path) -> None:
    """Advancing is the caller's last step; without it the source stays changed."""
    wm = KeywordWatermark()
    current = {"skill:a": (1.0, 1)}
    assert wm.changed(current) == ["skill:a"]
    # (pass fails here: no advance, no save)
    assert KeywordWatermark.load(tmp_path).changed(current) == ["skill:a"]
