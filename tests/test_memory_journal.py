"""Deterministic tests for the append-only memory journal (`_journal.py`).

The journal is the **dump** layer: write-once, append-only, no dedup, no
look-before-write. These tests pin that contract and the atomicity/identity
guarantees — no model, no network, no `agent-foundation` dependency.

What is covered:

1. **Record shape + validation** — `MemoryRecord` enforces the locked
   `{type, fact, born_on, source_ref}` schema, the locked `type` enum, and
   non-empty `fact`/`source_ref`.
2. **id = JSONL line number** — the 1-based line number assigned by append
   order, stable across reads.
3. **Append is a dump** — a duplicate fact legitimately appends a second record
   (newest-wins is resolved at render, never here); there is no `has`/dedup.
4. **Atomic block append** — `append_many` writes a chunk's records all-or-
   nothing; every line parses, no torn/corrupt lines.
5. **born_on normalization** — naive datetimes become UTC ISO; lexically sortable.
6. **Empty store** — `read_all`/`count` on a not-yet-existing journal return
   empty/0 and first append is line 1.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from paf.capabilities.memory import (
    JournalStore,
    MemoryRecord,
    render_journal_memory,
    write_render,
)

# ---------------------------------------------------------------------------
# Record shape + validation
# ---------------------------------------------------------------------------


def test_record_roundtrips_to_row_and_back() -> None:
    rec = MemoryRecord(
        type="preference",
        fact="Landon prefers flat-file, diffable storage",
        born_on="2026-08-17T00:00:00Z",
        source_ref="seed-lmem-3f9a:12",
    )
    assert rec.to_row() == {
        "type": "preference",
        "fact": "Landon prefers flat-file, diffable storage",
        "born_on": "2026-08-17T00:00:00+00:00",
        "source_ref": "seed-lmem-3f9a:12",
    }
    assert MemoryRecord.from_row(rec.to_row()) == rec


@pytest.mark.parametrize(
    "bad_type",
    ["section", "state", "ongoing", "note", "", "PREFERENCE"],
)
def test_record_rejects_invalid_type(bad_type: str) -> None:
    with pytest.raises(ValueError, match="invalid memory type"):
        MemoryRecord(
            type=bad_type,
            fact="anything",
            born_on="2026-08-17T00:00:00Z",
            source_ref="run:0",
        )


@pytest.mark.parametrize("which", ["fact", "source_ref"])
def test_record_rejects_blank_fields(which: str) -> None:
    kwargs = {
        "type": "fact",
        "fact": "a fact",
        "born_on": "2026-08-17T00:00:00Z",
        "source_ref": "run:0",
    }
    kwargs[which] = "   "
    with pytest.raises(ValueError, match="non-empty"):
        MemoryRecord(**kwargs)


def test_all_locked_types_accepted() -> None:
    for t in ("identity", "preference", "fact", "event", "task", "current"):
        # Should not raise.
        MemoryRecord(type=t, fact="x", born_on="2026-08-17T00:00:00Z", source_ref="run:0")


# ---------------------------------------------------------------------------
# born_on normalization
# ---------------------------------------------------------------------------


def test_naive_datetime_normalized_to_utc() -> None:
    rec = MemoryRecord(
        type="fact",
        fact="x",
        born_on=datetime(2026, 8, 17, 10, 30),
        source_ref="run:1",
    )
    assert rec.born_on == "2026-08-17T10:30:00+00:00"


def test_aware_datetime_kept() -> None:
    rec = MemoryRecord(
        type="fact",
        fact="x",
        born_on=datetime(2026, 8, 17, 10, 30, tzinfo=UTC),
        source_ref="run:1",
    )
    assert rec.born_on == "2026-08-17T10:30:00+00:00"


def test_iso_string_z_normalized() -> None:
    rec = MemoryRecord(type="fact", fact="x", born_on="2026-08-17T10:30:00Z", source_ref="run:1")
    assert rec.born_on == "2026-08-17T10:30:00+00:00"


def test_born_on_sorts_lexically() -> None:
    """ISO (UTC) born_on stringifies so lexical sort == chronological sort."""
    earlier = MemoryRecord(
        type="fact", fact="early", born_on="2026-08-16T00:00:00Z", source_ref="run:0"
    ).born_on
    later = MemoryRecord(
        type="fact", fact="late", born_on="2026-08-17T00:00:00Z", source_ref="run:1"
    ).born_on
    assert earlier < later


def test_empty_born_on_is_allowed_and_stored_empty(tmp_path) -> None:
    """A fact with no derivable date is recorded with empty born_on — never a
    fabricated one, and never rejected/dropped (the fabrication-bug fix)."""
    rec = MemoryRecord(type="fact", fact="no date", born_on="", source_ref="run:0")
    assert rec.born_on == ""
    rec_none = MemoryRecord(type="fact", fact="no date", source_ref="run:1")
    assert rec_none.born_on == ""
    # round-trips through the store without raising (previously ValueError)
    store = JournalStore(tmp_path)
    store.append(type="fact", fact="no date", born_on="", source_ref="run:0")
    _, stored = store.read_all()[0]
    assert stored.born_on == ""


# ---------------------------------------------------------------------------
# id = JSONL line number, append order
# ---------------------------------------------------------------------------


def test_ids_are_sequential_line_numbers(tmp_path) -> None:
    store = JournalStore(tmp_path)
    r1 = store.append(type="fact", fact="first", born_on="2026-08-17T00:00:00Z", source_ref="run:0")
    r2 = store.append(type="fact", fact="second", born_on="2026-08-17T00:01:00Z", source_ref="run:1")

    rows = store.read_all()
    assert [i for i, _ in rows] == [1, 2]
    assert [r.fact for _, r in rows] == ["first", "second"]
    assert r1 is not None and r2 is not None


def test_empty_store_is_empty_and_first_append_is_line_1(tmp_path) -> None:
    store = JournalStore(tmp_path)
    assert store.read_all() == []
    assert store.count() == 0

    store.append(type="fact", fact="hello", born_on="2026-08-17T00:00:00Z", source_ref="run:0")
    assert store.count() == 1
    assert store.read_all() == [(1, MemoryRecord(
        type="fact", fact="hello", born_on="2026-08-17T00:00:00Z", source_ref="run:0"
    ))]


def test_journal_file_lives_at_memory_dir(tmp_path) -> None:
    store = JournalStore(tmp_path)
    assert store.filename == "memory-journal.jsonl"
    assert store.path == str(tmp_path / "memory-journal.jsonl")
    store.append(type="fact", fact="x", born_on="2026-08-17T00:00:00Z", source_ref="run:0")
    assert (tmp_path / "memory-journal.jsonl").is_file()


# ---------------------------------------------------------------------------
# Append is a dump: no dedup / look-before-write
# ---------------------------------------------------------------------------


def test_duplicate_fact_legitimately_appends_second_record(tmp_path) -> None:
    """The journal is a dump: a fact seen again (even identical) appends.

    Newest-wins is resolved at RENDER, never here. There must be no `has` gate.
    """
    store = JournalStore(tmp_path)
    store.append(type="preference", fact="Prefers Terminal.app", born_on="2026-08-16T00:00:00Z", source_ref="run:0")
    store.append(type="preference", fact="Prefers Terminal.app", born_on="2026-08-17T00:00:00Z", source_ref="run:5")

    rows = store.read_all()
    assert len(rows) == 2
    # Two identical facts, different born_on -- exactly the supersession input.
    assert rows[0][1].fact == rows[1][1].fact == "Prefers Terminal.app"
    assert rows[1][1].born_on > rows[0][1].born_on


def test_append_is_never_a_rewrite(tmp_path) -> None:
    """Appending never rewrites/compacts prior records -- ids stay stable."""
    store = JournalStore(tmp_path)
    store.append(type="fact", fact="a", born_on="2026-08-17T00:00:00Z", source_ref="run:0")
    ids_after_one = [i for i, _ in store.read_all()]
    store.append(type="fact", fact="b", born_on="2026-08-17T00:01:00Z", source_ref="run:1")
    ids_after_two = [i for i, _ in store.read_all()]
    assert ids_after_one == [1]
    assert ids_after_two == [1, 2]


# ---------------------------------------------------------------------------
# Atomic block append
# ---------------------------------------------------------------------------


def test_append_many_writes_one_block(tmp_path) -> None:
    store = JournalStore(tmp_path)
    block = [
        MemoryRecord(type="fact", fact="one", born_on="2026-08-17T00:00:00Z", source_ref="run:0"),
        MemoryRecord(type="preference", fact="two", born_on="2026-08-17T00:01:00Z", source_ref="run:0"),
        MemoryRecord(type="task", fact="three", born_on="2026-08-17T00:02:00Z", source_ref="run:1"),
    ]
    store.append_many(block)
    rows = store.read_all()
    assert [i for i, _ in rows] == [1, 2, 3]
    assert [r.fact for _, r in rows] == ["one", "two", "three"]


def test_append_many_empty_is_noop(tmp_path) -> None:
    store = JournalStore(tmp_path)
    assert store.append_many([]) == []
    assert store.count() == 0
    assert not (tmp_path / "memory-journal.jsonl").exists()


def test_every_written_line_parses(tmp_path) -> None:
    """No torn/partial lines even with multi-line-ish facts (newlines stripped)."""
    store = JournalStore(tmp_path)
    for i in range(50):
        store.append(
            type="fact",
            fact=f"durable fact number {i} with some detail",
            born_on=f"2026-08-17T00:{i:02d}:00Z",
            source_ref=f"run:{i}",
        )
    raw = (tmp_path / "memory-journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw) == 50
    rows = store.read_all()
    assert len(rows) == 50
    assert all(isinstance(line, str) and line for line in raw)


def test_interleaved_append_and_append_many(tmp_path) -> None:
    store = JournalStore(tmp_path)
    store.append(type="fact", fact="solo-before", born_on="2026-08-17T00:00:00Z", source_ref="run:0")
    store.append_many([
        MemoryRecord(type="fact", fact="block-1", born_on="2026-08-17T00:01:00Z", source_ref="run:0"),
        MemoryRecord(type="fact", fact="block-2", born_on="2026-08-17T00:02:00Z", source_ref="run:0"),
    ])
    store.append(type="fact", fact="solo-after", born_on="2026-08-17T00:03:00Z", source_ref="run:1")
    ids = [i for i, _ in store.read_all()]
    assert ids == [1, 2, 3, 4]


def test_source_ref_and_type_preserved_through_append(tmp_path) -> None:
    store = JournalStore(tmp_path)
    store.append(
        type="event",
        fact="Deployed v2 on 2026-08-15",
        born_on="2026-08-15T12:00:00Z",
        source_ref="seed-lmem-a1b2c3d4:7",
    )
    _, rec = store.read_all()[0]
    assert rec.type == "event"
    assert rec.source_ref == "seed-lmem-a1b2c3d4:7"


# ---------------------------------------------------------------------------
# Deterministic render: journal -> MEMORY.md (the "database" step)
# ---------------------------------------------------------------------------


def _rec(type, fact, born_on, source_ref="run:0"):
    return MemoryRecord(type=type, fact=fact, born_on=born_on, source_ref=source_ref)


def test_render_groups_by_type_and_sorts_by_born(tmp_path) -> None:
    store = JournalStore(tmp_path)
    store.append_many([
        _rec("preference", "later pref", "2026-08-17T00:00:00Z"),
        _rec("preference", "earlier pref", "2026-08-16T00:00:00Z"),
        _rec("preference", "middle pref", "2026-08-16T12:00:00Z"),
        _rec("fact", "a plain fact", "2026-08-15T00:00:00Z"),
        _rec("task", "a pending task", "2026-08-17T00:00:00Z"),
    ])
    md = render_journal_memory(store)

    # Sections, sorted by type name: fact, preference, task (identity/etc absent).
    assert md.index("## fact") < md.index("## preference") < md.index("## task")
    # Within each section, birth-ascending.
    pref_block = md.split("## preference", 1)[1].split("## ", 1)[0]
    assert pref_block.index("earlier pref") < pref_block.index("middle pref") < pref_block.index("later pref")


def test_render_each_line_carries_fact_date_source(tmp_path) -> None:
    store = JournalStore(tmp_path)
    store.append(
        type="preference",
        fact="user prefers flat-file, diffable storage",
        born_on="2026-08-17T10:30:00Z",
        source_ref="seed-lmem-a1b2c3d4:12",
    )
    md = render_journal_memory(store)
    line = next(ln for ln in md.splitlines() if ln.startswith("- "))
    # fact, date (born_on), source all present on the one line.
    assert "user prefers flat-file, diffable storage" in line
    assert "2026-08-17T10:30:00+00:00" in line
    assert "seed-lmem-a1b2c3d4:12" in line


def test_render_omits_absent_types(tmp_path) -> None:
    store = JournalStore(tmp_path)
    store.append(type="fact", fact="only a fact", born_on="2026-08-17T00:00:00Z", source_ref="run:0")
    md = render_journal_memory(store)
    assert "## fact" in md
    for absent in ("## identity", "## preference", "## event", "## task", "## current"):
        assert absent not in md


def test_render_is_deterministic(tmp_path) -> None:
    store = JournalStore(tmp_path)
    store.append_many([
        _rec("preference", "a", "2026-08-17T00:00:00Z"),
        _rec("fact", "b", "2026-08-15T00:00:00Z"),
        _rec("task", "c", "2026-08-16T00:00:00Z"),
        _rec("event", "d", "2026-08-14T00:00:00Z"),
    ])
    assert render_journal_memory(store) == render_journal_memory(store)


def test_render_empty_journal_is_empty_string(tmp_path) -> None:
    store = JournalStore(tmp_path)
    assert render_journal_memory(store) == ""


def test_render_from_explicit_record_list(tmp_path) -> None:
    """render_journal_memory also accepts a record list (no disk needed)."""
    records = [
        _rec("current", "currently rebuilding staging.com", "2026-08-17T00:00:00Z"),
        _rec("current", "newer current state", "2026-08-17T01:00:00Z"),
    ]
    md = render_journal_memory(records)
    assert "## current" in md
    # Within a type, birth-ascending: older record renders first.
    assert md.index("currently rebuilding") < md.index("newer current state")


def test_write_render_writes_atomic_md(tmp_path) -> None:
    store = JournalStore(tmp_path)
    store.append(type="fact", fact="written fact", born_on="2026-08-17T00:00:00Z", source_ref="run:0")
    md_path = tmp_path / "MEMORY.md"
    rendered = write_render(store, str(md_path))
    assert md_path.is_file()
    assert md_path.read_text(encoding="utf-8") == rendered
    assert "written fact" in rendered


def test_render_multi_line_fact_stays_one_line(tmp_path) -> None:
    store = JournalStore(tmp_path)
    store.append(
        type="fact",
        fact="line one\nline two\nline three",
        born_on="2026-08-17T00:00:00Z",
        source_ref="run:0",
    )
    md = render_journal_memory(store)
    # The fact's internal newlines are collapsed so the one-per-line shape holds.
    lines = [ln for ln in md.splitlines() if ln.startswith("- ")]
    assert len(lines) == 1
    assert "line one line two line three" in lines[0]
