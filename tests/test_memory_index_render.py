"""Tests for the curated index renderer (``render_index`` / ``write_index_render``).

The index renderer is the curated counterpart to ``render_journal_memory``:
it presents the deduped/superseded entries from ``memory-index.json`` instead of
the raw, ever-growing journal. Presentation is identical in shape — grouped by
the 6 semantic ``type``s, ``## <type>`` sections, one line per entry — but the
recency axis is ``updated_on`` (bump-on-change), not ``born_on``.
"""

from __future__ import annotations

from paf.capabilities.memory import (
    IndexEntry,
    IndexStore,
    render_index,
    write_index_render,
)


def _entry(
    *,
    id: str,
    type: str,
    fact: str,
    updated_on: str,
    source_ref: str = "run-1:0",
    subject: str = "topic",
    born_on: str = "",
) -> IndexEntry:
    return IndexEntry(
        id=id,
        type=type,
        subject=subject,
        fact=fact,
        born_on=born_on,
        updated_on=updated_on,
        source_ref=source_ref,
        source=[],
    )


def test_index_render_groups_by_type_sorted_by_updated_on() -> None:
    entries = [
        # Mixed order on purpose — grouping + sort must normalize it.
        _entry(id="mem-3", type="preference", fact="later pref", updated_on="2026-08-19"),
        _entry(id="mem-1", type="preference", fact="earlier pref", updated_on="2026-08-17"),
        _entry(id="mem-2", type="preference", fact="middle pref", updated_on="2026-08-18"),
        _entry(id="mem-4", type="fact", fact="a plain fact", updated_on="2026-08-15"),
        _entry(id="mem-5", type="task", fact="a pending task", updated_on="2026-08-20"),
    ]
    md = render_index(entries)

    # Sections in sorted type order: fact, preference, task.
    assert md.index("## fact") < md.index("## preference") < md.index("## task")

    # Within a type, updated_on (recency) ascending — NOT born_on.
    pref_block = md.split("## preference", 1)[1].split("## ", 1)[0]
    assert (
        pref_block.index("earlier pref")
        < pref_block.index("middle pref")
        < pref_block.index("later pref")
    )


def test_index_render_line_present_fact_updated_on_source_ref() -> None:
    md = render_index(
        [
            _entry(
                id="mem-7",
                type="preference",
                fact="user prefers flat-file, diffable storage",
                updated_on="2026-08-18",
                source_ref="run-2:3",
            )
        ]
    )
    line = next(ln for ln in md.splitlines() if ln.startswith("- "))
    assert "user prefers flat-file, diffable storage" in line
    assert "2026-08-18" in line
    assert "run-2:3" in line
    # born_on is NOT shown on the line (stays in the index for audit).
    assert "2026-08-17" not in line


def test_index_render_empty_updated_on_omits_date_segment() -> None:
    md = render_index(
        [
            _entry(
                id="mem-9",
                type="fact",
                fact="undated fact",
                updated_on="",
                source_ref="run-5:1",
            )
        ]
    )
    # No dangling separator; date segment dropped but fact + source kept.
    assert "- undated fact — run-5:1" in md
    assert " — — " not in md


def test_index_render_empty_index() -> None:
    assert render_index([]) == ""


def test_index_render_from_store_on_disk(tmp_path) -> None:
    store = IndexStore(str(tmp_path))
    store._save(
        [
            _entry(
                id="mem-1",
                type="current",
                fact="current git branch is feature/m2-prod-sync",
                updated_on="2026-08-21",
            ),
            _entry(
                id="mem-2",
                type="fact",
                fact="a distilled fact",
                updated_on="2026-08-20",
            ),
        ],
        watermark=7,
        next_id=3,
    )
    md = render_index(store)
    assert "## current" in md
    assert "current git branch is feature/m2-prod-sync" in md
    assert "## fact" in md
    assert "a distilled fact" in md


def test_index_render_dedupes_vs_journal() -> None:
    """The whole point: 2 curated entries render 2 lines, even though the
    underlying journal may hold many records saying the same thing across runs.
    (Contrast: the journal renderer emits one line per journal record, so a
    fact restated 5x shows up 5x.)"""
    entries = [
        _entry(id="mem-1", type="fact", fact="branch is feature/m2-prod-sync", updated_on="2026-08-21"),
        _entry(id="mem-2", type="fact", fact="demo spec uses deepseek-v4", updated_on="2026-08-20"),
    ]
    md = render_index(entries)
    lines = [ln for ln in md.splitlines() if ln.startswith("- ")]
    assert len(lines) == 2


def test_write_index_render_atomic(tmp_path) -> None:
    md_path = str(tmp_path / "MEMORY.md")
    out = write_index_render(
        [_entry(id="mem-1", type="fact", fact="hello world", updated_on="2026-08-21")],
        md_path,
    )
    assert out == render_index(
        [_entry(id="mem-1", type="fact", fact="hello world", updated_on="2026-08-21")]
    )
    with open(md_path) as fh:
        on_disk = fh.read()
    assert on_disk == out
    assert "hello world" in on_disk


def test_index_render_deterministic() -> None:
    entries = [
        _entry(id="mem-2", type="fact", fact="b", updated_on="2026-08-20"),
        _entry(id="mem-1", type="fact", fact="a", updated_on="2026-08-21"),
        _entry(id="mem-3", type="preference", fact="pref", updated_on="2026-08-19"),
    ]
    assert render_index(entries) == render_index(list(reversed(entries)))
