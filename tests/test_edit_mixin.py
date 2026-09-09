"""Tests for the shared text-edit primitive ``paf.capabilities._edit``.

Covers the contract that vault/skills/fs wrappers all delegate to:
- batch (N edits, one write)
- single edit (list of one)
- validate-all-then-write (a stale/ambiguous pair rejects the WHOLE batch)
- CAS-at-commit (expected_version mismatch rejects)
- never-lie (a write that doesn't persist raises ModelRetry)
- per-pair diff (returned, capped)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelRetry

from paf.capabilities._edit import EditResult, apply_edits
from paf.capabilities._storage._atomic import write_atomic


def _seams(p: Path):
    def read(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def write(path: Path, content: str) -> None:
        write_atomic(path, content)

    return read, write


def test_single_edit(tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    p.write_text("one-two-three", encoding="utf-8")
    read, write = _seams(p)

    result = apply_edits(p, [{"old_text": "two", "new_text": "2"}], expected_version=None, read=read, write=write)

    assert isinstance(result, EditResult)
    assert p.read_text(encoding="utf-8") == "one-2-three"
    assert len(result.diffs) == 1
    assert "- two" in result.diffs[0]
    assert "+ 2" in result.diffs[0]


def test_batch_applies_all_in_one_write(tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    p.write_text("a b c d", encoding="utf-8")
    read, write = _seams(p)
    calls = {"writes": 0}

    def counting_write(path: Path, content: str) -> None:
        calls["writes"] += 1
        write(path, content)

    edits = [
        {"old_text": "a", "new_text": "A"},
        {"old_text": "b", "new_text": "B"},
        {"old_text": "c", "new_text": "C"},
        {"old_text": "d", "new_text": "D"},
    ]
    result = apply_edits(p, edits, expected_version=None, read=read, write=counting_write)

    assert p.read_text(encoding="utf-8") == "A B C D"
    assert calls["writes"] == 1  # exactly one write for the whole batch
    assert len(result.diffs) == 4


def test_validate_all_rejects_whole_batch_on_one_missing(tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    p.write_text("only this stays", encoding="utf-8")
    read, write = _seams(p)

    edits = [
        {"old_text": "this", "new_text": "THAT"},
        {"old_text": "does-not-exist", "new_text": "x"},
    ]
    with pytest.raises(ModelRetry, match="Refusing to edit"):
        apply_edits(p, edits, expected_version=None, read=read, write=write)

    # No partial write: the valid edit must NOT have been applied.
    assert "THAT" not in p.read_text(encoding="utf-8")
    assert p.read_text(encoding="utf-8") == "only this stays"


def test_validate_all_rejects_on_ambiguity(tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    p.write_text("dup dup", encoding="utf-8")
    read, write = _seams(p)

    with pytest.raises(ModelRetry, match="found 2 times"):
        apply_edits(p, [{"old_text": "dup", "new_text": "x"}], expected_version=None, read=read, write=write)


def test_cas_mismatch_rejects(tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    p.write_text("version one", encoding="utf-8")
    read, write = _seams(p)

    with pytest.raises(ModelRetry, match="has changed"):
        apply_edits(
            p,
            [{"old_text": "one", "new_text": "two"}],
            expected_version="deadbeef",  # wrong token
            read=read,
            write=write,
        )
    assert p.read_text(encoding="utf-8") == "version one"


def test_cas_pass_allows(tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    p.write_text("version one", encoding="utf-8")
    read, write = _seams(p)
    ver = __import__("paf.capabilities._edit", fromlist=["_content_hash"])._content_hash("version one")

    result = apply_edits(p, [{"old_text": "one", "new_text": "two"}], expected_version=ver, read=read, write=write)
    assert "two" in p.read_text(encoding="utf-8")
    assert result.version != ver


def test_never_lie_on_unpersisted_write(tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    p.write_text("original", encoding="utf-8")

    def read(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def lying_write(path: Path, content: str) -> None:
        # Pretend to write but don't persist — the tool must not claim success.
        return None

    with pytest.raises(ModelRetry, match="did not persist"):
        apply_edits(p, [{"old_text": "original", "new_text": "CHANGED"}], expected_version=None, read=read, write=lying_write)


def test_empty_edits_rejected(tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    p.write_text("x", encoding="utf-8")
    read, write = _seams(p)
    with pytest.raises(ModelRetry, match="non-empty list"):
        apply_edits(p, [], expected_version=None, read=read, write=write)


def test_missing_file_rejected(tmp_path: Path) -> None:
    p = tmp_path / "nope.md"
    read, write = _seams(p)
    with pytest.raises(ModelRetry, match="File not found"):
        apply_edits(p, [{"old_text": "a", "new_text": "b"}], expected_version=None, read=read, write=write)


def test_diff_capped(tmp_path: Path) -> None:
    p = tmp_path / "note.md"
    # A unique marker surrounded by lots of text, so the pair matches exactly once.
    body = "A" * 5000 + "UNIQUE_MARKER_HERE" + "B" * 5000
    p.write_text(body, encoding="utf-8")
    read, write = _seams(p)

    result = apply_edits(
        p,
        [{"old_text": "UNIQUE_MARKER_HERE", "new_text": "REPLACED"}],
        expected_version=None,
        read=read,
        write=write,
    )
    assert "REPLACED" in p.read_text(encoding="utf-8")
    # The old/new pair is small here; the point is the diff path works.
    assert len(result.diffs) == 1
    assert "- UNIQUE_MARKER_HERE" in result.diffs[0]
