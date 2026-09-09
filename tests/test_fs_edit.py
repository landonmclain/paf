"""Tests for the filesystem ``edit_file`` tool wired over the shared ``_edit``.

``edit_file`` now takes an ``edits=[{old_text, new_text}, ...]`` list (one atomic
write), delegates to the shared ``_edit.apply_edits`` primitive, and never
reports success for an edit that did not persist (raises a recoverable error).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from paf.capabilities.fs import FileSystemToolset


def _fs(tmp_path: Path) -> FileSystemToolset:
    return FileSystemToolset(
        root_dir=tmp_path,
        allowed_patterns=[],
        denied_patterns=[],
        protected_patterns=[".git/**", ".env"],
        max_read_lines=2000,
        max_search_results=1000,
        max_find_results=1000,
    )


def _run(coro):
    return asyncio.run(coro)


def test_edit_file_single_and_batch(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    fs = _fs(tmp_path)

    # single edit (list of one)
    out = _run(fs.edit_file("app.py", [{"old_text": "x = 1", "new_text": "x = 10"}]))
    assert "Edited app.py" in out
    assert "x = 10" in (tmp_path / "app.py").read_text(encoding="utf-8")

    # batch: two independent changes in one call
    out2 = _run(
        fs.edit_file(
            "app.py",
            [
                {"old_text": "x = 10", "new_text": "x = 100"},
                {"old_text": "y = 2", "new_text": "y = 20"},
            ],
        )
    )
    assert "Edited app.py" in out2
    body = (tmp_path / "app.py").read_text(encoding="utf-8")
    assert "x = 100" in body
    assert "y = 20" in body


def test_edit_file_never_lies_on_bad_batch(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    fs = _fs(tmp_path)

    # one valid + one garbage pair -> whole batch rejected, nothing lands
    with pytest.raises(Exception, match="Refusing to edit"):
        _run(
            fs.edit_file(
                "app.py",
                [
                    {"old_text": "x = 1", "new_text": "x = 10"},
                    {"old_text": "does not exist", "new_text": "boom"},
                ],
            )
        )
    body = (tmp_path / "app.py").read_text(encoding="utf-8")
    assert "x = 10" not in body
    assert "x = 1" in body


def test_edit_file_cas_conflict(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    fs = _fs(tmp_path)

    with pytest.raises(Exception, match="Conflict"):
        _run(
            fs.edit_file(
                "app.py", [{"old_text": "x = 1", "new_text": "x = 10"}], expected_hash="stale-hash"
            )
        )
    assert "x = 10" not in (tmp_path / "app.py").read_text(encoding="utf-8")


def test_edit_file_ambiguous_refuses(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("flag = True\nflag = False\n", encoding="utf-8")
    fs = _fs(tmp_path)
    with pytest.raises(Exception, match="found 2 times"):
        _run(fs.edit_file("app.py", [{"old_text": "flag", "new_text": "FLAG"}]))
