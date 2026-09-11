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
        max_search_file_bytes=200 * 1024,
        search_excerpt_chars=200,
        max_search_total_bytes=256 * 1024,
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


def test_write_file_uses_shared_atomic_no_debris(tmp_path: Path) -> None:
    fs = _fs(tmp_path)
    out = _run(fs.write_file("new.txt", "hello world\n"))
    assert "Wrote 12 chars" in out
    body = (tmp_path / "new.txt").read_text(encoding="utf-8")
    assert body == "hello world\n"
    # write_file goes through the shared mkstemp atomic: no temp/staging debris
    debris = [p.name for p in tmp_path.glob("*.tmp")] + [p.name for p in tmp_path.glob(".*.tmp")]
    assert debris == [], f"leftover temp files: {debris}"
    # overwrite also lands atomically + hash bumps

    out2 = _run(fs.write_file("new.txt", "second\n"))
    assert "[hash:" in out2
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "second\n"


def test_search_files_excerpts_long_lines(tmp_path: Path) -> None:
    """A single huge JSON line must be excerpted, not returned whole (PAF-UI fix)."""
    big = '{"TARGET":1,"' + "x" * 300000 + ',"TARGET":2,"' + "y" * 300000 + "}"
    (tmp_path / "big.json").write_text(big, encoding="utf-8")
    fs = _fs(tmp_path)
    out = _run(fs.search_files("TARGET", path="."))
    for line in out.splitlines():
        assert len(line) < 250, f"matching line not excerpted: {len(line)} chars"


def test_search_files_skips_oversized_files(tmp_path: Path) -> None:
    """Files over the size threshold are skipped, not read in full."""
    (tmp_path / "huge.dat").write_text("TARGET " * 100000, encoding="utf-8")  # ~700KB
    (tmp_path / "small.txt").write_text("TARGET small\n", encoding="utf-8")
    fs = _fs(tmp_path)
    out = _run(fs.search_files("TARGET", path="."))
    assert "huge.dat" not in out
    assert "small.txt" in out


def test_search_files_byte_cap(tmp_path: Path) -> None:
    """Total result string is byte-capped even with many small matches."""
    for i in range(40):
        lines = ["HIT " + "z" * 100 + f" file{i} line{j}" for j in range(200)]
        (tmp_path / f"f{i}.txt").write_text("\n".join(lines), encoding="utf-8")
    fs = FileSystemToolset(
        root_dir=tmp_path,
        allowed_patterns=[],
        denied_patterns=[],
        protected_patterns=[".git/**", ".env"],
        max_read_lines=2000,
        max_search_results=1_000_000,
        max_find_results=1000,
        max_search_file_bytes=200 * 1024,
        search_excerpt_chars=10_000,
        max_search_total_bytes=256 * 1024,
    )
    out = _run(fs.search_files("HIT", path="."))
    # Capped at ~256KB + one boundary entry (262,144 cap), versus ~880KB if the
    # 8,000 matching lines were all returned. The truncation marker must fire.
    assert "bytes]" in out
    assert len(out.encode("utf-8")) < 300 * 1024  # far under the uncapped result


