"""Bounded-walk behavior for the fs `search_files` / `find_files` / `list_directory` tools.

These lock in the fix for the PAF-UI stall: a recursive content/glob search over
a huge tree (paf's `data/` is ~1.8M files / ~25GB) used to hang for minutes
because it eagerly materialized and sorted *every* path (`sorted(rglob('*'))`)
before any per-file guard ran. Now the walk is lazy + dot-dir-pruning, with a
hard files-visited budget and a wall-clock timeout acting as backstops, and
`list_directory` surfaces a subdirectory's scale so the agent can see a big
tree *before* searching it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from paf.capabilities.fs import FileSystemToolset


def _fs(tmp_path: Path, *, walk_visit_budget: int = 50_000) -> FileSystemToolset:
    return FileSystemToolset(
        root_dir=tmp_path,
        allowed_patterns=[],
        denied_patterns=[],
        protected_patterns=[".git/**", ".env"],
        max_read_lines=2000,
        max_search_results=1000,
        max_find_results=1000,
        max_search_file_bytes=512 * 1024,
        search_excerpt_chars=200,
        max_search_total_bytes=256 * 1024,
        walk_visit_budget=walk_visit_budget,
    )


def _run(coro):
    return asyncio.run(coro)


def _scatter(root: Path, n: int) -> None:
    """Create `n` small files across a nested tree (each under 1KB)."""
    for i in range(n):
        (root / f"dir{i % 5}" / f"f{i:04d}.py").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / f"dir{i % 5}" / f"f{i:04d}.py").write_text(
            f"x = {i}\n", encoding="utf-8"
        )


def test_search_files_finds_without_hanging(tmp_path: Path) -> None:
    _scatter(tmp_path, 200)
    (tmp_path / "target.py").write_text("# needle-1\n", encoding="utf-8")
    fs = _fs(tmp_path)
    out = _run(fs.search_files("needle-1"))
    assert out == "target.py:1:# needle-1"


def test_list_directory_shows_scale_for_subdirs(tmp_path: Path) -> None:
    # one small dir and one dir large enough to push past a tiny display cap
    (tmp_path / "small").mkdir()
    (tmp_path / "small" / "a.txt").write_text("hi", encoding="utf-8")
    (tmp_path / "big").mkdir()
    for i in range(600):
        (tmp_path / "big" / f"f{i:04d}.py").write_text(f"x={i}\n", encoding="utf-8")
    fs = _fs(tmp_path)
    out = _run(fs.list_directory("."))
    small_row = next(line for line in out.splitlines() if line.startswith("small/"))
    big_row = next(line for line in out.splitlines() if line.startswith("big/"))
    # bounded scale cap (5k) not hit for either; counts are exact
    assert "1 files" in small_row
    assert "600 files" in big_row
    assert "(" in big_row and ")" in big_row  # carries a size hint


def test_search_files_budget_backstop_returns_marker(tmp_path: Path) -> None:
    # more files than the tiny budget so the walk must stop early
    _scatter(tmp_path, 4000)
    fs = _fs(tmp_path, walk_visit_budget=50)
    out = _run(fs.search_files("zzzz_no_match"))
    assert "stopped at the file-walk budget" in out
    assert "visiting 50 files" in out


def test_search_files_timeout_backstop_returns_marker(tmp_path: Path) -> None:
    _scatter(tmp_path, 4000)
    fs = _fs(tmp_path)
    out = _run(fs.search_files("zzzz_no_match", timeout=0.0))
    assert "timed out after" in out
    assert "narrow the search" in out


def test_search_files_skips_dot_dirs(tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "site.py").write_text("# hidden needle\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("needle too\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "real.py").write_text("# needle here\n", encoding="utf-8")
    fs = _fs(tmp_path)
    out = _run(fs.search_files("needle"))
    assert "hidden" not in out
    assert ".venv" not in out
    assert ".git" not in out
    assert "real.py" in out


def test_find_files_bounded_on_broad_glob(tmp_path: Path) -> None:
    _scatter(tmp_path, 20)
    fs = _fs(tmp_path)
    out = _run(fs.find_files("**/*.py"))
    assert out.count(".py") >= 20  # all findable, never hangs
