"""CodebaseMapToolset — the navigation-first five-tool ladder (LOCKED read-model).

Tests the agent-facing tools against a small synthetic fixture scanned with
``scan_and_write``:

- ``get_overview``      → tree with dirs/files + counts
- ``lookup_file``       → 4 shapes (head / symbol head / detail / symbol detail)
- ``read_block``        → exact source region + whole-file + missing-file
- ``search_codebase``   → end-to-end over the built tokens.json, head snippets
- ``get_repo_map``      → stub raises NotImplementedError (Phase 4)

All tool methods return bounded, readable ``str`` (None only for unknown
path/symbol), matching the git/fs/shell toolset convention.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paf.capabilities.codebase_map import (
    CodebaseMapToolset,
    FlatFileStorage,
)
from paf.capabilities.codebase_map._scan import scan_and_write


def _seed(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


_SRC_A = (
    '"""A module with a docstring."""\n'
    "\n"
    "class Widget:\n"
    '    """A widget factory."""\n'
    "    def make(self, tool: str) -> str:\n"
    '        """Make something from a tool."""\n'
    "        return f\"made-{tool}\"\n"
    "\n"
    "def helper(x: int) -> int:\n"
    '    """Add one."""\n'
    "    return x + 1\n"
)


@pytest.fixture()
def scanned(tmp_path: Path) -> tuple[Path, CodebaseMapToolset]:
    """A scanned fixture + a toolset rooted at it."""
    root = tmp_path / "project"
    root.mkdir()
    _seed(root, "src/a.py", _SRC_A)
    _seed(root, "src/b.py", "def b_only():\n    return 2\n")
    _seed(root, "top.py", "x = 1\n")
    store = FlatFileStorage()
    scan_and_write(root, store)
    toolset = CodebaseMapToolset(scan_root=root, storage=store)
    return root, toolset


# ------------------------------------------------------------------ get_overview


def test_get_overview_shows_tree(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    root, t = scanned
    out = t.get_overview()
    assert f"Overview of {root.resolve()}" in out
    assert "src/  (2 files)" in out
    assert "src/a.py" in out
    assert "top.py" in out


# ------------------------------------------------------------------ lookup_file


def test_lookup_file_head(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    _, t = scanned
    out = t.lookup_file("src/a.py")
    assert out is not None
    assert "# src/a.py" in out
    assert "A module with a docstring" in out
    assert "class Widget" in out.lower() or "Widget" in out  # head shows name list
    assert "# def make" not in out  # head has NO signature detail


def test_lookup_file_symbol_head(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    _, t = scanned
    out = t.lookup_file("src/a.py", symbol="Widget")
    assert out is not None
    assert "Widget" in out
    assert "class" in out.lower()


def test_lookup_file_unknown_symbol_is_none(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    _, t = scanned
    assert t.lookup_file("src/a.py", symbol="Nope") is None


def test_lookup_file_unknown_path_is_none(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    _, t = scanned
    assert t.lookup_file("src/missing.py") is None


def test_lookup_file_detail_has_signature_and_body(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    _, t = scanned
    out = t.lookup_file("src/a.py", detail=True)
    assert out is not None
    assert "make" in out
    assert "tool: str" in out  # signature
    assert "made-" in out  # body text present


def test_lookup_file_symbol_detail(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    _, t = scanned
    out = t.lookup_file("src/a.py", symbol="helper", detail=True)
    assert out is not None
    assert "helper" in out
    assert "x: int" in out or "Add one" in out


# ------------------------------------------------------------------ read_block


def test_read_block_whole_file(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    _, t = scanned
    out = t.read_block("src/a.py")
    assert out is not None
    assert "class Widget" in out
    assert "return f\"made-{tool}\"" in out


def test_read_block_region_by_lines(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    _, t = scanned
    out = t.read_block("src/a.py:1-4")
    assert out is not None
    # line 1 is the module docstring line
    assert out.count("\n") <= 4
    assert "module" in out


def test_read_block_missing_file_none(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    _, t = scanned
    assert t.read_block("src/does-not-exist.py") is None


def test_read_block_path_escape_blocked(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    root, t = scanned
    outside = root.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    assert t.read_block("../outside.txt") is None


# ------------------------------------------------------------- search_codebase


def test_search_codebase_hits_live_artifacts(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    _, t = scanned
    out = t.search_codebase("Widget")
    assert "Widget" in out
    assert "src/a.py" in out


def test_search_codebase_no_results(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    _, t = scanned
    assert "No results" in t.search_codebase("zznotawordzz")


# ------------------------------------------------------------------ get_repo_map


def test_get_repo_map_stub(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    _, t = scanned
    with pytest.raises(NotImplementedError):
        t.get_repo_map()
