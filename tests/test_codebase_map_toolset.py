"""CodebaseMapToolset — the navigation-first five-tool ladder (LOCKED read-model).

Tests the agent-facing tools against a small synthetic fixture scanned with
``scan_and_write``:

- ``get_overview``      → tree with dirs/files + counts
- ``lookup_file``       → 4 shapes (head / symbol head / detail / symbol detail)
- ``read_block``        → exact source region + whole-file + missing-file
- ``search_codebase``   → end-to-end over the built tokens.json, head snippets
- ``get_repo_map``      → ranked zero-LLM orient map (Phase 2), budget-bound
- ``run_codebase_map``  → re-indexes the rooted repo (new/changed files surfaced)

All tool methods return bounded, readable ``str`` (None only for unknown
path/symbol), matching the git/fs/shell toolset convention.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paf.capabilities.codebase_map import (
    CodebaseMap,
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


def test_get_overview_shows_nested_tree(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    root, t = scanned
    out = t.get_overview()
    assert f"Overview of {root.resolve()}" in out
    # Real nesting: dir header then its files indented underneath, basenames only
    # (not the requalified paths repeated on every line).
    assert "src/  (2 files)" in out
    assert "  a.py  (" in out  # a.py indented under src/
    assert "top.py" in out


def test_get_overview_max_depth_folds_subdirs(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    """max_depth caps the open tree: subdirs below the cap show their header with
    contents folded (not expanded), keeping the overview bounded on big repos."""
    _, t = scanned
    out = t.get_overview(max_depth=1)
    # depth-1 dirs show but their children (depth 2) are folded away.
    assert "src/  (2 files)" in out
    assert "  a.py  (" not in out  # depth-2 file under src/ is folded
    assert "top.py" in out  # depth-1 file still visible


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


def test_lookup_file_detail_has_signature_no_body(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    # Whole-file detail is METADATA only (kind/signature/docstring/span) — no bodies.
    # It must stay bounded no matter the file size; source/body comes from read_block
    # or the single-symbol form. See the lookup_file docstring contract.
    _, t = scanned
    out = t.lookup_file("src/a.py", detail=True)
    assert out is not None
    assert "make" in out
    assert "tool: str" in out  # signature
    assert "made-" not in out  # body text must NOT be dumped


def test_lookup_file_symbol_detail_has_body(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    # The single-symbol form DOES include the body — the bounded "read one symbol".
    _, t = scanned
    out = t.lookup_file("src/a.py", symbol="helper", detail=True)
    assert out is not None
    assert "helper" in out
    assert "x: int" in out or "Add one" in out
    assert "return x + 1" in out  # the specific symbol's body is present


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


class _FakeStorage(FlatFileStorage):
    """Storage-seam fake: artifacts delegate to a real FlatFileStorage, but
    ``read_block`` is fully overridden to serve canned bytes — proving the
    toolset routes the pointer through the seam instead of touching the FS."""

    def __init__(self, *, underlying: FlatFileStorage, canned: str):
        self._underlying = underlying
        self._canned = canned

    # -- delegate artifact I/O to a real backend ---------------------------------
    def read(self, kind, scan_root):
        return self._underlying.read(kind, scan_root)

    def write(self, kind, scan_root, payload):
        self._underlying.write(kind, scan_root, payload)

    def list_scan_roots(self):
        return self._underlying.list_scan_roots()

    def sync_changed(self, scan_root):
        return self._underlying.sync_changed(scan_root)

    # -- the point: source bytes come from THIS backend, not the local file ------
    def read_block(self, block_ref) -> str | None:
        return f"SERVED-BY-SEAM:{block_ref.file}"


def test_read_block_routes_through_storage_seam(tmp_path: Path) -> None:
    """A custom backend's read_block is honored: the toolset must NOT read the
    local file directly, it must cross the seam (option 1 fix)."""
    root = tmp_path / "project"
    root.mkdir()
    _seed(root, "src/a.py", "def f():\n    return 1\n")

    real = FlatFileStorage()
    fake = _FakeStorage(underlying=real, canned="SERVED")
    t = CodebaseMapToolset(scan_root=root, storage=fake)  # type: ignore[arg-type]

    out = t.read_block("src/a.py")
    assert out is not None
    assert out.startswith("SERVED-BY-SEAM:")
    # the seam got the root-resolved absolute path, and the escape guard passed
    assert (root / "src/a.py").resolve().as_posix() in out


# ------------------------------------------------------------- search_codebase


def test_search_codebase_hits_live_artifacts(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    _, t = scanned
    out = t.search_codebase("Widget")
    assert "Widget" in out
    assert "src/a.py" in out


def test_search_codebase_no_results(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    _, t = scanned
    assert "No results" in t.search_codebase("zznotawordzz")


def test_search_codebase_emits_pasteable_read_block_ref(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    """The whole point of the index: a search hit collapses to a pointer. The hit
    must expose a directly-usable ``read_block`` ref (``read: path:L1-L2``) so the
    agent can paste it straight into ``read_block`` without re-typing the path."""
    _, t = scanned
    out = t.search_codebase("Widget")
    assert "Widget" in out
    assert "read: " in out
    # Pull the first "read: <ref>" line and feed it back into read_block — it must
    # resolve to the matched symbol's source region, not return None.
    ref = next(line.strip().removeprefix("read: ") for line in out.splitlines() if "read: " in line)
    assert ref.startswith("src/a.py:")
    block = t.read_block(ref)
    assert block is not None
    assert "Widget" in block  # the source region actually contains the matched symbol


# ------------------------------------------------------------------ get_repo_map


def test_get_repo_map_built(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    """get_repo_map is now BUILT (Phase 2): produces a ranked zero-LLM map over the
    scanned fixture (a fresh scan carries call_graph.json), no longer a stub."""
    _, t = scanned
    out = t.get_repo_map(map_tokens=20)
    assert "Repo map of" in out
    assert "symbols" in out
    # The fixture is ranked: src/a.py (Widget + helper, referenced by others) should
    # appear; and it must be real manifest data, not an error message.
    assert "src/a.py" in out or "capabilities" in out


def test_search_exposed_tools_are_the_six_live_ladder(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    """The agent-facing surface is the six live tools — get_repo_map is registered."""
    _, t = scanned
    assert sorted(t.tools) == [
        "get_overview",
        "get_repo_map",
        "lookup_file",
        "read_block",
        "run_codebase_map",
        "search_codebase",
    ]


def test_run_codebase_map_reindexes(scanned: tuple[Path, CodebaseMapToolset]) -> None:
    root, t = scanned
    # A file added AFTER the initial scan won't be in the index yet.
    _seed(root, "src/late.py", "def late_fn():\n    return 42\n")
    assert t.lookup_file("src/late.py") is None  # not indexed yet
    out = t.run_codebase_map()
    assert "src/" in out or "files extracted" in out
    # After re-index, the new file is surfaced.
    assert t.lookup_file("src/late.py") is not None


def test_codebase_map_capability_steers_code_questions_to_ladder(tmp_path: Path) -> None:
    """The capability injects a tool-choice steer toward the ladder for code.

    The steer (mere guidance, not removal) exists so the demo agent reaches for
    the scoped codebase tools (which skip `data/`/`.agent-codeindex/`) instead of
    a raw `search_files` over the whole tree on code-related questions.
    """
    cap = CodebaseMap(scan_root=tmp_path)
    instructions = cap.get_instructions()
    assert instructions is not None
    text = " ".join(instructions)
    assert "reach for the codebase tools" in text or "start with the codebase tools" in text
    assert "get_overview" in text
    assert "search_codebase" in text
    assert "lookup_file" in text
    assert "read_block" in text
    # It steers, it does not remove — must not tell the model to avoid editing tools.
    assert "Keep the raw FileSystem tools" in text
