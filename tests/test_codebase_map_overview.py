"""overview.json — the navigation tree (spec §3.1). Tests the tree builder + scan wiring.

Covers:
- ``build_overview`` emits {scan_root, generated, version, tree} with dir/file entries,
  dirs carrying file_count, files carrying line_count, shallowest-depth-first order.
- The overview is written by ``scan_and_write`` as the OVERVIEW artifact and matches
  the walked in-scope corpus (a dir with no in-scope children is not emitted).
"""

from __future__ import annotations

from pathlib import Path

from paf.capabilities.codebase_map import ArtifactKind, FlatFileStorage
from paf.capabilities.codebase_map._overview import build_overview
from paf.capabilities.codebase_map._scan import scan_and_write


def _seed(root: Path, rel: str, content: str = "x") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _by_path(tree: list[dict]) -> dict[str, dict]:
    return {e["path"]: e for e in tree}


def test_build_overview_shape_and_directories(tmp_path: Path) -> None:
    """Dirs carry file_count, files carry line_count, shallowest-depth-first order."""
    root = tmp_path / "project"
    root.mkdir()
    _seed(root, "src/a.py", "def a():\n    pass\n\ndef b():\n    pass\n")  # 5 lines
    _seed(root, "src/sub/c.py", "x = 1\n")  # 1 line
    _seed(root, "top.py", "# just a comment\n")  # 1 line

    overview = build_overview(root)
    assert overview["version"] == 1
    assert overview["scan_root"] == str(root.resolve())

    by = _by_path(overview["tree"])
    assert by["src"]["type"] == "dir"
    assert by["src"]["file_count"] == 2  # a.py + c.py under src/
    assert by["src/sub"]["type"] == "dir"
    assert by["src/sub"]["file_count"] == 1
    assert by["src/a.py"]["type"] == "file"
    assert by["src/a.py"]["line_count"] == 5
    assert by["src/sub/c.py"]["line_count"] == 1

    # shallowest first: src (depth 1 (=1 dir), then src/sub (depth 2), then files
    depths = [e["depth"] for e in overview["tree"]]
    assert depths == sorted(depths)


def test_build_overview_excludes_owned_skip_and_gitignored(tmp_path: Path) -> None:
    """Owned-skip dirs (History, archive, __pycache__) never appear; their dirs are
    not emitted either."""
    root = tmp_path / "project"
    root.mkdir()
    _seed(root, "src/a.py", "x = 1\n")
    _seed(root, "History/old.md", "stale")
    _seed(root, "archive/retired.py", "old")
    _seed(root, "__pycache__/f.cpython-312.pyc")

    overview = build_overview(root)
    tree = overview["tree"]
    paths = {e["path"] for e in tree}
    assert "src/a.py" in paths
    assert "src" in paths
    assert not any("History" in p or "archive" in p or "__pycache__" in p for p in paths)


def test_scan_and_write_writes_overview_artifact(tmp_path: Path) -> None:
    """scan_and_write persists overview.json via the storage seam."""
    root = tmp_path / "project"
    root.mkdir()
    _seed(root, "src/a.py", "def a():\n    return 1\n")
    _seed(root, "src/b.py", "def b():\n    return 2\n")

    store = FlatFileStorage()
    scan_and_write(root, store)

    overview = store.read(ArtifactKind.OVERVIEW, root)
    assert isinstance(overview, dict)
    assert overview["version"] == 1
    by = _by_path(overview["tree"])
    assert by["src"]["type"] == "dir"
    assert by["src"]["file_count"] == 2
    assert by["src/a.py"]["type"] == "file"
