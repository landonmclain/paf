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
    assert not any(
        "History" in p or "archive" in p or "__pycache__" in p for p in paths
    )


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


def test_dir_importance_aggregates_descendant_files() -> None:
    """_dir_importance maps each dir to the (mean, max) imp across files rooted
    under it — through nested subdirs — mirroring the module-level stat."""
    from paf.capabilities.codebase_map._toolset import _dir_importance

    tree = [
        {"path": "capabilities", "type": "dir"},
        {"path": "capabilities/memory", "type": "dir"},
        {"path": "capabilities/_turns.py", "type": "file"},
        {"path": "capabilities/memory/_capability.py", "type": "file"},
        {"path": "config.py", "type": "file"},
    ]
    imp = {
        "capabilities/_turns.py": 10.0,
        "capabilities/memory/_capability.py": 28.0,
        "config.py": 21.0,  # top-level file: no dir shares its imp
    }
    di = _dir_importance(tree, imp)
    # capabilities/ covers BOTH descendants (through nesting)
    assert di["capabilities"] == (19.0, 28.0)  # mean((10,28)), max=28
    assert di["capabilities/memory"] == (28.0, 28.0)  # single file
    assert "config.py" not in di  # only dirs get keys


def test_render_tree_annotates_dir_and_file_imp() -> None:
    """_render_tree appends per-file imp=N and per-dir mean/max annotations."""
    from paf.capabilities.codebase_map._toolset import CodebaseMapToolset

    tree = [
        {"path": "capabilities", "type": "dir", "depth": 1, "file_count": 1},
        {
            "path": "capabilities/_turns.py",
            "type": "file",
            "depth": 2,
            "line_count": 32,
        },
        {"path": "config.py", "type": "file", "depth": 1, "line_count": 117},
    ]
    imp = {"capabilities/_turns.py": 13.0, "config.py": 21.0}
    dir_imp = {"capabilities": (13.0, 13.0)}
    out = CodebaseMapToolset._render_tree(tree, imp=imp, dir_imp=dir_imp)
    assert "capabilities/  (1 files)  imp: mean=13.0 max=13" in out
    assert "  _turns.py  (32 lines)  imp=13" in out
    assert "config.py  (117 lines)  imp=21" in out
    # degrades to plain tree when no annotations provided
    plain = CodebaseMapToolset._render_tree(tree)
    assert "imp" not in plain
