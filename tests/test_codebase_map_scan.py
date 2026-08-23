"""Scan-scope tests (spec §5.y): walk-everything-not-ignored + unparsed records.

Covers the LOCKED 2026-08-22 design:
- ``walk_all_files`` returns every file that isn't (gitignore ∪ owned-skip) — language
  agnostic, never hardcodes an extension.
- In-scope files with no parseable symbols appear in the artifacts as ``unparsed: true``
  records and land in the snapshot as *seen* (stable — never "changed").
- ``sync_changed`` reports only in-scope files whose content actually changed; gitignored /
  out-of-scope files (e.g. ``.venv/``) are never reported as "changed".
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from paf.capabilities.codebase_map import (
    ArtifactKind,
    FlatFileStorage,
    ScanScope,
    walk_all_files,
)
from paf.capabilities.codebase_map._scan import scan_and_write


def _seed(root: Path, rel: str, content: str = "x") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- walk scope


def test_owned_skip_dirs_excluded_without_git(tmp_path: Path) -> None:
    """With no git repo, owned-skip dirs (.git, .agent-codeindex, History, archive,
    node_modules, __pycache__) are excluded; everything else is walked."""
    root = tmp_path / "project"
    root.mkdir()
    _seed(root, ".git/config")
    _seed(root, ".agent-codeindex/codebase_indexes/x/snapshot.json")
    _seed(root, "node_modules/pkg/index.js")
    _seed(root, "__pycache__/f.cpython-312.pyc")
    _seed(root, "History/old-note.md")
    _seed(root, "archive/stale.py")
    _seed(root, "src/a.py", "x = 1")
    _seed(root, "notes.md", "# note")
    _seed(root, "data.json", "{}")

    files = walk_all_files(root)
    rels = {p.relative_to(root).as_posix() for p in files}
    # everything that should be excluded stays out; in-scope files are walked
    for excluded in (".git/config", ".agent-codeindex", "node_modules", "__pycache__",
                     "History", "archive"):
        assert not any(r.startswith(excluded) for r in rels), f"{excluded} should be skipped"
    assert rels == {"src/a.py", "notes.md", "data.json"}


def test_walk_is_language_agnostic(tmp_path: Path) -> None:
    """The walker never hardcodes an extension — code files AND non-code files both
    appear (the extractor decides what to parse, not the walk)."""
    root = tmp_path / "project"
    root.mkdir()
    _seed(root, "mod.py")
    _seed(root, "app.ts")
    _seed(root, "index.html")
    _seed(root, "logo.png", b"\x89PNG\r\n".decode("latin1"))
    rels = {p.relative_to(root).as_posix() for p in walk_all_files(root)}
    assert rels == {"mod.py", "app.ts", "index.html", "logo.png"}


def test_gitignore_excludes_with_real_git(tmp_path: Path) -> None:
    """When the scan root is a real git repo, git-check-ignore excludes gitignored
    files (.venv, data/, tests/results/) and honors owned-skip (History)."""
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text(
        ".venv/\ndata/\ntests/results/\n__pycache__/\n", encoding="utf-8"
    )
    _seed(root, ".venv/lib/python/site.py")
    _seed(root, "data/paf/run.json")
    _seed(root, "tests/results/out.json")
    _seed(root, "src/mod.py")
    _seed(root, "History/stale.md")

    rels = {p.relative_to(root).as_posix() for p in walk_all_files(root)}
    # .gitignore itself is a real repo file (in scope); gitignored + owned-skip excluded
    assert rels == {"src/mod.py", ".gitignore"}


def test_scan_scope_in_scope_direct(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _seed(root, "a.py")
    _seed(root, "History/x.md")
    scope = ScanScope(root)
    assert scope.in_scope(root / "a.py")
    assert not scope.in_scope(root / "History" / "x.md")


# --------------------------------------------------------------------------- unparsed records


def test_scan_and_write_records_unparsed_files(tmp_path: Path) -> None:
    """In-scope files with no parseable symbols appear in manifest + structure as
    unparsed:true, and are snapshotted as seen (so they never read as 'changed')."""
    root = tmp_path / "project"
    root.mkdir()
    _seed(root, "mod.py", "def f():\n    return 1\n")
    _seed(root, "notes.md", "just a note")
    _seed(root, "data.json", "{}")

    store = FlatFileStorage()
    summary = scan_and_write(root, store)
    assert summary["files_found"] == 3
    assert set(summary["unparsed"]) == {"notes.md", "data.json"}

    manifest = store.read(ArtifactKind.MANIFEST, root)["files"]
    by_path = {m["path"]: m for m in manifest}
    assert by_path["mod.py"]["unparsed"] is False
    assert by_path["mod.py"]["symbols"] == ["f"]
    assert by_path["notes.md"]["unparsed"] is True
    assert by_path["notes.md"]["language"] is None
    assert by_path["data.json"]["unparsed"] is True

    structure = store.read(ArtifactKind.STRUCTURE, root)["files"]
    assert structure["notes.md"]["unparsed"] is True
    assert structure["mod.py"]["unparsed"] is False

    # unparsed files are in the snapshot too -> seen, stable
    snapshot = store.read(ArtifactKind.SNAPSHOT, root)["files"]
    assert "notes.md" in snapshot and "data.json" in snapshot


# --------------------------------------------------------------------------- freshness scope


def test_sync_changed_ignores_out_of_scope_files(tmp_path: Path) -> None:
    """sync_changed must NOT report gitignored/out-of-scope files (e.g. .venv) as
    'changed' — the root cause of the 62,715-false-changed bug. It reports only real
    in-scope changes."""
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text(".venv/\n", encoding="utf-8")

    store = FlatFileStorage()
    scan_and_write(root, store)  # snapshot wins scope: only in-scope files

    # nothing changed -> empty (no false positives from .venv)
    assert store.sync_changed(root) == set()

    # touch a gitignored file -> still not "changed" (out of scope)
    venv_file = root / ".venv" / "x.py"
    venv_file.parent.mkdir(parents=True, exist_ok=True)
    venv_file.write_text("v2" * 100, encoding="utf-8")
    assert store.sync_changed(root) == set()

    # change a real in-scope file -> it IS reported
    mod = root / "mod.py"
    mod.write_text("def f():\n    return 2\n", encoding="utf-8")
    mod.write_text("def f():\n    return 99999\n", encoding="utf-8")
    changed = store.sync_changed(root)
    assert "mod.py" in changed
    assert not any(".venv" in c for c in changed)
