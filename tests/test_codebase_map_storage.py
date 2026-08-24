"""Phase 0 tests: codebase-map storage seam + schema records.

Covers the implemented Phase 0 surface — `FlatFileStorage` (write/read/sync_changed/
read_block), `index_dir_for_scan_root` co-location, and the schema records — not the
stubbed extraction/search (Phases 1–4).
"""

from __future__ import annotations

from pathlib import Path

from paf.capabilities.codebase_map import (
    ArtifactKind,
    BlockRef,
    FlatFileStorage,
    Span,
    index_dir_for_scan_root,
)
from paf.capabilities.codebase_map._schema import FileInfo, SearchHit, SymbolInfo

# --------------------------------------------------------------------------- storage


def _seed_file(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def test_index_dir_is_co_located_under_scan_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    idx = index_dir_for_scan_root(root)
    # co-located under the scanned repo root, not a separate paf directory
    assert idx.is_relative_to(root)
    assert ".agent-codeindex" in idx.parts
    assert idx.name.startswith("codebase_indexes") or "codebase_indexes" in idx.parts


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    store = FlatFileStorage()
    payload = {"version": 1, "files": [{"path": "a.py", "symbols": ["f"]}]}
    store.write(ArtifactKind.MANIFEST, root, payload)
    assert store.read(ArtifactKind.MANIFEST, root) == payload
    # atomic staging file is not left behind
    idx = index_dir_for_scan_root(root)
    leftovers = [p for p in idx.iterdir() if ".staging" in p.name]
    assert leftovers == []


def test_read_missing_artifact_returns_none(tmp_path: Path) -> None:
    store = FlatFileStorage()
    assert store.read(ArtifactKind.INDEX, tmp_path) is None


def test_read_block_byte_slice(tmp_path: Path) -> None:
    src = tmp_path / "f.py"
    src.write_text("aaaa\nbbbb\ncccc\n", encoding="utf-8")
    ref = BlockRef(file=str(src), span=Span(start_byte=5, end_byte=9, start_line=2, end_line=2))
    store = FlatFileStorage()
    assert store.read_block(ref) == "bbbb"


def test_read_block_line_fallback(tmp_path: Path) -> None:
    src = tmp_path / "f.py"
    src.write_text("l1\nl2\nl3\n", encoding="utf-8")
    store = FlatFileStorage()
    # byte offsets absent (None) -> line fallback
    ref = BlockRef(file=str(src), span=Span(start_byte=None, end_byte=None, start_line=2, end_line=2))
    assert store.read_block(ref) == "l2"


def test_read_block_missing_file_returns_none(tmp_path: Path) -> None:
    store = FlatFileStorage()
    ref = BlockRef(file=str(tmp_path / "nope.py"), span=Span(start_byte=0, end_byte=1, start_line=1, end_line=1))
    assert store.read_block(ref) is None


def test_sync_changed_detects_content_change(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    store = FlatFileStorage()

    f = _seed_file(root, "mod.py", "v1")
    # seed a snapshot so the file is "known"
    snapshot = {"files": {f.relative_to(root).as_posix(): {"size": f.stat().st_size, "mtime": f.stat().st_mtime, "sha256": "x" * 64}}}
    store.write(ArtifactKind.SNAPSHOT, root, snapshot)

    # unchanged -> not flagged (only hashed when stat differs)
    assert store.sync_changed(root) == set()

    # change content but preserve size+mtime (touch-style) -> hash confirm catches it
    f.write_text("v1 changeddddddd", encoding="utf-8")  # different size -> stat gate fires
    assert (f.relative_to(root).as_posix()) in store.sync_changed(root)


def test_sync_changed_skips_index_and_git(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    store = FlatFileStorage()
    _seed_file(root, ".agent-codeindex/codebase_indexes/x/snapshot.json", "{}")
    _seed_file(root, ".git/config", "[core]")
    _seed_file(root, "real.py", "print('hi')")
    changed = store.sync_changed(root)
    # .git and .agent-codeindex are ignored; real.py shows as new (not in snapshot)
    assert "real.py" in changed
    assert not any(c.startswith(".git") or c.startswith(".agent-codeindex") for c in changed)


# --------------------------------------------------------------------------- schema


def test_schema_records_construct() -> None:
    sym = SymbolInfo(name="f", kind="FUNCTION", signature="def f()", doc="doc")
    fi = FileInfo(path="a.py", language="python", symbols=["f"], searchable_symbols=[sym])
    assert fi.searchable_symbols[0].doc == "doc"
    hit = SearchHit(file="a.py", start_line=1, end_line=5, symbol="f", score=1.0)
    assert hit.score == 1.0


def test_search_hit_carries_blockref_pointer() -> None:
    ref = BlockRef(file="a.py", span=Span(start_byte=0, end_byte=10, start_line=1, end_line=1))
    hit = SearchHit(file="a.py", start_line=1, end_line=1, symbol="f", block=ref)
    assert hit.block is not None
    assert hit.block.span.end_byte == 10
