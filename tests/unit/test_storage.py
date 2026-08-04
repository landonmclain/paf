"""Unit tests for the flat-file storage seam."""

from __future__ import annotations

from pathlib import Path

from paf.storage.flatfile import FlatFileStorage


def test_set_get_roundtrip(tmp_path: Path) -> None:
    store = FlatFileStorage(tmp_path)
    store.set("alice", {"name": "alice", "n": 1})
    assert store.get("alice") == {"name": "alice", "n": 1}


def test_get_missing_returns_none(tmp_path: Path) -> None:
    store = FlatFileStorage(tmp_path)
    assert store.get("nope") is None


def test_search_matches_filenames(tmp_path: Path) -> None:
    store = FlatFileStorage(tmp_path)
    store.set("greetings_01", {"x": 1})
    store.set("notes_abc", {"x": 2})
    hits = store.search("greet")
    assert "greetings_01" in hits
    assert "notes_abc" not in hits


def test_delete_removes_key(tmp_path: Path) -> None:
    store = FlatFileStorage(tmp_path)
    store.set("tmp", 1)
    store.delete("tmp")
    assert store.get("tmp") is None
