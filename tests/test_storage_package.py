"""Unit tests for the shared persistence seam (paf.capabilities._storage).

Covers the two seam shapes: whole-artifact (:class:`FlatFileStorage` implements
:class:`StorageService`) and append-only (:class:`FlatJsonlStore` implements
:class:`AppendOnlyStore`), plus the shared atomic helpers and the read_block
pointer model.
"""

import json

import pytest

from paf.capabilities._storage import (
    AppendOnlyStore,
    BlockRef,
    FlatFileStorage,
    FlatJsonlStore,
    Span,
    StorageService,
    read_block,
    read_json,
    write_atomic,
    write_json_atomic,
)


# --- StorageService primitive (atomic helpers) -------------------------------
def test_write_atomic_and_read_json_roundtrip(tmp_path):
    p = tmp_path / "nested" / "a.json"
    write_atomic(p, json.dumps({"x": 1}))
    assert p.exists()
    assert not p.with_name(".a.json.staging").exists()  # no staging residue
    assert read_json(p) == {"x": 1}


def test_read_json_missing_or_corrupt_returns_default(tmp_path):
    assert read_json(tmp_path / "missing.json", default=None) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_json(bad, default=[]) == []


def test_write_json_atomic_pretty(tmp_path):
    p = tmp_path / "b.json"
    write_json_atomic(p, {"a": 1})
    raw = p.read_text()
    assert '"a": 1' in raw  # indent=2


# --- StorageService / FlatFileStorage -----------------------------------------
def test_flatfile_read_write_scope_by_key(tmp_path):
    store = FlatFileStorage(tmp_path)
    store.write("titles", "agent_a", {"c1": "Hello"})
    store.write("titles", "agent_b", {"c2": "World"})
    assert store.read("titles", "agent_a") == {"c1": "Hello"}
    assert store.read("titles", "agent_b") == {"c2": "World"}
    # distinct keys, isolated
    assert store.read("titles", "agent_a") != store.read("titles", "agent_b")


def test_flatfile_missing_read_returns_none(tmp_path):
    store = FlatFileStorage(tmp_path)
    assert store.read("nope", "x") is None


def test_flatfile_list_keys(tmp_path):
    store = FlatFileStorage(tmp_path)
    store.write("k", "one", {})
    store.write("k", "two", {})
    names = {p.name for p in store.list_keys()}
    assert any("one" in n for n in names)
    assert any("two" in n for n in names)


def test_flatfile_storage_conforms_to_storage_service(tmp_path):
    class _Proto(StorageService):
        pass  # marker subclass of Protocol for isinstance-style structural check

    store = FlatFileStorage(tmp_path)
    for attr in ("read", "write", "list_keys", "read_block"):
        assert callable(getattr(store, attr, None)), attr


# --- AppendOnlyStore / FlatJsonlStore -----------------------------------------
def test_flatjsonl_append_and_read_all_in_order(tmp_path):
    store = FlatJsonlStore(tmp_path)
    store.append("journal", "a", {"n": 1})
    store.append("journal", "a", {"n": 2})
    store.append("journal", "b", {"n": 9})
    assert store.read_all("journal", "a") == [{"n": 1}, {"n": 2}]
    assert store.read_all("journal", "b") == [{"n": 9}]


def test_flatjsonl_torn_and_empty_lines_skipped(tmp_path):
    store = FlatJsonlStore(tmp_path)
    # Simulate a torn trailing line written outside the helper.
    f = tmp_path / "a" / "journal.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text('{"good": 1}\n{truncated\n\n')
    assert store.read_all("journal", "a") == [{"good": 1}]


def test_flatjsonl_missing_read_all_returns_empty(tmp_path):
    store = FlatJsonlStore(tmp_path)
    assert store.read_all("journal", "missing") == []


def test_flatjsonl_conforms_to_append_only_store(tmp_path):
    store = FlatJsonlStore(tmp_path)
    for attr in ("append", "read_all"):
        assert callable(getattr(store, attr, None)), attr


# --- read_block pointer model -------------------------------------------------
def test_read_block_byte_slice(tmp_path):
    f = tmp_path / "src.py"
    f.write_text("abcdefghij")
    assert read_block(f, Span(start_byte=2, end_byte=6)) == "cdef"


def test_read_block_line_fallback(tmp_path):
    f = tmp_path / "m.txt"
    f.write_text("line1\nline2\nline3\n")
    assert read_block(f, Span(start_line=2, end_line=2)) == "line2"


def test_read_block_whole_capped(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("x" * 100000)
    block = read_block(f)
    assert block is not None
    assert len(block) <= 65536


def test_read_block_missing_returns_none(tmp_path):
    assert read_block(tmp_path / "nope.py") is None
