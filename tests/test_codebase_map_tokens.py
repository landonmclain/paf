"""Tokens search backend + slimmed lean-head index tests (LOCKED read-model, 2026-08-23).

Covers the new Phase-3-adjacent surface landed with the read-model:
  - ``build_tokens`` — inverted, SYMBOL-SCOPED postings; tokenized fields are
    name + signature + doc ONLY (body deliberately excluded).
  - ``tokens_rank`` — field-weighted relevance (name > signature > doc); hits
    collapse to a symbol's ``manifest_ref`` ``BlockRef``.
  - ``scan_and_write`` writes the LEAN ``index.json`` head (NO searchable_symbols /
    body / signature / span / role / tags) AND ``tokens.json``.
  - ``LexicalRetrieval.search`` routes over tokens and presents the index head.
"""

from __future__ import annotations

from pathlib import Path

from paf.capabilities.codebase_map import (
    ArtifactKind,
    FlatFileStorage,
    LexicalRetrieval,
    build_tokens,
    scan_and_write,
    tokens_rank,
)
from paf.capabilities.codebase_map._schema import FileInfo, Span, SymbolInfo


def _fi(
    rel: str, symbols: list[SymbolInfo], doc: str = "", imports: list[str] | None = None
) -> FileInfo:
    return FileInfo(
        path=rel,
        language="python",
        doc=doc,
        symbols=[s.name for s in symbols],
        imports=imports or [],
        searchable_symbols=symbols,
    )


def _sym(name: str, kind: str = "FUNCTION", **kw) -> SymbolInfo:
    kw.setdefault("span", Span(start_line=3, end_line=10))
    return SymbolInfo(name=name, kind=kind, **kw)


# ------------------------------------------------------------------ build_tokens


def test_build_tokens_symbol_scoped_with_field_tags() -> None:
    fi = _fi(
        "a.py",
        [_sym("widget", signature="def widget(maker, tool)", doc="make a widget")],
    )
    tok = build_tokens([fi])["files"]["a.py"]
    # the symbol name appears as a name-field posting
    name_posts = tok.get("widget", [])
    assert any(
        p["field"] == "name" and p["symbol"] == "widget" and p["kind"] == "FUNCTION"
        for p in name_posts
    )
    # tokens from signature and doc are also present, tagged with their field
    sig_posts = tok.get("widget", [])  # 'widget' appears in name + signature
    assert any(p["field"] == "signature" for p in sig_posts)
    # a signature-only word is tagged signature
    assert any(p["field"] == "signature" for p in tok.get("maker", []))
    # a doc-only word is tagged doc
    assert any(
        p["field"] == "doc" and p["symbol"] == "widget" for p in tok.get("make", [])
    )
    # every posting carries a manifest_ref pointing at the symbol's block
    for post in tok.get("widget", []):
        assert post["manifest_ref"] == {"start_line": 3, "end_line": 10}


def test_build_tokens_filters_stopwords_and_short_tokens() -> None:
    fi = _fi(
        "a.py",
        [_sym("widget", doc="there is a widget in the systemzone")],
    )
    tok = build_tokens([fi])["files"]["a.py"]
    # envelope words and short tokens are dropped from the index...
    for noise in ("there", "is", "a", "in", "the"):
        assert noise not in tok
    # ...while the signal words survive
    assert "widget" in tok and "systemzone" in tok


def test_tokens_rank_stopword_query_is_empty() -> None:
    # a query that is only stopwords must not match anything (vocabulary parity
    # between build and query keeps a stopword-only query from matching the
    # same stopword in the index)
    a = _fi("a.py", [_sym("widget", doc="the there is")])
    hits = tokens_rank("the there is", build_tokens([a]))
    assert hits == []


def test_build_tokens_excludes_body() -> None:
    # body text must NOT be tokenized (locked: name + signature + doc only)
    fi = _fi("a.py", [_sym("widget", body="widget_body_word_xyz only in here")])
    tok = build_tokens([fi])["files"]["a.py"]
    name_posts = tok.get("widget", [])
    # symbol name 'widget' present (name field), but body-only words are absent
    assert name_posts and any(p["field"] == "name" for p in name_posts)
    assert "only_in_here" not in tok
    assert "widget_body_word_xyz" not in tok


def test_build_tokens_skips_unparsed_files() -> None:
    unparsed = FileInfo(
        path="data.bin", language=None, symbols=[], imports=[], unparsed=True
    )
    tok = build_tokens([unparsed])
    assert tok["files"] == {}


# ------------------------------------------------------------------ tokens_rank


def test_tokens_rank_name_field_beats_doc() -> None:
    a = _fi("a.py", [_sym("foo_brag")], doc="unrelated prose")
    b = _fi("b.py", [_sym("other")], doc="the foo_brag thing happens here")
    tok = build_tokens([a, b])
    hits = tokens_rank("foo_brag", tok)
    assert hits and hits[0].file == "a.py"  # name hit outweighs doc-only mention


def test_tokens_rank_hit_collapses_to_symbol_block() -> None:
    a = _fi("a.py", [_sym("foo", span=Span(start_line=7, end_line=12))])
    hits = tokens_rank("foo", build_tokens([a]), heads={"a.py": {"doc": "head doc"}})
    assert hits
    hit = hits[0]
    assert hit.symbol == "foo"
    assert hit.block is not None and hit.block.file == "a.py"
    assert (
        hit.block.span.start_line == 7 and hit.block.span.end_line == 12
    )  # symbol block, not whole file
    assert hit.snippet == "head doc"  # index HEAD presentation


def test_tokens_rank_empty_on_no_match() -> None:
    a = _fi("a.py", [_sym("foo")])
    assert tokens_rank("zzzznope", build_tokens([a])) == []


# --------------------------------------------------- scan writes slim head + tokens


def test_scan_writes_lean_head_and_tokens(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text(
        '"""module doc."""\nclass Widget:\n    def spin(self): ...\ndef helper(): ...\n',
        encoding="utf-8",
    )
    scan_and_write(root)
    storage = FlatFileStorage()

    # lean head: name lists only, no symbol detail, no role/tags
    index = storage.read(ArtifactKind.INDEX, root)
    row = index["files"][0]
    assert set(row) == {
        "path",
        "language",
        "doc",
        "imports",
        "classes",
        "functions",
        "symbols",
        "references_count",
        "unparsed",
    }
    assert "Widget" in row["classes"]
    assert "helper" in row["functions"]
    assert row["doc"] == "module doc."
    assert "searchable_symbols" not in row and "body" not in row and "span" not in row

    # tokens artifact exists and is searchable-symbol-scoped
    tokens = storage.read(ArtifactKind.TOKENS, root)
    assert tokens["version"] == 1
    assert "a.py" in tokens["files"]
    assert any(
        p["symbol"] == "Widget" for p in tokens["files"]["a.py"].get("widget", [])
    )


def test_lexical_retrieval_search_over_tokens(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def Settings(): ...\n", encoding="utf-8")
    (root / "b.py").write_text(
        "from a import Settings\ndef run(): ...\n", encoding="utf-8"
    )
    scan_and_write(root)

    retr = LexicalRetrieval(storage=FlatFileStorage(), scan_root=root)
    hits = retr.search("Settings", limit=5)
    assert hits
    # 'Settings' is a defined symbol NAME in a.py (highest-weight field); b.py only
    # imports it (imports not tokenized) — so a.py must rank first and collapse to the
    # Settings symbol block rather than the whole file.
    assert hits[0].file == "a.py"
    assert hits[0].symbol == "Settings"
    assert hits[0].block is not None and hits[0].block.file == "a.py"


def test_manifest_keeps_detail_and_references(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("class Settings: ...\n", encoding="utf-8")
    (root / "b.py").write_text("from a import Settings\n", encoding="utf-8")
    scan_and_write(root)
    storage = FlatFileStorage()
    manifest = storage.read(ArtifactKind.MANIFEST, root)["files"]
    # detail layer keeps full symbols (with signature/span) for a.py
    a_syms = manifest["a.py"]["symbols"]
    assert a_syms and "signature" in a_syms[0] and "span" in a_syms[0]
    # references calculated over manifest, mirrored onto the lean index head
    index_row = {r["path"]: r for r in storage.read(ArtifactKind.INDEX, root)["files"]}[
        "a.py"
    ]
    assert index_row["references_count"] == 1  # b.py references Settings
    assert manifest["a.py"]["references_count"] == 1
