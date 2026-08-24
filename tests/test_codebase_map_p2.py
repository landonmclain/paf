"""Phase 2 tests: references_count population + BM25 first-pass search.

Covers the new Phase-2 surface:
  - ``reference_counts`` / ``apply_references`` — inbound in-degree at scan time,
    persisted into each ``FileInfo.references_count`` in the manifest.
  - ``rank`` / ``rank_fused`` — field-weighted BM25 over ``FileInfo`` records.
  - ``LexicalRetrieval.search`` — end-to-end from a written manifest.
"""

from __future__ import annotations

from pathlib import Path

from paf.capabilities.codebase_map import (
    FlatFileStorage,
    LexicalRetrieval,
    apply_references,
    rank,
    rank_fused,
    reference_counts,
    scan_and_write,
)
from paf.capabilities.codebase_map._schema import FileInfo, SymbolInfo


def _file(
    rel: str,
    symbols: list[str] | None = None,
    imports: list[str] | None = None,
    doc: str = "",
) -> dict:
    """A manifest-shaped record (the detail layer `apply_references` reads).

    `symbols` here is the full asdict(SymbolInfo) list — the manifest record shape,
    NOT the lean-index name list. `imports` feeds import-based reference counting.
    """
    symbols = symbols or []
    imports = imports or []
    syms = [SymbolInfo(name=n, kind="FUNCTION", signature=f"def {n}()", body=f"def {n}(): ...") for n in symbols]
    return {
        "path": rel,
        "language": "python",
        "doc": doc,
        "imports": list(imports),
        "symbols": [__import__("dataclasses").asdict(s) for s in syms],
        "references_count": 0,
        "unparsed": False,
    }


# ------------------------------------------------------------------ references


def test_reference_counts_inbound_across_files() -> None:
    files = [
        FileInfo(path="a.py", symbols=["Settings"], imports=[], searchable_symbols=[]),
        FileInfo(path="b.py", symbols=["run"], imports=["Settings"], searchable_symbols=[]),
        FileInfo(path="c.py", symbols=["helper"], imports=["Settings", "run"], searchable_symbols=[]),
    ]
    counts = reference_counts(files)
    # b and c both reference Settings (defined in a); c references run (defined in b)
    assert counts["a.py"] == 2
    assert counts["b.py"] == 1
    assert counts["c.py"] == 0


def test_reference_counts_excludes_self_reference() -> None:
    files = [
        FileInfo(path="a.py", symbols=["Settings"], imports=["Settings"], searchable_symbols=[]),
        FileInfo(path="b.py", symbols=["x"], imports=[], searchable_symbols=[]),
    ]
    counts = reference_counts(files)
    assert counts["a.py"] == 0  # a imports its own Settings → not counted


def test_apply_references_mutates_manifest_records() -> None:
    records = [
        _file("a.py", symbols=["Settings"]),
        _file("b.py", imports=["Settings"]),
        _file("c.py", imports=["Settings", "run"]),
    ]
    apply_references(records)
    by_path = {r["path"]: r["references_count"] for r in records}
    assert by_path["a.py"] == 2
    assert by_path["b.py"] == 0
    assert by_path["c.py"] == 0


# ------------------------------------------------------------------ rank (BM25)


def _f(rel: str, symbols: list[str] | None = None, imports: list[str] | None = None, doc: str = "") -> FileInfo:
    symbols = symbols or []
    imports = imports or []
    syms = [SymbolInfo(name=n, signature=f"def {n}()", body=f"def {n}(): ...") for n in symbols]
    return FileInfo(path=rel, symbols=list(symbols), imports=list(imports), doc=doc, searchable_symbols=syms)


def test_rank_name_field_weights_high() -> None:
    files = [
        _f("a.py", symbols=["foo_brag"], doc="unrelated text"),
        _f("b.py", symbols=["other"], doc="the foo_brag thing happens here"),
    ]
    hits = rank("foo_brag", files)
    assert hits and hits[0].file == "a.py"  # name hit beats doc-only hit


def test_rank_returns_searchhit_with_blockref_pointer() -> None:
    files = [_f("a.py", symbols=["foo"], doc="about foo")]
    hits = rank("foo", files)
    assert hits
    hit = hits[0]
    assert hit.file == "a.py"
    assert hit.block is not None and hit.block.file == "a.py"


def test_rank_ref_count_is_weak_tiebreak() -> None:
    # Two files score identically on text; the one with the higher reference
    # count edges ahead ONLY by the tiny tiebreak (never text-dominating).
    low = _f("a.py", symbols=["widget"], doc="widget")
    high = _f("b.py", symbols=["widget"], doc="widget")
    low = FileInfo(**{**low.__dict__, "references_count": 1})
    high = FileInfo(**{**high.__dict__, "references_count": 9})
    hits = rank("widget", [low, high], ref_weight=True)
    assert hits[0].file == "b.py"
    # without the tiebreak they'd tie — confirm the tiebreak is what ordered it
    hits_no = rank("widget", [low, high], ref_weight=False)
    assert hits_no[0].file == "a.py"  # a comes first on stable (equal) order


def test_rank_empty_result_on_no_match() -> None:
    files = [_f("a.py", symbols=["foo"])]
    assert rank("zzzznope", files) == []


def test_rank_fused_alias() -> None:
    files = [_f("a.py", symbols=["foo"])]
    assert rank_fused("foo", files) == rank("foo", files)


# ------------------------------------------------------------------ LexicalRetrieval


def test_lexical_retrieval_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def Settings(): ...\n", encoding="utf-8")
    (root / "b.py").write_text("from a import Settings\ndef run(): ...\n", encoding="utf-8")
    scan_and_write(root)

    retr = LexicalRetrieval(storage=FlatFileStorage(), scan_root=root)
    hits = retr.search("Settings", limit=5)
    assert hits
    assert hits[0].file == "a.py"  # defines Settings (name field weighs highest)
