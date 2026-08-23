"""p1-gate: evaluate extraction completeness against the eval gold set (real paf tree).

Runs ``analyze_file`` over the real paf ``src/`` tree (the ship path) and checks the
Phase-1-relevant gold answers in ``tests/code_search_eval/queries.jsonl`` are actually
surfaced by extraction — every gold ``answer_files`` path must extract, and every gold
``answer_symbols`` name must appear in that file's ``searchable_symbols``.

This is the repeatable form of the Phase 1 exit gate "extract real paf python files;
hand-inspect; check against eval gold". Call-graph gold queries (``mode='callers'``)
are skipped here — call graph is derived on read in Phase 2.

The gate is intentionally tolerant to repo growth: it runs over whatever is in
``src/paf`` today and only asserts the gold entries that point at existing files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paf.capabilities.codebase_map._analyze import analyze_file

ROOT = Path(__file__).resolve().parents[2]  # repo root
QUERIES = ROOT / "tests" / "code_search_eval" / "queries.jsonl"
SRC = ROOT / "src" / "paf"


def _gold_entries() -> list[dict]:
    if not QUERIES.exists():
        return []  # empty param set -> parametrized tests are skipped naturally
    return [json.loads(line) for line in QUERIES.read_text().splitlines() if line.strip()]


def test_extraction_completeness_zero_errors() -> None:
    """Every paf python file either extracts cleanly or is a legit no-symbol skip."""
    errors: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        try:
            analyze_file(py, ROOT)
        except Exception as e:  # noqa: BLE001 — gate must never crash
            errors.append(f"{py.relative_to(ROOT)}: {e!r}")
    assert not errors, f"{len(errors)} extraction errors:\n" + "\n".join(errors[:10])


@pytest.mark.parametrize("entry", _gold_entries(), ids=lambda e: e["query"])
def test_gold_answer_surfaced_by_extraction(entry: dict) -> None:
    """Each Phase-1 (non-call-graph) gold answer's file+symbols are extracted."""
    if entry.get("mode") == "callers":
        pytest.skip("call-graph gold — Phase 2 (derived on read)")
    for ans_file in entry.get("answer_files", []):
        p = ROOT / ans_file
        if not p.exists():
            pytest.skip(f"gold file no longer exists: {ans_file}")
        fi = analyze_file(p, ROOT)
        assert fi is not None, f"gold file not extracted: {ans_file}"

        want = entry.get("answer_symbols", [])
        if not want:
            return  # file-level query only — extraction to non-None is enough

        names = {s.name for s in fi.searchable_symbols}
        quals = {s.qualified for s in fi.searchable_symbols if s.qualified}
        for w in want:
            assert w in names or any(q.endswith(w) for q in quals), (
                f"gold symbol {w!r} not surfaced in {ans_file}"
            )
