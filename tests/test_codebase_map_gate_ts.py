"""TS/TSX/JS extraction gate — real chatbot fixture gold (p1-other-langs, TS slice).

Mirrors ``test_codebase_map_gate.py`` (the Python gate) for the TypeScript family:
runs ``analyze_file`` over real ``tests/fixtures/chatbot`` files and asserts that
the gold ``answer_files`` / ``answer_symbols`` in ``queries_ts.jsonl`` are surfaced
by extraction — every gold file must extract, and every gold symbol must appear in
that file's ``searchable_symbols``.

Why a real-repo fixture: the corpus is 154 tracked TS/TSX/JS files (93 tsx, 59 ts,
1 js, 1 jsx) from the pinned vercel/chatbot clone. The gold concentrates on
**module-level named assignments** — the gap this pass fixed (previously python-only
``module_assignments`` meant schema/config/constants files indexed as *empty* and
were invisible to search). See tests/fixtures/README.md for how to re-materialize
the (gitignored) clone.

The gate is intentionally tolerant: it skips any gold file that no longer exists in
the fixture, so it stays green across fixture re-clones. The fixture itself is
gitignored, so on a clean checkout this file has no fixture -> all gold params skip.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paf.capabilities.codebase_map._analyze import analyze_file

ROOT = Path(__file__).resolve().parents[1]  # repo root
QUERIES = ROOT / "tests" / "code_search_eval" / "queries_ts.jsonl"
FIXTURE = ROOT / "tests" / "fixtures" / "chatbot"


def _gold_entries() -> list[dict]:
    if not QUERIES.exists():
        return []
    return [json.loads(line) for line in QUERIES.read_text().splitlines() if line.strip()]


def _fixture_present() -> bool:
    # vercel/chatbot has no `src/` — it uses app/ + components/ + lib/ + hooks/. Use `lib/`.
    return FIXTURE.exists() and (FIXTURE / "lib").exists()


def test_ts_extraction_completeness_zero_errors() -> None:
    """Every TS/TSX/JS fixture file either extracts cleanly or is a legit skip."""
    if not _fixture_present():
        pytest.skip("chatbot fixture not present (gitignored) — re-clone per tests/fixtures/README.md")
    errors: list[str] = []
    for f in sorted(FIXTURE.rglob("*")):
        if f.suffix.lower() not in (".ts", ".tsx", ".js", ".jsx"):
            continue
        if "node_modules" in f.parts or ".git" in f.parts:
            continue
        try:
            analyze_file(f, FIXTURE)
        except Exception as e:  # noqa: BLE001 — gate must never crash
            errors.append(f"{f.relative_to(FIXTURE)}: {e!r}")
    assert not errors, f"{len(errors)} extraction errors:\n" + "\n".join(errors[:10])


@pytest.mark.parametrize("entry", _gold_entries(), ids=lambda e: e["query"])
def test_ts_gold_answer_surfaced_by_extraction(entry: dict) -> None:
    """Each TS gold answer's file+symbols are surfaced by extraction."""
    if not _fixture_present():
        pytest.skip("chatbot fixture not present — re-clone per tests/fixtures/README.md")
    for ans_file in entry.get("answer_files", []):
        p = FIXTURE / ans_file
        if not p.exists():
            pytest.skip(f"gold file no longer exists: {ans_file}")
        fi = analyze_file(p, FIXTURE)
        assert fi is not None, f"gold file not extracted: {ans_file}"

        want = entry.get("answer_symbols", [])
        if not want:
            return  # file-level query only — extraction to non-None is enough

        names = {s.name for s in fi.searchable_symbols}
        quals = {s.qualified for s in fi.searchable_symbols if s.qualified}
        missing = [w for w in want
                   if w not in names and not any(q.endswith(w) for q in quals)]
        assert not missing, (
            f"gold symbol(s) {missing} not surfaced in {ans_file} "
            f"(have {sorted(names)})"
        )
