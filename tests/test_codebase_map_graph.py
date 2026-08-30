"""Graph gate — build_edges() reproduces the hand-verified gold edge set (Phase 2).

Runs ``build_edges`` over the **paf-src fixture** (a snapshot of this repo's own
``src/paf/``) and asserts that every gold ``{source -> target, symbols}`` edge in
``tests/code_search_eval/gold_edges.jsonl`` is present with all of its referenced
symbols. The gold set is hand-written from the actual import/use relationships in
paf code the author knows — so "the graph is correct" means "it reproduces the
edges a competent human can verify by reading the source", not that it matches an
opaque machine-derived answer.

Why the fixture (not live ``src/paf``): the graph/map tests must stay independent
of refactors — they assert against the frozen copy. See tests/fixtures/README.md.

We build ``FileInfo`` records with ``analyze_file`` (per-file) rather than
``scan_and_write``: the scan routes through ``ScanScope`` which treats a gitignored
fixture inside the repo as *out of scope* (0 files). Calling the analyzer directly
is exactly the TS-gate pattern and isolates ``build_edges`` — the unit under test —
from the git-ignore scope machinery.

Two expectations encoded here:
  1. **Edge presence** — every gold (source-path, target-path) pair exists in the
     extracted edge set (i.e. source really does reference/use something target
     defines).
  2. **Symbol fidelity** — every gold ``symbols`` name rides on that edge, which
     proves the *resolution* is correct (``BlockRef`` resolves to ``_schema.py``,
     not ``_block.py``, etc.).

The fixture is gitignored; if absent this gate skips (suite stays green on a
clean checkout).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paf.capabilities.codebase_map._analyze import analyze_file
from paf.capabilities.codebase_map._graph import build_edges
from paf.capabilities.codebase_map._schema import FileInfo

ROOT = Path(__file__).resolve().parents[1]  # repo root
GOLD = ROOT / "tests" / "code_search_eval" / "gold_edges.jsonl"
FIXTURE = ROOT / "tests" / "fixtures" / "paf-src"

_SRC_EXTS = (".py",)


def _gold_edges() -> list[dict]:
    if not GOLD.exists():
        return []
    return [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]


def _fixture_present() -> bool:
    # paf-src snapshot of src/paf/ -> has capabilities/ + runtime.py + config.py
    return FIXTURE.exists() and (FIXTURE / "capabilities").exists()


def _load_infos() -> list[FileInfo]:
    """Analyze every .py fixture file directly (bypasses git-ignore ScanScope)."""
    infos: list[FileInfo] = []
    for py in sorted(FIXTURE.rglob("*.py")):
        if "node_modules" in py.parts or ".agent-codeindex" in py.parts:
            continue
        fi = analyze_file(py, FIXTURE)
        if fi is not None:
            infos.append(fi)
    return infos


@pytest.fixture(scope="module")
def edge_map() -> dict[tuple[str, str], set[str]]:
    """Build edges over the fixture and index by (source, target) -> symbols."""
    if not _fixture_present():
        pytest.skip("paf-src fixture not present (gitignored) — sync per tests/fixtures/README.md")
    edges = build_edges(_load_infos())
    return {(e["source"], e["target"]): set(e["symbols"]) for e in edges}


def test_build_edges_is_deterministic() -> None:
    """Same input -> identical output across runs (no hash-order flakiness)."""
    if not _fixture_present():
        pytest.skip("paf-src fixture not present (gitignored) — sync per tests/fixtures/README.md")
    infos = _load_infos()
    assert infos, "fixture produced no FileInfo records"
    e1 = build_edges(infos)
    e2 = build_edges(infos)
    assert e1 == e2
    assert e1, "expected some edges over the paf-src fixture"


def test_every_gold_edge_pair_present(edge_map: dict) -> None:
    """Every hand-verified (source -> target) pair is in the extracted edge set."""
    assert edge_map, "edge_map empty — fixture produced no edges"
    for g in _gold_edges():
        src, dst = g["edge"].split(" -> ")
        assert (src, dst) in edge_map, (
            f"gold edge missing: {g['edge']} — source does not appear to use target's symbols"
        )


@pytest.mark.parametrize("entry", _gold_edges(), ids=lambda e: e["edge"])
def test_gold_edge_symbol_fidelity(edge_map: dict, entry: dict) -> None:
    """Each gold edge's symbols resolve onto that exact edge (correct definition)."""
    src, dst = entry["edge"].split(" -> ")
    got = edge_map.get((src, dst), set())
    missing = set(entry.get("symbols", [])) - got
    assert not missing, (
        f"symbol(s) {sorted(missing)} not resolved on {entry['edge']} "
        f"(have {sorted(got)}) — likely defined in a different file"
    )
