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

Reference expectations (gold_edges.jsonl):
  1. **Edge presence** — every gold (source-path, target-path) pair is present.
  2. **Symbol fidelity** — every gold ``symbols`` name rides on that edge.

Call-edge expectations (Phase 1.2; gold_calls.jsonl + gold_no_call.jsonl):
  3. **Positive calls** — every gold call edge is ``kind: \"call\"`` and carries
     every gold ``call_symbols`` name (the resolver really proved those invocations).
  4. **Negative non-calls** — every gold no-call edge stays ``kind: \"reference\"``.
     This is the real safety check: it proves we never *over-claim* a call for a
     symbol that is merely referenced (type annotations, docstrings, mentions).

The fixture is gitignored; if absent this gate skips (suite stays green on a
clean checkout).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paf.capabilities.codebase_map._graph import build_edges
from tests._shared import shared_build

ROOT = Path(__file__).resolve().parents[1]  # repo root
GOLD = ROOT / "tests" / "code_search_eval" / "gold_edges.jsonl"
GOLD_CALLS = ROOT / "tests" / "code_search_eval" / "gold_calls.jsonl"
GOLD_NO_CALL = ROOT / "tests" / "code_search_eval" / "gold_no_call.jsonl"
_SRC_EXTS = (".py",)


def _gold_edges() -> list[dict]:
    if not GOLD.exists():
        return []
    return [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]


def _gold_calls() -> list[dict]:
    if not GOLD_CALLS.exists():
        return []
    return [
        json.loads(line) for line in GOLD_CALLS.read_text().splitlines() if line.strip()
    ]


def _gold_no_call() -> list[dict]:
    if not GOLD_NO_CALL.exists():
        return []
    return [
        json.loads(line)
        for line in GOLD_NO_CALL.read_text().splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def edges(pytestconfig, tmp_path_factory, paf_src_scan) -> list[dict]:
    """The full edge set over the fixture (reference + call refinement)."""
    if paf_src_scan is None:
        pytest.skip(
            "paf-src fixture not present (gitignored) — sync per tests/fixtures/README.md"
        )
    infos, sources = paf_src_scan
    return shared_build(
        pytestconfig,
        tmp_path_factory,
        "paf_src_edges",
        lambda: build_edges(infos, sources),
    )


@pytest.fixture(scope="module")
def edge_map(edges: list[dict]) -> dict[tuple[str, str], set[str]]:
    """Index edges by (source, target) -> full referenced symbol set."""
    return {(e["source"], e["target"]): set(e["symbols"]) for e in edges}


@pytest.fixture(scope="module")
def call_map(edges: list[dict]) -> dict[tuple[str, str], dict]:
    """Index edges by (source, target) -> {kind, call_symbols}."""
    return {(e["source"], e["target"]): e for e in edges}


def test_build_edges_is_deterministic(paf_src_scan) -> None:
    """Same input -> identical output across runs (no hash-order flakiness)."""
    if paf_src_scan is None:
        pytest.skip(
            "paf-src fixture not present (gitignored) — sync per tests/fixtures/README.md"
        )
    infos, sources = paf_src_scan
    assert infos, "fixture produced no FileInfo records"
    e1 = build_edges(infos, sources)
    e2 = build_edges(infos, sources)
    assert e1 == e2
    assert e1, "expected some edges over the paf-src fixture"
    # Some edges must be calls once sources are supplied (Phase 1.2 works).
    assert any(e["kind"] == "call" for e in e1), (
        "call refinement produced no call edges"
    )


def test_no_sources_gives_all_reference(paf_src_scan) -> None:
    """Without sources, build_edges degrades to all-reference (graceful)."""
    if paf_src_scan is None:
        pytest.skip(
            "paf-src fixture not present (gitignored) — sync per tests/fixtures/README.md"
        )
    infos, _ = paf_src_scan
    refs = build_edges(infos)
    assert refs and all(e["kind"] == "reference" for e in refs)
    assert all(e["call_symbols"] == [] for e in refs)


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


@pytest.mark.parametrize("entry", _gold_calls(), ids=lambda e: e["edge"])
def test_gold_call_is_call(call_map: dict, entry: dict) -> None:
    """Every gold call edge is marked kind=call and carries the verified call symbols."""
    src, dst = entry["edge"].split(" -> ")
    assert (src, dst) in call_map, f"gold call edge missing outright: {entry['edge']}"
    e = call_map[(src, dst)]
    assert e["kind"] == "call", (
        f"expected {entry['edge']} to be a CALL but got kind={e['kind']}"
    )
    missing = set(entry.get("call_symbols", [])) - set(e["call_symbols"])
    assert not missing, (
        f"call symbol(s) {sorted(missing)} not resolved as calls on {entry['edge']} "
        f"(call_symbols={e['call_symbols']})"
    )


@pytest.mark.parametrize("entry", _gold_no_call(), ids=lambda e: e["edge"])
def test_gold_no_call_stays_reference(call_map: dict, entry: dict) -> None:
    """Edges that are only references (types/docstrings/mentions) must NOT be calls."""
    src, dst = entry["edge"].split(" -> ")
    assert (src, dst) in call_map, f"no-call edge not present at all: {entry['edge']}"
    e = call_map[(src, dst)]
    assert e["kind"] == "reference", (
        f"expected {entry['edge']} to stay REFERENCE (symbols {entry.get('symbols')}) "
        f"but it was upgraded to a call: call_symbols={e['call_symbols']}"
    )
    assert e["call_symbols"] == [], (
        f"no-call edge {entry['edge']} wrongly claims call_symbols={e['call_symbols']}"
    )
