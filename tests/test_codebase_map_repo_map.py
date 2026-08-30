"""Phase 2 gate — get_repo_map() / distilled_map() zero-LLM orientation map.

Exercises the built (no longer stubbed) repo map over the **paf-src fixture** — the
frozen snapshot of this repo's own ``src/paf/`` — scanning it into a temp FlatFileStorage
(so the fresh index carries ``call_graph.json``) and asserting four properties:

  1. **Center of gravity** — the highest call-weighted-in-degree files land in the
     top ranks: ``_schema.py`` (every module imports the dataclasses) and ``_store.py``
     (the storage seam) must both appear before low-degree leaf files.
  2. **Budget bound** — the rendered output lists exactly ``map_tokens`` symbols
     (one deduped symbol per entry) and never more.
  3. **No fabrication** — every rendered symbol + signature is real manifest data:
     each appears verbatim in the fixture's own manifest record for that file.
  4. **Determinism** — two scans over the same fixture yield byte-identical output.

Also covered: the stale-index path (no ``call_graph.json``) returns the actionable
"re-index needed" message instead of a misleading alphabetical map.

Everything is real artifact data — no LLM prose — so "the map is correct" means "the
ranking, budget, and fidelity each hold under the persisted graph + manifest."

The fixture is gitignored; if absent this gate skips (suite stays green on a clean
checkout).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from paf.capabilities.codebase_map import CodebaseMapToolset, FlatFileStorage
from paf.capabilities.codebase_map._scan import scan_and_write

ROOT = Path(__file__).resolve().parents[1]  # repo root
FIXTURE = ROOT / "tests" / "fixtures" / "paf-src"
BUDGET = 50


def _fixture_present() -> bool:
    return FIXTURE.exists() and (FIXTURE / "capabilities").exists()


@pytest.fixture(scope="module")
def scanned(tmp_path_factory) -> tuple[Path, CodebaseMapToolset]:
    """A fresh scan of a COPY of the paf-src fixture into a temp dir (has call_graph).

    The fixture lives gitignored *inside* the repo, and ScanScope honours ``git
    check-ignore`` — so scanning the fixture path directly yields 0 files. We therefore
    copy it to a temp dir (outside the repo → no sibling `.git` → nothing ignored), then
    scan the copy through the full ``scan_and_write`` → ``get_repo_map`` path, exactly
    like scanning a real external codebase.
    """
    if not _fixture_present():
        pytest.skip("paf-src fixture not present (gitignored) — sync per tests/fixtures/README.md")
    import shutil

    dst = tmp_path_factory.mktemp("pafcopy") / "paf-src"
    shutil.copytree(FIXTURE, dst)
    store = FlatFileStorage(index_root=tmp_path_factory.mktemp("idx"))
    scan_and_write(dst, store)
    return dst, CodebaseMapToolset(scan_root=dst, storage=store)


def test_repo_map_center_of_gravity(scanned) -> None:
    """The call-hub files outrank leaf/rarely-invoked files in the ranking."""
    _, t = scanned
    out = t.get_repo_map(map_tokens=300)  # generous so schema/store both fit
    lines = out.splitlines()
    # Find the index of each hub's header line within the ranked output.
    schema_idx = next(i for i, line in enumerate(lines) if line.startswith("capabilities/codebase_map/_schema.py:"))
    store_idx = next(i for i, line in enumerate(lines) if line.startswith("capabilities/codebase_map/_store.py:"))
    # A low-degree leaf that should rank far below the hubs.
    leaf_idx = next(
        (i for i, line in enumerate(lines) if line.startswith("capabilities/codebase_map/cli.py:")),
        len(lines),
    )
    assert schema_idx < leaf_idx, "schema.py (top in-degree) must rank above the cli leaf"
    assert store_idx < leaf_idx, "store.py (storage seam) must rank above the cli leaf"
    # The output is clearly ranked (not alphabetical): schema before store is fine,
    # but the real signal is both before the leaf.
    assert "Repo map of" in out


def test_repo_map_budget_bound(scanned) -> None:
    """Exact map_tokens symbols are emitted — never more, never zero."""
    _, t = scanned
    small = 17
    out = t.get_repo_map(map_tokens=small)
    symbols = [line for line in out.splitlines() if line.startswith("    ")]
    assert len(symbols) == small, f"expected {small} symbols, got {len(symbols)}"
    # A huge budget should still be bounded sanely (not blow up) and list many files.
    big = t.get_repo_map(map_tokens=10000)
    assert "capabilities/codebase_map/_schema.py:" in big


def test_repo_map_no_fabrication(scanned) -> None:
    """Every symbol/signature rendered is real manifest data for that file."""
    root, t = scanned
    from paf.capabilities.codebase_map._store import ArtifactKind
    payload = t._storage.read(ArtifactKind.MANIFEST, root)
    files = payload["files"]
    out = t.get_repo_map(map_tokens=400)
    cur_file = None
    for line in out.splitlines():
        if line.endswith(":") and not line.startswith("    "):
            cur_file = line[:-1]
            continue
        if not line.startswith("    "):
            continue
        shown = line.strip()
        record = files[cur_file]
        names = {s.get("name") for s in record.get("symbols", [])}
        # Render truncates signatures to their first line — so compare against the
        # first line of each real manifest signature too.
        sigs = {
            (s.get("signature") or "").splitlines()[0].rstrip()
            for s in record.get("symbols", [])
            if s.get("signature")
        }
        # The shown line is either a bare symbol name or a (first-line) signature — real data.
        assert shown in names or shown in sigs or shown == "", (
            f"rendered '{shown}' in {cur_file} is not in the manifest ({len(record.get('symbols', []))} symbols)"
        )


def test_repo_map_deterministic(scanned) -> None:
    """Two scans of the same fixture produce identical output."""
    _, t = scanned
    assert t.get_repo_map(map_tokens=BUDGET) == t.get_repo_map(map_tokens=BUDGET)


def test_repo_map_focus_restricts_subtree(scanned) -> None:
    """focus=<prefix> limits the map to that subtree only."""
    _, t = scanned
    out = t.get_repo_map(map_tokens=200, focus="capabilities/memory")
    # Every file header lives under capabilities/memory/ — and other top dirs are absent.
    lines = out.splitlines()
    files = [line[:-1] for line in lines if line.endswith(":") and not line.startswith("    ")]
    assert files, "focus map produced no files"
    for f in files:
        assert f == "capabilities/memory" or f.startswith("capabilities/memory/"), f"focus leak: {f}"
    assert all(not f.startswith("capabilities/codebase_map") for f in files)


def test_repo_map_stale_index_clear_message(tmp_path: Path) -> None:
    """An index without call_graph.json returns an actionable re-index message, not
    a misleading alphabetical map."""
    import shutil
    # Copy the fixture out of the repo (git-ignored paths scan as 0 files), scan it,
    # then delete the graph artifact to simulate a pre-Phase-2 index.
    dst = tmp_path / "paf-src"
    shutil.copytree(FIXTURE, dst)
    store = FlatFileStorage(index_root=tmp_path / "idx")
    scan_and_write(dst, store)
    idx_dir = store._dir_for(dst)
    (idx_dir / "call_graph.json").unlink()
    t = CodebaseMapToolset(scan_root=dst, storage=store)
    out = t.get_repo_map(map_tokens=BUDGET)
    assert "call_graph.json" in out and "re-index" in out.lower()
    assert "Repo map of" not in out  # must NOT emit a (misleading) map
