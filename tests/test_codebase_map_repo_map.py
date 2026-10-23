"""The repo map (``get_repo_map``) over a scanned copy of the sample repo.

The sample repo is copied out of this checkout before scanning so the scan sees
a standalone codebase rather than a subtree of paf's own git repository.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from paf.capabilities.codebase_map import CodebaseMapToolset, FlatFileStorage
from paf.capabilities.codebase_map._scan import scan_and_write
from paf.capabilities.codebase_map._store import ArtifactKind

BUDGET = 50


def _scan_copy(sample_repo: Path, into: Path) -> tuple[Path, FlatFileStorage]:
    repo = into / "repo"
    shutil.copytree(sample_repo, repo)
    store = FlatFileStorage(index_root=into / "idx")
    scan_and_write(repo, store)
    return repo, store


@pytest.fixture(scope="module")
def scanned(sample_repo, tmp_path_factory) -> tuple[Path, CodebaseMapToolset]:
    repo, store = _scan_copy(sample_repo, tmp_path_factory.mktemp("repo_map"))
    return repo, CodebaseMapToolset(scan_root=repo, storage=store)


def _file_headers(out: str) -> list[str]:
    return [
        line[:-1]
        for line in out.splitlines()
        if line.endswith(":") and not line.startswith("    ")
    ]


def test_the_cross_package_hub_outranks_self_coupled_code(scanned) -> None:
    """config.py is called from every package; reports/ only talks to itself."""
    _, t = scanned
    files = _file_headers(t.get_repo_map(map_tokens=300))
    assert files[0] == "shop/config.py", files
    first_report = next(i for i, f in enumerate(files) if f.startswith("shop/reports/"))
    assert files.index("shop/config.py") < first_report


def test_the_budget_is_the_exact_number_of_symbols_shown(scanned) -> None:
    _, t = scanned
    out = t.get_repo_map(map_tokens=17)
    assert sum(1 for line in out.splitlines() if line.startswith("    ")) == 17
    assert "web/next.config.ts:" in t.get_repo_map(map_tokens=10000)


def test_every_rendered_symbol_is_real_manifest_data(scanned) -> None:
    root, t = scanned
    files = t._storage.read(ArtifactKind.MANIFEST, root)["files"]  # noqa: SLF001
    current = None
    for line in t.get_repo_map(map_tokens=400).splitlines():
        if line.endswith(":") and not line.startswith("    "):
            current = line[:-1]
            continue
        if not line.startswith("    "):
            continue
        shown = line.strip()
        symbols = files[current].get("symbols", [])
        names = {s.get("name") for s in symbols}
        # Signatures render as their first line only.
        sigs = {
            (s.get("signature") or "").splitlines()[0].rstrip()
            for s in symbols
            if s.get("signature")
        }
        assert shown in names or shown in sigs, f"{current}: invented {shown!r}"


def test_the_map_is_deterministic(scanned) -> None:
    _, t = scanned
    assert t.get_repo_map(map_tokens=BUDGET) == t.get_repo_map(map_tokens=BUDGET)


def test_focus_restricts_the_map_to_a_subtree(scanned) -> None:
    _, t = scanned
    files = _file_headers(t.get_repo_map(map_tokens=200, focus="shop/orders"))
    assert files and all(f.startswith("shop/orders/") for f in files), files


def test_an_index_without_a_call_graph_asks_for_a_reindex(
    sample_repo, tmp_path
) -> None:
    """A pre-graph index gets an actionable message, not a misleading map."""
    repo, store = _scan_copy(sample_repo, tmp_path)
    (store._dir_for(repo) / "call_graph.json").unlink()  # noqa: SLF001
    out = CodebaseMapToolset(scan_root=repo, storage=store).get_repo_map(
        map_tokens=BUDGET
    )
    assert "call_graph.json" in out and "re-index" in out.lower()
    assert "Repo map of" not in out
