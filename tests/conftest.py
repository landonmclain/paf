"""Shared fixtures for the codebase_map gate tests.

The paf-src fixture (a frozen snapshot of ``src/paf/``) is scanned by several
codebase_map test modules. Scanning it is the slowest part of the suite, so it is
built ONCE per test run and shared — across modules and across pytest-xdist
workers (see ``tests/_shared.py``). Fixtures are lazy: they only run when a test
requests them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paf.capabilities.codebase_map._analyze import analyze_file
from paf.capabilities.codebase_map._schema import FileInfo
from tests._shared import shared_build

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "paf-src"


def _scan_paf_src() -> tuple[list[FileInfo], dict[str, str]] | None:
    if not (FIXTURE / "capabilities").exists():
        return None
    infos: list[FileInfo] = []
    sources: dict[str, str] = {}
    for py in sorted(FIXTURE.rglob("*.py")):
        if "node_modules" in py.parts or ".agent-codeindex" in py.parts:
            continue
        fi = analyze_file(py, FIXTURE)
        if fi is not None:
            infos.append(fi)
            sources[fi.path] = py.read_text()
    return infos, sources


@pytest.fixture(scope="session")
def paf_src_scan(
    pytestconfig: pytest.Config, tmp_path_factory: pytest.TempPathFactory
) -> tuple[list[FileInfo], dict[str, str]] | None:
    """Analyze every .py in the paf-src fixture once per test run (lazy).

    Returns ``(infos, sources)`` — the FileInfo records plus the original source
    text keyed by repo-relative path (needed for call-site re-parsing). ``None``
    if the gitignored fixture is absent (callers skip).
    """
    return shared_build(pytestconfig, tmp_path_factory, "paf_src_scan", _scan_paf_src)


@pytest.fixture(autouse=True)
def _no_jev_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the pairwise reviewer's Jev screen off so no test hits OpenRouter."""
    monkeypatch.setenv("PAF_PAIR_SCREEN", "0")
