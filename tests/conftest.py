"""Shared test fixtures.

The codebase-map tests analyze ``tests/fixtures/sample_repo``, a small committed
codebase built to exercise specific analyzer behaviors (see its README).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paf.capabilities.codebase_map._analyze import analyze_file
from paf.capabilities.codebase_map._schema import FileInfo

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REPO = ROOT / "tests" / "fixtures" / "sample_repo"


def _analyze(suffixes: tuple[str, ...]) -> dict[str, FileInfo]:
    infos: dict[str, FileInfo] = {}
    for path in sorted(SAMPLE_REPO.rglob("*")):
        if path.is_file() and path.suffix in suffixes:
            fi = analyze_file(path, SAMPLE_REPO)
            if fi is not None:
                infos[fi.path] = fi
    return infos


@pytest.fixture(scope="session")
def sample_repo() -> Path:
    return SAMPLE_REPO


@pytest.fixture(scope="session")
def sample_py() -> tuple[list[FileInfo], dict[str, str]]:
    """The sample repo's Python files: ``(infos, sources)`` for ``build_edges``."""
    infos = _analyze((".py",))
    sources = {p: (SAMPLE_REPO / p).read_text(encoding="utf-8") for p in infos}
    return list(infos.values()), sources


@pytest.fixture(scope="session")
def sample_ts() -> dict[str, FileInfo]:
    """The sample repo's TypeScript files, keyed by repo-relative path."""
    return _analyze((".ts", ".tsx"))


@pytest.fixture(autouse=True)
def _no_jev_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the pairwise reviewer's Jev screen off so no test hits OpenRouter."""
    monkeypatch.setenv("PAF_PAIR_SCREEN", "0")
