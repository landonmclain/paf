"""Shared fixtures for the codebase_map gate tests.

The paf-src fixture (a frozen snapshot of ``src/paf/``) is scanned by several
codebase_map test modules. Scanning it is the slowest part of the suite, so we
scan it ONCE per session and share the result instead of re-scanning per module.
The fixture is lazy (session-scoped): it only runs when a test requests it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from paf.capabilities.codebase_map._analyze import analyze_file
from paf.capabilities.codebase_map._schema import FileInfo

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "paf-src"


@pytest.fixture(scope="session")
def paf_src_scan() -> tuple[list[FileInfo], dict[str, str]] | None:
    """Analyze every .py in the paf-src fixture once (session-scoped, lazy).

    Returns ``(infos, sources)`` — the FileInfo records plus the original source
    text keyed by repo-relative path (needed for call-site re-parsing). ``None``
    if the gitignored fixture is absent (callers skip).
    """
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
