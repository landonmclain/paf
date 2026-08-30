"""Fixture-coverage gate — every real corpus in the fixtures produces clean, language-correct
import reference data with NO module-path-basename leakage.

The point of this gate is to lock the property we just fixed against regression, FOR EVERY
codebase we actually test against:

  - ``tests/fixtures/paf-src/``   — Python (snapshot of this repo's own src/paf/)
  - ``tests/fixtures/chatbot/``   — TS/TSX/JS (vercel/chatbot, pinned commit)

Regression this guards (2026-08-29): ``_file_referenced_names`` used to regex the raw import
string, which leaked module-path basenames into the reference-name set — ``from ._store import
MemoryFile`` produced ``_store``, and ``from \"@/lib/utils\"`` produced ``lib``/``utils``.
Those leaked path segments then resolved through ``_defined_name_map`` to any file defining a
symbol of that name, manufacturing false graph edges (``visual/_store.py`` scored 14.0 purely
from ``. _store`` basename collisions across the codebase).

The fix moved imports to tree-sitter's STRUCTURED ``ImportInfo.items``/``alias`` (via
``FileInfo.import_names``), so only actual imported SYMBOL names reach the resolver.

Invariants asserted per fixture:
  1. No IMPOSSIBLE import names (whitespace/quote/operator junk) — only identifiers or plain
     dotted module targets (``anyio.abc`` is a real one).
  2. The specific ``_store`` cross-package collision is gone: ``memory/_view.py`` (cross-pkg,
     no legit ``_store`` binding) must carry ``MemoryFile`` and NOT ``_store``; while
     ``visual/_capability.py`` (same-pkg, genuinely imports its sibling's ``_store`` var)
     MAY legitimately carry it. A blanket "no _store" is wrong — we assert the RESOLUTION
     is scoped correctly, not that common names never exist.
  3. The behavioral symptom is gone: ``visual/_store.py`` must NOT be a cross-module hub in
     the reference graph (it was 14.0 from collisions; should be ~0).
  4. TS/TSX/JS: a real chatbot module resolves only true imports, never leaked path keywords.

Fixtures are gitignored; if absent the corresponding tests skip (suite stays green).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from paf.capabilities.codebase_map._analyze import analyze_file

ROOT = Path(__file__).resolve().parents[1]
PAF_FIXTURE = ROOT / "tests" / "fixtures" / "paf-src"
CB_FIXTURE = ROOT / "tests" / "fixtures" / "chatbot"

# Tokens that can ONLY ever be path segments/keywords, never a legit symbol binding.
_FORBIDDEN_IMPORT_TOKENS = {"from", "import", "as", "return", "def", "class", "export", "default"}
# Dotted module targets and bare identifiers are both legit; this allows `anyio.abc`.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _analyze_dir(root: Path, langs: tuple[str, ...]) -> dict[str, object]:
    out: dict[str, object] = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in langs and "node_modules" not in p.parts:
            fi = analyze_file(p, root)
            if fi is not None:
                out[fi.path] = fi
    return out


@pytest.fixture(scope="module")
def paf_infos() -> dict[str, object]:
    if not PAF_FIXTURE.exists():
        pytest.skip("paf-src fixture not present (gitignored)")
    return _analyze_dir(PAF_FIXTURE, (".py",))


@pytest.fixture(scope="module")
def cb_infos() -> dict[str, object]:
    if not CB_FIXTURE.exists():
        pytest.skip("chatbot fixture not present (gitignored)")
    return _analyze_dir(CB_FIXTURE, (".ts", ".tsx", ".js", ".jsx"))


def test_python_import_names_are_plausible(paf_infos) -> None:
    """Every Python import_names entry is a real identifier or dotted module target."""
    assert paf_infos, "paf-src fixture produced no FileInfo"
    for path, fi in paf_infos.items():
        for name in fi.import_names:
            assert _IDENT_RE.match(name), f"{path}: impossible import name {name!r}"
            assert name not in _FORBIDDEN_IMPORT_TOKENS, f"{path}: keyword leaked: {name!r}"


def test_python__store_cross_package_regression(paf_infos) -> None:
    """Cross-package `from ._store import MemoryFile` must NOT leak `_store`.

    memory/_view.py has no legit `_store` binding — it imports only MemoryFile. So
    `_store` must NOT appear in ITS import_names (the bug had it resolve to
    visual/_store.py's module var via a bare-basename collision).
    """
    view = paf_infos.get("capabilities/memory/_view.py")
    assert view is not None, "fixture missing capabilities/memory/_view.py"
    assert "MemoryFile" in view.import_names, "the real item must resolve"
    assert "_store" not in view.import_names, (
        "_store (module basename) must NOT resolve as a symbol from a cross-package import"
    )
    # No file OUTSIDE the visual package should resolve `_store` — visual is the only
    # package whose _store module var is legitimately imported by same-package siblings.
    for path, fi in paf_infos.items():
        if path.startswith("capabilities/visual/"):
            continue
        assert "_store" not in fi.import_names, f"{path}: leaked cross-package _store basename"


def test_visual_sibling__store_is_legitimate(paf_infos) -> None:
    """within visual/, `from ._store import _store` is a genuine symbol import (sibling
    module var) — it must resolve, proving we didn't over-broadly ban common names."""
    cap = paf_infos.get("capabilities/visual/_capability.py")
    assert cap is not None, "fixture missing capabilities/visual/_capability.py"
    assert "_store" in cap.import_names, "visual/_capability.py legitimately imports sibling _store var"


def test_store_no_longer_cross_module_hub(tmp_path_factory) -> None:
    """Behavioral symptom: visual/_store.py must NOT be a cross-module hub in the graph.

    Before the fix it scored 14.0 because ~15 files' `from ..._store import` all false-
    resolved to it by basename. After the fix only same-directory visual/ references
    remain, and cross-module weighting (used for get_repo_map) excludes those entirely.
    """
    import shutil

    from paf.capabilities.codebase_map._graph import build_edges
    from paf.capabilities.codebase_map._retrieval import _weighted_in_degree
    src = ROOT / "tests" / "fixtures" / "paf-src"
    if not src.exists():
        pytest.skip("paf-src fixture not present (gitignored)")
    dst = tmp_path_factory.mktemp("pafcopy") / "paf-src"
    shutil.copytree(src, dst)
    infos = []
    sources = {}
    for p in sorted(dst.rglob("*.py")):
        fi = analyze_file(p, dst)
        if fi is not None:
            infos.append(fi)
            sources[fi.path] = p.read_text(encoding="utf-8", errors="replace")
    w = _weighted_in_degree(build_edges(infos, sources))
    score = w.get("capabilities/visual/_store.py", 0.0)
    assert score < 4.0, (
        f"visual/_store.py cross-module hub score {score} — basename-collision edges still present"
    )


def test_ts_representative_modules_clean(cb_infos) -> None:
    """Real chatbot modules resolve only their true imports, never leaked path keywords."""
    assert cb_infos, "chatbot fixture produced no FileInfo"
    for path, fi in cb_infos.items():
        for name in fi.import_names:
            assert _IDENT_RE.match(name), f"{path}: impossible import name {name!r}"
            assert name not in _FORBIDDEN_IMPORT_TOKENS, f"{path}: keyword leaked: {name!r}"
    # The heavily-importing message.tsx must be clean (regex version leaked lib/db/from).
    cand = next(
        (fi for p, fi in cb_infos.items() if p.endswith("components/chat/message.tsx")),
        None,
    )
    if cand is not None:
        assert "MessageContent" in cand.import_names
        assert "cn" in cand.import_names
        for banned in ("lib", "db", "from", "import", "utils"):
            assert banned not in cand.import_names, f"{banned!r} leaked in message.tsx"
