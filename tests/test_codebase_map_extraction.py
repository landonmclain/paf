"""Symbol extraction over the sample repo (``analyze_file``), Python and TypeScript."""

from __future__ import annotations

import pytest

from paf.capabilities.codebase_map._analyze import analyze_file


def test_every_sample_file_extracts_without_errors(sample_repo) -> None:
    """The analyzer must never raise on real code; a file may legitimately have no
    symbols (an ``__init__.py``) but must not crash."""
    errors = []
    for path in sorted(sample_repo.rglob("*")):
        if path.suffix not in (".py", ".ts", ".tsx"):
            continue
        try:
            analyze_file(path, sample_repo)
        except Exception as e:  # noqa: BLE001 - report every failure
            errors.append(f"{path.relative_to(sample_repo)}: {e!r}")
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize(
    ("path", "symbols"),
    [
        ("shop/config.py", {"get_setting", "region", "SETTINGS", "DEFAULT_REGION"}),
        ("shop/orders/_store.py", {"Order", "OrderStore", "save", "load"}),
        # A module-level variable sharing its module's name is still a symbol.
        ("shop/billing/_store.py", {"Invoice", "record", "_store"}),
        # Module-level `export const` assignments, not just functions.
        ("web/lib/db/schema.ts", {"user", "order"}),
        # A typed const (`nextConfig: NextConfig`) next to a plain one.
        ("web/next.config.ts", {"basePath", "nextConfig"}),
        ("web/components/message.tsx", {"Message"}),
        ("web/lib/utils.ts", {"cn"}),
    ],
)
def test_expected_symbols_are_extracted(sample_repo, path, symbols) -> None:
    fi = analyze_file(sample_repo / path, sample_repo)
    assert fi is not None, f"{path} not extracted"
    names = {s.name for s in fi.searchable_symbols}
    quals = {s.qualified for s in fi.searchable_symbols if s.qualified}
    missing = {
        s for s in symbols if s not in names and not any(q.endswith(s) for q in quals)
    }
    assert not missing, f"{path}: missing {sorted(missing)} (have {sorted(names)})"
