"""Resolved import names over the sample repo (``FileInfo.import_names``).

Regression for basename collisions: several packages have a module named
``_store.py``, and only ``shop/billing`` defines a module variable named
``_store`` (imported by its sibling ``ledger.py``). A bare-basename lookup once
resolved every ``from ._store import X`` to that variable, which leaked ``_store``
into unrelated files and made ``billing/_store.py`` look like a repo-wide hub.
"""

from __future__ import annotations

import re

from paf.capabilities.codebase_map._graph import build_edges
from paf.capabilities.codebase_map._retrieval import _weighted_in_degree

# Path segments and keywords that can never be a real imported binding.
_NOT_A_BINDING = {"from", "import", "as", "return", "def", "class", "export", "default"}
# A bare identifier or a dotted module target (e.g. `anyio.abc`).
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def test_every_import_name_is_a_plausible_binding(sample_py, sample_ts) -> None:
    infos = list(sample_py[0]) + list(sample_ts.values())
    assert infos
    for fi in infos:
        for name in fi.import_names:
            assert _IDENT.match(name), f"{fi.path}: impossible import name {name!r}"
            assert name not in _NOT_A_BINDING, f"{fi.path}: keyword leaked: {name!r}"


def test_a_module_basename_is_not_imported_across_packages(sample_py) -> None:
    by_path = {fi.path: fi for fi in sample_py[0]}
    assert set(by_path["shop/inventory/stock.py"].import_names) == {
        "Item",
        "get_setting",
    }
    for path, fi in by_path.items():
        if path.startswith("shop/billing/"):
            continue
        assert "_store" not in fi.import_names, f"{path}: leaked the _store basename"


def test_a_sibling_module_variable_import_still_resolves(sample_py) -> None:
    """Within billing/, `from ._store import _store` is a genuine symbol import."""
    ledger = next(fi for fi in sample_py[0] if fi.path == "shop/billing/ledger.py")
    assert {"Invoice", "_store"} <= set(ledger.import_names)


def test_a_shared_basename_does_not_make_a_hub(sample_py) -> None:
    """Cross-module weight goes to the real hub, never to billing/_store.py."""
    infos, sources = sample_py
    weights = _weighted_in_degree(build_edges(infos, sources))
    assert weights.get("shop/billing/_store.py", 0.0) == 0.0
    assert max(weights, key=weights.__getitem__) == "shop/config.py"


def test_ts_component_imports_are_the_bound_names_only(sample_ts) -> None:
    """Imports resolve to what they bind (`cn`), not path segments (`lib`, `utils`)."""
    message = sample_ts["web/components/message.tsx"]
    assert set(message.import_names) == {"MessageContent", "cn", "User"}
