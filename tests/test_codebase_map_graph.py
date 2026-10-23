"""Import and call edges over the sample repo (``build_edges``).

An edge links a file to a file whose symbols it uses. With source text, edges
where a symbol is actually called are upgraded from ``reference`` to ``call``;
type hints and other mentions stay references.
"""

from __future__ import annotations

import pytest

from paf.capabilities.codebase_map._graph import build_edges


@pytest.fixture(scope="module")
def edges(sample_py) -> dict[tuple[str, str], dict]:
    infos, sources = sample_py
    return {(e["source"], e["target"]): e for e in build_edges(infos, sources)}


def test_build_edges_is_deterministic(sample_py) -> None:
    infos, sources = sample_py
    assert build_edges(infos, sources) == build_edges(infos, sources)


def test_without_sources_every_edge_is_a_reference(sample_py) -> None:
    infos, _ = sample_py
    refs = build_edges(infos)
    assert refs and all(e["kind"] == "reference" for e in refs)
    assert all(e["call_symbols"] == [] for e in refs)


@pytest.mark.parametrize(
    ("source", "target", "symbols", "calls"),
    [
        # A cross-package call into the hub.
        ("shop/orders/service.py", "shop/config.py", {"get_setting"}, {"get_setting"}),
        ("shop/billing/ledger.py", "shop/config.py", {"get_setting"}, {"get_setting"}),
        # A sibling call: `Order(...)` is called, `OrderStore` only annotates.
        (
            "shop/orders/service.py",
            "shop/orders/_store.py",
            {"Order", "OrderStore"},
            {"Order"},
        ),
        ("shop/inventory/stock.py", "shop/inventory/_store.py", {"Item"}, {"Item"}),
        (
            "shop/reports/summary.py",
            "shop/reports/monthly.py",
            {"monthly_total"},
            {"monthly_total"},
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_called_symbols_make_a_call_edge(edges, source, target, symbols, calls) -> None:
    edge = edges.get((source, target))
    assert edge is not None, f"missing edge {source} -> {target}"
    assert edge["kind"] == "call"
    assert symbols <= set(edge["symbols"])
    assert calls <= set(edge["call_symbols"])


def test_a_type_hint_only_use_stays_a_reference(edges) -> None:
    """summary.py names OrderStore only in an annotation: never upgraded to a call."""
    edge = edges[("shop/reports/summary.py", "shop/orders/_store.py")]
    assert edge["kind"] == "reference"
    assert "OrderStore" in edge["symbols"]
    assert edge["call_symbols"] == []


def test_a_relative_store_import_resolves_to_its_own_package(edges) -> None:
    """`from ._store import X` must link to this package's _store.py, not another's."""
    for source, own in [
        ("shop/orders/service.py", "shop/orders/_store.py"),
        ("shop/inventory/stock.py", "shop/inventory/_store.py"),
        ("shop/shipping/tracker.py", "shop/shipping/_store.py"),
        ("shop/billing/ledger.py", "shop/billing/_store.py"),
    ]:
        assert (source, own) in edges, f"{source} must link to {own}"
        others = {
            t
            for (s, t) in edges
            if s == source and t.endswith("/_store.py") and t != own
        }
        assert not others, f"{source} resolved ._store to another package: {others}"
