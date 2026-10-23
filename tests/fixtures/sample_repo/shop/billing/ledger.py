"""Totals over recorded invoices."""

from ..config import get_setting
from ._store import Invoice, _store


def total(currency: str | None = None) -> float:
    wanted = currency or get_setting("currency")
    return sum(i.amount for i in _store.values() if i.currency == wanted)


def newest() -> Invoice | None:
    return next(reversed(list(_store.values())), None)
