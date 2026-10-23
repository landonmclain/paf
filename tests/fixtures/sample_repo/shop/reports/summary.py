"""Report summaries. Uses OrderStore in type hints only: a reference, never a call."""

from ..orders._store import OrderStore
from .monthly import monthly_total
from .yearly import yearly_total


def describe(store: OrderStore) -> str:
    """Name the store being summarized."""
    return f"{monthly_total()} / {yearly_total()}"
