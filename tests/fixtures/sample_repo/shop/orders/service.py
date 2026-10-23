"""Placing orders."""

from ..config import get_setting
from ._store import Order, OrderStore


def place_order(store: OrderStore, order_id: str, subtotal: float) -> Order:
    """Price an order with tax and save it."""
    rate = float(get_setting("tax_rate", "0"))
    order = Order(order_id, subtotal * (1 + rate))
    store.save(order)
    return order
