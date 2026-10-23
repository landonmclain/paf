"""Order persistence."""

from ..config import get_setting


class Order:
    """One customer order."""

    def __init__(self, order_id: str, total: float) -> None:
        self.order_id = order_id
        self.total = total


class OrderStore:
    """Keeps orders in memory."""

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}

    def save(self, order: Order) -> None:
        self._orders[order.order_id] = order

    def load(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def currency(self) -> str:
        return get_setting("currency")
