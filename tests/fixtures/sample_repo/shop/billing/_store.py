"""Invoice records. `_store` is a real module variable, imported by ledger.py."""

from ..config import get_setting


class Invoice:
    """A billed amount."""

    def __init__(self, number: str, amount: float) -> None:
        self.number = number
        self.amount = amount
        self.currency = get_setting("currency")


_store: dict[str, Invoice] = {}


def record(invoice: Invoice) -> None:
    _store[invoice.number] = invoice
