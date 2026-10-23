"""Monthly figures."""

from .yearly import yearly_total


def monthly_total() -> float:
    return yearly_total() / 12


def monthly_label() -> str:
    return f"month: {monthly_total():.2f}"
