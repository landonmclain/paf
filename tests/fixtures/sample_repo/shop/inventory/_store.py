"""Item records for the inventory package."""

from ..config import get_setting


class Item:
    """One item."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.region = get_setting("region")
