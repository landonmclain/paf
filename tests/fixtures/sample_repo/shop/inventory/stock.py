"""Uses this package's own _store module. Imports Item only, never `_store`."""

from ..config import get_setting
from ._store import Item


def make(key: str) -> Item:
    if get_setting("strict") == "1" and not key:
        raise ValueError("key required")
    return Item(key)
