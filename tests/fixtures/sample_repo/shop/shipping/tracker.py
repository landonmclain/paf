"""Uses this package's own _store module. Imports Parcel only, never `_store`."""

from ..config import get_setting
from ._store import Parcel


def make(key: str) -> Parcel:
    if get_setting("strict") == "1" and not key:
        raise ValueError("key required")
    return Parcel(key)
