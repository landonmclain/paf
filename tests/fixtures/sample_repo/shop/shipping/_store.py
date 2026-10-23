"""Parcel records for the shipping package."""

from ..config import get_setting


class Parcel:
    """One parcel."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.region = get_setting("region")
