"""storage — the Storage Protocol + swappable backends.

Core differentiator: pydantic-ai doesn't dictate this. Backends swap via config.
"""

from paf.storage.flatfile import FlatFileStorage
from paf.storage.protocol import Storage

__all__ = ["Storage", "FlatFileStorage"]
