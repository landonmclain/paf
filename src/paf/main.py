"""main.py — entrypoint / wiring.

Builds the Agent(s) from capabilities + config and exposes the runtime hooks
the CLI and other UIs import. Kept minimal; the root stays lean.
"""

from __future__ import annotations

from paf.config import settings
from paf.observability.instrumentation import setup_observability
from paf.storage import FlatFileStorage
from paf.storage.protocol import Storage


def init_runtime() -> None:
    """One-time runtime wiring: observability + storage provided to agents."""
    setup_observability()


def get_storage() -> Storage:
    """Return the configured storage backend (default: flat-file/JSON)."""
    return FlatFileStorage(settings.storage_root)
