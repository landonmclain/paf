"""main.py — entrypoint / wiring.

Builds the Agent(s) from specs + capabilities and exposes the runtime hooks
the CLI and other UIs import. Kept minimal; the root stays lean.
"""

from __future__ import annotations

from paf.config import ASSISTANT_SPEC, settings
from paf.observability.instrumentation import setup_observability
from paf.storage import Persistence, build_persistence


def init_runtime() -> None:
    """One-time runtime wiring: observability provided to agents."""
    setup_observability()


def build_assistant(*, database: str | None = None) -> Persistence:
    """Build the assistant agent from its spec, with persistence attached.

    ``database`` defaults to ``settings.db_path`` (PROJECT_ROOT-anchored), so
    callers get one canonical DB unless they explicitly override it.
    """
    return build_persistence(
        ASSISTANT_SPEC,
        database=database or settings.db_path,
    )
