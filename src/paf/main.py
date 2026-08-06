"""main.py — entrypoint / wiring.

Builds the Agent(s) from capabilities + config and exposes the runtime hooks
the CLI and other UIs import. Kept minimal; the root stays lean.
"""

from __future__ import annotations

from paf.observability.instrumentation import setup_observability


def init_runtime() -> None:
    """One-time runtime wiring: observability provided to agents."""
    setup_observability()
