"""Tracing survives a process closing its last Session and opening another.

Regression: shutdown_observability() used to shut the global TracerProvider
down on the last release. OTel never lets a provider be replaced, so the next
setup_observability() (chat.py /resume, or the SSE server after the idle sweep
evicted every lane) warned "Overriding of current TracerProvider is not
allowed" and every later span went to the dead provider.

Runs in a subprocess: setup installs a process-global provider and turns on
pydantic-ai instrumentation, which must not leak into the rest of the suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import logging
    from opentelemetry import trace
    from paf.observability.instrumentation import (
        setup_observability, shutdown_observability,
    )

    records = []
    logging.getLogger().addHandler(
        type("H", (logging.Handler,), {"emit": lambda self, r: records.append(r.getMessage())})()
    )

    assert setup_observability()
    provider = trace.get_tracer_provider()
    calls = []
    real_shutdown = provider.shutdown
    provider.shutdown = lambda: calls.append(1) or real_shutdown()

    shutdown_observability()          # last Session closes
    assert setup_observability()      # next Session opens
    assert trace.get_tracer_provider() is provider
    assert not calls, "provider was shut down on last release"
    assert not any("Overriding" in m for m in records), records
    shutdown_observability()
    print("OK")
    """
)


def test_reopen_after_last_close_keeps_tracing() -> None:
    env = {
        **os.environ,
        "PYTHONPATH": "",
        "LANGFUSE_PUBLIC_KEY": "pk-test",
        "LANGFUSE_SECRET_KEY": "sk-test",
        # Nothing listens here; flushes fail fast and are swallowed.
        "OTLP_ENDPOINT": "http://127.0.0.1:9/v1/traces",
    }
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "OK" in result.stdout
