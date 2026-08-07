"""Observability — structured traces to Langfuse via OTLP (no Logfire).

pydantic-ai emits OpenTelemetry natively. We configure an OTLP exporter pointed
at our self-hosted Langfuse and instrument pydantic-ai. Fully local.
"""

from __future__ import annotations

import os

from paf.config import settings


def setup_observability() -> bool:
    """Enable tracing if Langfuse keys + an endpoint are configured.

    Returns True if tracing was enabled (so callers know whether to flush on
    exit). Safe to call unconditionally: no-ops when any of the keys or the
    endpoint are absent (all come from .env), so the package also works with
    tracing off (e.g. in tests or offline).
    """
    if (
        not settings.langfuse_public_key
        or not settings.langfuse_secret_key
        or not settings.otlp_endpoint
    ):
        return False

    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", settings.otlp_endpoint)
    # NOTE: do NOT set OTEL_EXPORTER_OTLP_HEADERS here. An empty-string value
    # makes the OTLP exporter override/drop the Authorization header we pass
    # via `headers=` below, so Langfuse silently rejects every POST (traces
    # never land). Auth is carried on the exporter itself.

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:  # opentelemetry not installed → tracing disabled
        return False

    headers = {
        "Authorization": (
            "Basic "
            + __import__("base64")
            .b64encode(
                f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
            )
            .decode()
        )
    }

    provider = TracerProvider(resource=Resource.create({"service.name": "paf"}))
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=settings.otlp_endpoint, headers=headers)
        )
    )
    # Set the global provider so pydantic-ai's OTel instrumentation picks it up.
    from opentelemetry import trace

    trace.set_tracer_provider(provider)

    # REQUIRED: pydantic-ai does NOT emit spans by default (Agent.instrument is
    # off unless enabled). Without this, no agent/metadata/model spans are ever
    # created and Langfuse receives nothing. See pydantic docs "OTel without
    # Logfire" and langfuse.com/docs/integrations/pydantic-ai (Step 3).
    from pydantic_ai import Agent

    Agent.instrument_all()
    return True


def shutdown_observability() -> None:
    """Flush + shut down the tracer so short-lived processes don't lose spans.

    Call before process exit (e.g. at the end of a CLI run).
    """
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    shutdown = getattr(provider, "shutdown", None)
    if shutdown is not None:
        shutdown()
