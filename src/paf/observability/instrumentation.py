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
    exit). Safe to call unconditionally: no-ops when keys are absent, so the
    package also works with tracing off (e.g. in tests or offline).
    """
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return False

    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", settings.otlp_endpoint)
    # Langfuse expects Basic auth via the public/secret key pair.
    os.environ.setdefault("OTEL_EXPORTER_OTLP_HEADERS", "")

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:  # opentelemetry not installed → tracing disabled
        return False

    headers = {
        "Authorization": (
            "Basic "
            + __import__("base64").b64encode(
                f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
            ).decode()
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
