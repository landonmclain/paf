"""Observability — structured traces to Langfuse via OTLP (no Logfire).

pydantic-ai emits OpenTelemetry natively. We configure an OTLP exporter pointed
at our self-hosted Langfuse and instrument pydantic-ai. Fully local.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from paf.config import settings

# Ref-count for observability lifecycle. A long-lived process creates a Session
# per lane, each constructor calling setup_observability() and each close()
# calling shutdown_observability(). We must configure the global OTel provider
# once and only truly shut it down once (last ref), otherwise per-lane sessions
# tear the provider down out from under everyone else (traces silently stop).
_setup_active: bool = False
_setup_refs: int = 0


@contextmanager
def root_span(name: str, tracer: Any | None = None, **attributes: Any) -> Iterator[Any]:
    """Open the trace's ROOT span and yield it, so the caller can attach Langfuse I/O.

    Langfuse's OTLP mapper reads ``langfuse.observation.input`` /
    ``langfuse.observation.output`` for the trace/observation I/O columns -- but
    pydantic-ai only emits ``gen_ai.input.messages`` / ``gen_ai.output.messages``
    (which Langfuse does NOT map), so without this root span every paf trace shows
    an EMPTY ``Input`` in the trace-list view. This is the documented "trace
    hygiene" fix (option #1, SDK-free): wrap the whole run in one root span that
    sets both attributes. See Portfolio-GitHub/Turn-Index-Testing#Trace hygiene.

    ``name`` is the span name (convention ``paf.run.<conversation_id>``).
    ``tracer`` lets callers inject a specific tracer (used by tests to capture
    spans in-memory); defaults to the process-wide tracer. Attributes may be set
    either here (known before the run) or on the yielded span later (e.g.
    ``output`` once the settled result lands). No-op when tracing is disabled --
    returns a null span so callers need no branch.
    """
    from opentelemetry import trace

    tracer = tracer or trace.get_tracer("paf")
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            # Only the langfuse.observation.* keys feed Langfuse's structured
            # I/O parser and need JSON encoding. All OTHER attributes (notably
            # langfuse.session.id) must be set as RAW strings -- JSON-encoding
            # a session id would store '"foo"' (with embedded quotes) and break
            # downstream filters/grouping on that exact value.
            if key.startswith("langfuse.observation."):
                span.set_attribute(key, _json_dumps(value))
            else:
                span.set_attribute(key, value)
        yield span


def _json_dumps(value: Any) -> str:
    """Serialize span attribute values the way Langfuse's OTel mapper expects.

    Attribute values must be JSON-encodable strings; dicts/list-scalars are sent
    as their JSON text so Langfuse's ``langfuse.observation.*`` parser can decode
    them back into structured input/output rather than a bare repr.
    """
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def setup_observability() -> bool:
    """Enable tracing if Langfuse keys + an endpoint are configured.

    Returns True if tracing was enabled (so callers know whether to flush on
    exit). Safe to call unconditionally: no-ops when any of the keys or the
    endpoint are absent (all come from .env), so the package also works with
    tracing off (e.g. in tests or offline).

    **Idempotent / ref-counted:** a long-lived process (web server) constructs
    a ``Session`` per lane and each constructor calls this. We must configure
    the OTel provider EXACTLY ONCE — ``trace.set_tracer_provider`` rejects an
    override and the exporter would otherwise be torn down by other sessions'
    ``shutdown``. Subsequent calls bump a refcount and return their earlier
    result without touching the provider.
    """
    global _setup_active, _setup_refs
    if (
        not settings.langfuse_public_key
        or not settings.langfuse_secret_key
        or not settings.otlp_endpoint
    ):
        return False

    if _setup_active:
        _setup_refs += 1
        return True

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
    _setup_active = True
    _setup_refs += 1
    return True


def shutdown_observability() -> None:
    """Flush + shut down the tracer so short-lived processes don't lose spans.

    **Ref-counted:** every ``setup_observability()`` (Session constructor) must
    be balanced by one ``shutdown_observability()`` (Session.close). We only
    call ``provider.shutdown()`` on the LAST release — a per-lane Session.close
    must flush but must never tear down the global provider that other
    concurrent sessions are still tracing into. On a long-lived server the
    last ref (process shutdown) is what flushes+shuts down.
    """
    global _setup_active, _setup_refs
    if not _setup_active or _setup_refs <= 0:
        return
    _setup_refs -= 1
    if _setup_refs > 0:
        # Other sessions still active — DO NOT shut the provider down. Just
        # force pending spans to flush so THIS session's spans aren't lost on
        # idle eviction.
        try:
            from opentelemetry import trace

            provider = trace.get_tracer_provider()
            force_flush = getattr(provider, "force_flush", None)
            if force_flush is not None:
                force_flush()
        except Exception:  # noqa: BLE001 - flush must never break close
            pass
        return
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    shutdown = getattr(provider, "shutdown", None)
    if shutdown is not None:
        shutdown()
    _setup_active = False
