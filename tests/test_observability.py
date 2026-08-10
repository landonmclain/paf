"""Trace-hygiene verification (Portfolio-GitHub/Turn-Index-Testing#Trace hygiene).

The bug: pydantic-ai emits only ``gen_ai.input.messages`` / ``gen_ai.output.messages``
on its model spans, which Langfuse's OTLP mapper does NOT read for the trace
Input/Output columns -- so every paf trace showed an EMPTY Input in the
trace-list view. The fix (option #1, SDK-free): a root span ``paf.run.<cid>`` in
``runtime.stream`` sets ``langfuse.observation.input`` (known up front) and
``langfuse.observation.output`` (once the settled result lands, before the span
closes).

We verify the *mechanism* here with the SDK's canonical in-memory exporter -- no
network, no keys, no real model, and no fighting the process-global
TracerProvider (which OTel only lets you set once). If the root span carries both
Langfuse attributes, then any real run through ``Session.stream`` will too (it
uses the same helper via the process tracer).
"""
from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from paf.observability.instrumentation import root_span


@pytest.fixture()
def captured() -> InMemorySpanExporter:
    return InMemorySpanExporter()


def _tracer(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test")


def _attrs(span) -> dict:
    return dict(span.attributes)


def test_root_span_sets_langfuse_input_and_output(captured) -> None:
    """The root span carries both Langfuse I/O attrs when opened + when output set."""
    with root_span(
        "paf.run.cli",
        _tracer(captured),
        **({"langfuse.observation.input": {"prompt": "hi", "conversation_id": "cli"}}),
    ) as span:
        span.set_attribute("langfuse.observation.output", '{"output": "hello"}')
    captured.force_flush()

    spans = captured.get_finished_spans()
    assert len(spans) == 1, f"expected 1 root span, got {len(spans)}"
    span = spans[0]
    attrs = _attrs(span)
    assert span.name == "paf.run.cli"
    assert "langfuse.observation.input" in attrs, attrs
    assert "langfuse.observation.output" in attrs, attrs
    # Non-empty string values so Langfuse's mapper can decode them.
    assert "hi" in attrs["langfuse.observation.input"]
    assert "hello" in attrs["langfuse.observation.output"]


def test_root_span_values_are_json_serialized(captured) -> None:
    """Dict input is JSON-serialized (not a bare repr) so Langfuse can decode."""
    with root_span(
        "paf.run.x",
        _tracer(captured),
        **({"langfuse.observation.input": {"prompt": "remember", "conversation_id": "x"}}),
    ):
        pass
    captured.force_flush()
    attrs = _attrs(captured.get_finished_spans()[0])
    decoded = json.loads(attrs["langfuse.observation.input"])
    assert decoded == {"prompt": "remember", "conversation_id": "x"}, decoded


def test_root_span_graceful_under_noop_tracer() -> None:
    """root_span must not error if the process tracer is the default no-op.

    The default tracer provider defines no matching SpanProcessor -- the context
    manager still enters/exits cleanly. runtime.stream depends on this not raising.
    """
    from opentelemetry import trace

    with root_span("paf.run.off", trace.get_tracer("paf.noop")) as span:
        span.set_attribute("langfuse.observation.output", "{}")
    # No assertion beyond "didn't raise"; a no-op span keeps no real attributes.


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-q", "-s", __file__]))
