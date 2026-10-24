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

from paf.observability.instrumentation import _json_dumps, root_span


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
        **(
            {
                "langfuse.observation.input": {
                    "prompt": "remember",
                    "conversation_id": "x",
                }
            }
        ),
    ):
        pass
    captured.force_flush()
    attrs = _attrs(captured.get_finished_spans()[0])
    decoded = json.loads(attrs["langfuse.observation.input"])
    assert decoded == {"prompt": "remember", "conversation_id": "x"}, decoded


def test_root_span_non_observation_values_are_set_raw(captured) -> None:
    """Attributes other than langfuse.observation.* must be stored RAW.

    Regression guard: ``langfuse.session.id`` (and any other non-observation
    attribute) must reach the span as the exact string given -- NOT wrapped in
    JSON quotes. If ``root_span`` JSON-encodes every attribute, a session id
    ``beam-lite-abc`` is stored as ``"beam-lite-abc"`` (with embedded quote
    chars), so Langfuse stores that quoted string as the session key and
    ``traces?sessionId=beam-lite-abc`` returns nothing -- silently breaking
    session grouping. Only the ``langfuse.observation.*`` keys feed Langfuse's
    structured I/O parser and need JSON encoding.
    """
    with root_span(
        "paf.run.session-regression",
        _tracer(captured),
        **{
            "langfuse.session.id": "beam-lite-20260810-110122",
            "langfuse.observation.input": {"prompt": "hi", "conversation_id": "x"},
        },
    ):
        pass
    captured.force_flush()
    attrs = _attrs(captured.get_finished_spans()[0])

    # Session id must be the EXACT raw string (no JSON quotes, no escaping).
    assert attrs["langfuse.session.id"] == "beam-lite-20260810-110122", attrs
    # While observation input is still JSON-serialized for Langfuse's parser.
    assert json.loads(attrs["langfuse.observation.input"]) == {
        "prompt": "hi",
        "conversation_id": "x",
    }


def test_output_attribute_must_be_json_string_not_raw_dict(captured) -> None:
    """The run output goes on the span as a JSON *string* (what Langfuse reads).

    ``runtime.stream`` sets ``langfuse.observation.output`` with
    ``_json_dumps(_run_output(result))``: the dict is serialized first. Older
    OpenTelemetry SDKs silently dropped a raw dict attribute (Langfuse then showed
    a BLANK Output column while Input, serialized in ``root_span``, still worked);
    newer ones keep it as a mapping. Either way Langfuse needs the string, so this
    pins the string contract without depending on what OTel does with a dict.
    """
    with root_span(
        "paf.run.out-regression",
        _tracer(captured),
        **(json.loads(_json_dumps({"prompt": "p", "conversation_id": "c"}))),
    ) as span:
        span.set_attribute(
            "langfuse.observation.output", _json_dumps({"output": "apple"})
        )
    captured.force_flush()
    attrs = _attrs(captured.get_finished_spans()[0])

    saved = attrs["langfuse.observation.output"]
    assert isinstance(saved, str), type(saved)
    assert json.loads(saved) == {"output": "apple"}, saved


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
