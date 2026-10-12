"""Observability — structured traces to Langfuse via OTLP (no Logfire).

pydantic-ai emits OpenTelemetry natively. We configure an OTLP exporter pointed
at our self-hosted Langfuse and instrument pydantic-ai. Fully local.
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from paf.config import settings
from paf.observability.pricing import price_usage, start_price_refresh

try:
    from opentelemetry.sdk.trace import SpanProcessor
except ImportError:  # opentelemetry absent → tracing disabled, class unused
    SpanProcessor = object  # type: ignore[assignment,misc]

# Ref-count for observability lifecycle. A long-lived process creates a Session
# per lane, each constructor calling setup_observability() and each close()
# calling shutdown_observability(). We must configure the global OTel provider
# once and only truly shut it down once (last ref), otherwise per-lane sessions
# tear the provider down out from under everyone else (traces silently stop).
_setup_active: bool = False
_setup_refs: int = 0

# Cache-bust → Langfuse-error routing. ``WarnOnCacheBusts`` (a BUILTIN) fires
# ``CacheBustWarning`` (a ``UserWarning``) via stdlib ``warnings`` when a
# previously-established prompt-cache prefix collapses mid-run (moved prefix or
# provider-side expiry → re-charged tokens = billing spike). By default the
# runtime suppresses UserWarning and nothing reaches Langfuse. We install a
# ``warnings.showwarning`` hook that detects that SPECIFIC category and flips
# the active OTel span to ERROR (+ records it), so a bust shows up as a red
# error observation on the model-request span — while never aborting the turn.
_cache_bust_hook_installed: bool = False


def _install_cache_bust_error_hook() -> None:
    """Route a ``CacheBustWarning`` to an ERROR on the current OTel span.

    Idempotent / safe: wraps ``warnings.showwarning`` once, chains to the
    original handler (so the warning is still printed/logged via
    ``logging.captureWarnings(True)`` if that's on), and only touches the span
    for the ``CacheBustWarning`` category. Does NOT raise — the turn continues;
    this is an observe-and-flag signal, not a failure.
    """
    global _cache_bust_hook_installed
    if _cache_bust_hook_installed:
        return
    try:
        from pydantic_ai_harness.warn_on_cache_busts import CacheBustWarning
    except ImportError:  # pragma: no cover - harness not installed
        return

    import warnings as _warnings

    _orig = _warnings.showwarning

    def _hook(
        message: Any,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: Any = None,
    ) -> None:
        if issubclass(category, CacheBustWarning):
            _flag_current_span_error(message, filename, lineno)
        # Always chain: the warning is still surfaced the normal way
        # (stderr / logging if captureWarnings(True) is on).
        _orig(message, category, filename, lineno, file, line)

    _warnings.showwarning = _hook
    _cache_bust_hook_installed = True


def _flag_current_span_error(message: Any, filename: str, lineno: int) -> None:
    """Mark the active OTel span ERROR + record the bust, best-effort."""
    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return
        text = str(message)
        span.set_status(Status(StatusCode.ERROR, f"prompt cache bust: {text}"))
        # record_exception attaches the stack + message; the exception object
        # is just a vehicle — pass a plain constructed UserWarning so we don't
        # need the real instance (the warnings layer already consumed it).
        exc = UserWarning(f"CACHE_BUST {text.strip()} [{filename}:{lineno}]")
        span.record_exception(exc)
    except Exception:  # noqa: BLE001 - observability must never break the run
        pass


# Child → root rollup. Langfuse v4 is observation-centric: the root-observation
# view shows only what is on the ROOT span itself, and no longer rolls tokens,
# errors or tool calls up from descendants (v3's trace view did). So we
# aggregate them ourselves from ended child spans and stamp them on the root
# when it closes. Only traces opened via ``root_span`` are tracked, so spans of
# untracked traces never accumulate here.
_rollups: dict[int, _Rollup] = {}
_rollups_lock = threading.Lock()


class _Rollup:
    """Per-trace aggregate of what descendants of the root span did."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.cost = 0.0
        self.tools: Counter[str] = Counter()
        self.errors: list[str] = []
        self.compactions = 0
        self.compacted_tokens = 0

    def add(self, span: Any) -> None:
        attrs = span.attributes or {}
        op = attrs.get("gen_ai.operation.name")
        if op == "chat":
            n_in = int(attrs.get("gen_ai.usage.input_tokens") or 0)
            n_out = int(attrs.get("gen_ai.usage.output_tokens") or 0)
            n_cached = int(attrs.get("gen_ai.usage.cache_read.input_tokens") or 0)
            n_reason = int(attrs.get("gen_ai.usage.details.reasoning_tokens") or 0)
            self.input_tokens += n_in
            self.output_tokens += n_out
            self.cached_tokens += n_cached
            self.reasoning_tokens += n_reason
            # Same usage keys Langfuse prices a generation with.
            self.cost += price_usage(
                str(
                    attrs.get("gen_ai.response.model")
                    or attrs.get("gen_ai.request.model")
                    or ""
                ),
                {
                    "input": max(n_in - n_cached, 0),
                    "output": n_out,
                    "input_cached_tokens": n_cached,
                    "reasoning_tokens": n_reason,
                },
            )
        elif op == "execute_tool":
            self.tools[str(attrs.get("gen_ai.tool.name") or span.name)] += 1
        if span.name == "compact_messages":
            self.compactions += 1
            before = attrs.get("compaction.tokens_before")
            after = attrs.get("compaction.tokens_after")
            if isinstance(before, int) and isinstance(after, int):
                self.compacted_tokens += max(before - after, 0)
        status = span.status
        if status is not None and status.status_code.name == "ERROR":
            self.errors.append(f"{span.name}: {status.description or 'error'}")


class _RollupProcessor(SpanProcessor):
    """Feeds ended spans into the rollup of their trace, if it is tracked."""

    def on_end(self, span: Any) -> None:
        try:
            ctx = span.get_span_context()
            with _rollups_lock:
                rollup = _rollups.get(ctx.trace_id)
                if rollup is not None:
                    rollup.add(span)
        except Exception:  # noqa: BLE001 - observability must never break the run
            pass


def _apply_rollup(span: Any, rollup: _Rollup) -> None:
    """Stamp the aggregate onto the root span as Langfuse observation attributes."""
    total = rollup.input_tokens + rollup.output_tokens
    if total:
        # Match Langfuse's per-generation split: ``input`` EXCLUDES cached tokens.
        usage: dict[str, int] = {
            "input": max(rollup.input_tokens - rollup.cached_tokens, 0),
            "output": rollup.output_tokens,
            "total": total,
        }
        if rollup.cached_tokens:
            usage["input_cached_tokens"] = rollup.cached_tokens
        if rollup.reasoning_tokens:
            usage["output_reasoning_tokens"] = rollup.reasoning_tokens
        span.set_attribute("langfuse.observation.usage_details", _json_dumps(usage))
    if rollup.cost:
        span.set_attribute(
            "langfuse.observation.cost_details", _json_dumps({"total": rollup.cost})
        )
    if rollup.errors:
        span.set_attribute("langfuse.observation.level", "ERROR")
        span.set_attribute(
            "langfuse.observation.status_message",
            f"{len(rollup.errors)} error(s): " + "; ".join(rollup.errors[:3])[:500],
        )
    meta = "langfuse.observation.metadata."
    if rollup.tools:
        span.set_attribute(
            meta + "tool_calls",
            ", ".join(
                f"{n}×{c}" if c > 1 else n for n, c in rollup.tools.most_common()
            ),
        )
        span.set_attribute(meta + "tool_call_count", sum(rollup.tools.values()))
    if rollup.compactions:
        span.set_attribute(meta + "compactions", rollup.compactions)
        span.set_attribute(meta + "compacted_tokens", rollup.compacted_tokens)


@contextmanager
def root_span(
    name: str,
    tracer: Any | None = None,
    *,
    new_trace: bool = False,
    **attributes: Any,
) -> Iterator[Any]:
    """Open the trace's ROOT span and yield it, so the caller can attach Langfuse I/O.

    Langfuse's OTLP mapper reads ``langfuse.observation.input`` /
    ``langfuse.observation.output`` for the trace/observation I/O columns -- but
    pydantic-ai only emits ``gen_ai.input.messages`` / ``gen_ai.output.messages``
    (which Langfuse does NOT map), so without this root span every paf trace shows
    an EMPTY ``Input`` in the trace-list view. This is the documented "trace
    hygiene" fix (option #1, SDK-free): wrap the whole run in one root span that
    sets both attributes.

    ``name`` is the span name (convention ``paf.run.<conversation_id>``).
    ``tracer`` lets callers inject a specific tracer (used by tests to capture
    spans in-memory); defaults to the process-wide tracer. Attributes may be set
    either here (known before the run) or on the yielded span later (e.g.
    ``output`` once the settled result lands). No-op when tracing is disabled --
    returns a null span so callers need no branch.

    By default the span attaches to whatever span is CURRENT (so a turn run
    inside another span nests under it). ``new_trace=True`` starts it from an
    empty OTel context instead — a genuine root of a NEW trace, whatever is
    current. Background work that is not part of the turn that happened to
    trigger it (memory consolidation/review, thread summaries, notify turns)
    uses this: ``asyncio.create_task`` copies the current span into every task,
    so without it those spans nest inside an unrelated, often long-finished,
    conversation turn. Relate such traces with ``langfuse.session.id`` instead.
    """
    from opentelemetry import context as otel_context
    from opentelemetry import trace

    tracer = tracer or trace.get_tracer("paf")
    assert tracer is not None  # OTel SDK returns a no-op tracer, never None
    # An empty Context has no current span, so the new span roots a new trace.
    parent = otel_context.Context() if new_trace else None
    with tracer.start_as_current_span(name, context=parent) as span:
        tid = span.get_span_context().trace_id
        owns_rollup = False
        if span.is_recording():
            with _rollups_lock:
                # A nested root (subagent) shares its parent's trace and rollup.
                if tid not in _rollups:
                    _rollups[tid] = _Rollup()
                    owns_rollup = True
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
        try:
            yield span
        finally:
            if owns_rollup:
                with _rollups_lock:
                    rollup = _rollups.pop(tid)
                try:
                    _apply_rollup(span, rollup)
                except Exception:  # noqa: BLE001 - observability must never break the run
                    pass


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
            + base64.b64encode(
                f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
            ).decode()
        )
    }

    provider = TracerProvider(resource=Resource.create({"service.name": "paf"}))
    start_price_refresh(
        settings.otlp_endpoint,
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
    )
    # Registered BEFORE the exporter so rollups see every span as it ends.
    provider.add_span_processor(_RollupProcessor())
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
    # created and Langfuse receives nothing.
    from pydantic_ai import Agent

    Agent.instrument_all()

    # Route CacheBustWarning → ERROR on the active model-request span so a
    # prompt-cache collapse shows up as a red error observation in Langfuse
    # (it's suppressed by default otherwise). Idempotent; never aborts a turn.
    _install_cache_bust_error_hook()
    # The global provider can never be replaced (OTel rejects a second
    # set_tracer_provider), so it lives until process exit: shut it down then,
    # not when the last Session closes (see shutdown_observability).
    atexit.register(provider.shutdown)
    _setup_active = True
    _setup_refs += 1
    return True


def shutdown_observability() -> None:
    """Flush pending spans when a Session closes; never tear the provider down.

    **Ref-counted:** every ``setup_observability()`` (Session constructor) is
    balanced by one ``shutdown_observability()`` (Session.close), and each
    release force-flushes so a closing session's spans aren't lost.

    The provider itself is NOT shut down on the last release. OTel refuses to
    replace a global provider once set, so a process that closes its last
    Session and then opens another (``chat.py`` ``/resume``/``/new``, the SSE
    server after the idle sweep evicts every lane) would otherwise keep the
    shut-down provider as the global one and silently drop every later span —
    with an "Overriding of current TracerProvider is not allowed" warning on
    the re-setup. Shutdown happens once, at process exit (``atexit``, registered
    by ``setup_observability``); the final force-flush here covers short-lived
    processes (benchmarks) that call this explicitly before exiting.
    """
    global _setup_refs
    if not _setup_active or _setup_refs <= 0:
        return
    _setup_refs -= 1
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        force_flush = getattr(provider, "force_flush", None)
        if force_flush is not None:
            force_flush()
    except Exception:  # noqa: BLE001 - flush must never break close
        pass
