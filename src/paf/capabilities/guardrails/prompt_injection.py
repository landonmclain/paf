"""paf-owned prompt-injection observability policy.

The harness ``PromptInjectionDefender`` is the shell; paf owns the wiring here
(the settled don't-pin stance). The builtin classifies tool results and, with
``block_high_risk=True``, withholds a risky result by replacing it with a
catchable ``blocked_message`` ``ToolReturn`` — the model reads it, adapts, and
the run continues. That is the "catchable error" half of the requirement.

The gap the builtin leaves is the *trace*: it withholds silently, with no span
flowing to observability. This module is the paf-owned ``on_detection`` handler
that closes that gap. It emits an OpenTelemetry span (``paf.guardrail.
prompt_injection``) carrying the tool name, risk level, and the verdict's
diagnostics, so a blocked injection is visible in Langfuse with the reason.

The handler is deliberately non-raising: the builtin's contract is that raising
in ``on_detection`` fails the run, which is the opposite of the catchable-error
requirement. A trace must never break the run, so the span emission is wrapped
in a try/except that swallows observability failures.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext
from stackone_defender import DefenseResult

# Span name for a prompt-injection detection. Convention: paf.<domain>.<event>.
_SPAN_NAME = "paf.guardrail.prompt_injection"

# The verdict fields worth surfacing in the trace. Kept to plain-JSON scalars so
# the span attributes stay small and Langfuse-friendly.
_DIAGNOSTIC_FIELDS = (
    "blocked",
    "risk_level",
    "detections",
    "fields_sanitized",
    "tier2_score",
    "latency_ms",
)


def _diagnostics(verdict: DefenseResult) -> dict[str, Any]:
    """A plain-JSON summary of a verdict, safe for span attributes."""
    return {
        "blocked": not verdict.allowed,
        "risk_level": verdict.risk_level,
        "detections": list(verdict.detections),
        "fields_sanitized": list(verdict.fields_sanitized),
        "tier2_score": verdict.tier2_score,
        "latency_ms": verdict.latency_ms,
    }


def prompt_injection_on_detection(
    ctx: RunContext[Any],
    call: ToolCallPart,
    verdict: DefenseResult,
) -> None:
    """Emit a trace span for a flagged prompt-injection verdict.

    Called by the builtin ``PromptInjectionDefender`` for every flagged verdict
    (a detection, a sanitization, a rejection, or an escalated risk level). The
    span carries the tool name, the risk level, and the verdict diagnostics so
    the decision is observable in Langfuse.

    Non-raising by design: a trace must never fail the run. Any observability
    failure is swallowed.
    """
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("paf")
        with tracer.start_as_current_span(_SPAN_NAME) as span:
            span.set_attribute("tool", call.tool_name)
            span.set_attribute("risk_level", verdict.risk_level)
            for key, value in _diagnostics(verdict).items():
                span.set_attribute(f"verdict.{key}", value)
    except Exception:  # noqa: BLE001 - a trace must never break the run
        pass
