"""The ONE execution path for a subagent task (sync and background alike).

``run_subagent`` wraps the child run in a ``paf.subagent.<cid>.<n>`` span,
applies the leaf's own timeout + usage budget, walks the task record through
``running -> done|error|cancelled``, writes the output file, and publishes each
transition to the per-agent process feed. Sync and background delegations both
call it, so tracing and persistence cannot drift between the two modes.

The span nests under whatever span is current: the parent's ``delegate_task``
tool span for a sync call, or — for a background call — the parent turn's trace
the worker was spawned from (``BackgroundRunner.spawn`` keeps the OTel context
and resets only the harness run scope). The span also carries
``parent_run_id`` / ``conversation_id`` attributes as an explicit link.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from typing import Any

from paf.background_processes._events import (
    ProcessEvent,
    ProcessEventType,
    get_agent_emitter,
)
from paf.observability.instrumentation import root_span

from ._notify import excerpt
from ._roster import Leaf
from ._store import SubagentTaskStore, feed_view

log = logging.getLogger(__name__)


async def emit(store: SubagentTaskStore, rec: dict[str, Any]) -> None:
    """Publish one task transition to the conversation-lane process feed."""
    status, detail = feed_view(rec)
    try:
        await get_agent_emitter(store.agent).emit_process(
            ProcessEvent(
                id=str(rec["task_id"]),
                label=f"subagent: {rec.get('agent')}",
                status=ProcessEventType(status),
                detail=detail,
                lane="conversation",
                scope=store.conversation_id,
            )
        )
    except Exception:  # the feed is best-effort; never fail a task over it
        log.exception("subagent feed emit failed for %s", rec.get("task_id"))


def _trace_id(span: Any) -> str | None:
    """The span's trace id as 32-hex, or None when tracing is off."""
    try:
        ctx = span.get_span_context()
    except AttributeError:
        return None
    return f"{ctx.trace_id:032x}" if getattr(ctx, "is_valid", False) else None


def _json_safe(value: Any) -> Any:
    """Coerce to JSON-native types (OpenRouter reports cost as ``Decimal``)."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    """Plain, JSON-safe dict of a pydantic-ai ``RunUsage`` for the JSONL record."""
    if usage is None or not is_dataclass(usage) or isinstance(usage, type):
        return None
    return _json_safe(
        {k: v for k, v in asdict(usage).items() if v not in (None, 0, {})}
    )


async def run_subagent(
    rec: dict[str, Any], leaf: Leaf, store: SubagentTaskStore
) -> tuple[dict[str, Any], str | None]:
    """Run one task to a terminal state; return ``(final_record, output)``.

    ``output`` is the leaf's full text on success, else ``None`` (the record's
    ``detail`` says why). Failures never raise — except cancellation, which is
    recorded and then re-raised so the awaiting task actually cancels.
    """
    task_id = str(rec["task_id"])
    with root_span(
        f"paf.{task_id}",  # task ids are already "subagent.<cid>.<n>"
        agent=leaf.name,
        task_id=task_id,
        parent_run_id=rec.get("parent_run_id") or "",
        conversation_id=store.conversation_id or "",
        **{"langfuse.observation.input": rec.get("task") or ""},
    ) as span:
        running = store.update(task_id, status="running", trace_id=_trace_id(span))
        await emit(store, running)
        try:
            result = await asyncio.wait_for(
                leaf.agent.run(rec["task"], usage_limits=leaf.policy.usage_limits),
                leaf.policy.timeout_seconds,
            )
        except asyncio.CancelledError:
            final = store.update(task_id, status="cancelled", detail="cancelled")
            await emit(store, final)
            raise
        except TimeoutError:
            detail = f"timed out after {leaf.policy.timeout_seconds:g}s"
            final = store.update(task_id, status="error", detail=detail)
            await emit(store, final)
            return final, None
        except Exception as exc:  # usage limit, model failure, tool crash, ...
            log.warning("subagent %s failed", task_id, exc_info=exc)
            final = store.update(
                task_id, status="error", detail=f"{type(exc).__name__}: {exc}"
            )
            await emit(store, final)
            return final, None
        output = str(result.output)
        # langfuse.observation.* values must be JSON text (root_span only
        # encodes the attributes it is opened with).
        span.set_attribute(
            "langfuse.observation.output", json.dumps(excerpt(output, task_id))
        )
        try:
            final = store.mark_done(
                task_id, output=output, usage=_usage_dict(result.usage)
            )
        except Exception as exc:  # persisting must not turn a result into a crash
            log.exception("subagent %s: saving output failed", task_id)
            final = store.update(
                task_id, status="error", detail=f"saving output failed: {exc}"
            )
            await emit(store, final)
            return final, output
        await emit(store, final)
        return final, output
