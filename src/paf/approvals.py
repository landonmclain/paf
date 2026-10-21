"""Human approval of tool calls.

A guard (the gating policy's ``approve`` rules, InjectionGuard's post-flag
tightening, later the shell command rules) decides a call needs a person. If a
person is attached to the turn, the turn pauses on that one call: the request is
published into the turn's event stream, every attached front end can show it,
and the first decision wins. If nobody is attached, the call is refused, which
is the behaviour before approvals existed. So approvals can only make paf less
restrictive where a human said yes, never more permissive on its own.

Pieces:

- :class:`ApprovalChannel`: one per interactive turn. Holds the pending
  requests, publishes ``ApprovalRequested`` / ``ApprovalResolved`` events, applies
  the timeout, and writes the audit trail.
- The channel is bound to the running turn through a context variable
  (:func:`bind_channel`), so guards deep inside the agent run can ask
  :func:`approvals_available` without the Session being threaded through them.
- :class:`ApprovalBroker`: a capability answering pydantic-ai's
  ``handle_deferred_tool_calls`` hook. It turns each approval-required call into
  a request on the channel, waits, and returns approved / ``ToolDenied``. With no
  channel it denies everything at once.

Fails safe: a timeout denies; stopping the turn denies; a server restart drops
the pending call, and restart recovery never executes an unresolved tool call.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from opentelemetry import trace
from pydantic_ai import DeferredToolResults, ToolDenied
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext

DEFAULT_TIMEOUT_S = 600.0
"""How long a request waits for a person before it is denied (spec: approvals.timeout)."""

Outcome = Literal["approved", "denied", "timeout", "cancelled", "unattended"]

# Span names. Convention: paf.<domain>.<event>.
REQUEST_SPAN = "paf.approval.request"
DECISION_SPAN = "paf.approval.decision"
# Module-level so tests can point it at an in-memory exporter.
_TRACER = trace.get_tracer("paf")

_UNATTENDED = (
    "Tool '{tool}' needs human approval, and no one is attached to this turn to "
    "give it, so the call was refused before it executed. Do not retry it or work "
    "around it; tell the user what you wanted to do and why."
)
_DENIED = (
    "The user denied this call to '{tool}'{note}. It was not executed. Do not "
    "retry it or work around it; ask the user how they want to proceed."
)
_TIMEOUT = (
    "No one answered the approval request for '{tool}' within {minutes} "
    "minute(s), so it was not executed. Do not retry it; tell the user what you "
    "wanted to do."
)
_CANCELLED = (
    "The approval request for '{tool}' was cancelled because the turn was "
    "stopped. The call was not executed."
)


# --- data ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ApprovalRequest:
    """What a person is asked to decide. ``args`` is JSON text, secrets scrubbed."""

    id: str
    tool_call_id: str
    tool_name: str
    args: str
    reason: str
    preview: dict[str, Any] | None
    created_at: float
    expires_at: float
    conversation_id: str | None = None


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    outcome: Outcome
    note: str = ""
    actor: str = ""


@dataclass(frozen=True)
class ApprovalRequested:
    """Turn-stream event: a call is waiting for a decision."""

    request: ApprovalRequest


@dataclass(frozen=True)
class ApprovalResolved:
    """Turn-stream event: a request was settled (by a person, a timeout, a stop)."""

    approval_id: str
    tool_call_id: str
    tool_name: str
    outcome: Outcome
    actor: str = ""
    note: str = ""


# --- audit ----------------------------------------------------------------------------
class ApprovalAudit:
    """Append-only JSONL trail: one ``request`` and one ``decision`` record each."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, record: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError:  # an audit write must never break the turn
            pass


def _span(name: str, attributes: dict[str, Any]) -> None:
    try:
        with _TRACER.start_as_current_span(name) as span:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
    except Exception:  # noqa: BLE001 - observability must never fail the run
        pass


# --- the channel ------------------------------------------------------------------------
@dataclass
class _Pending:
    request: ApprovalRequest
    future: asyncio.Future[ApprovalDecision]


class ApprovalChannel:
    """The approval surface of ONE interactive turn."""

    def __init__(
        self,
        *,
        emit: Callable[[Any], None],
        timeout: float = DEFAULT_TIMEOUT_S,
        conversation_id: str | None = None,
        audit: ApprovalAudit | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._emit = emit
        self.timeout = timeout
        self.conversation_id = conversation_id
        self._audit = audit
        self._clock = clock
        self._pending: dict[str, _Pending] = {}
        # Why a guard asked for a call, keyed by tool call id (see `annotate`).
        self._notes: dict[str, dict[str, Any]] = {}

    # Guards attach their reason (and any preview) before asking, keyed by the
    # tool call id, so the broker can tell the person WHY.
    def annotate(
        self, tool_call_id: str, *, reason: str, preview: dict[str, Any] | None = None
    ) -> None:
        self._notes[tool_call_id] = {"reason": reason, "preview": preview}

    def pending(self) -> list[ApprovalRequest]:
        return [p.request for p in self._pending.values() if not p.future.done()]

    def resolve(
        self, approval_id: str, approved: bool, *, note: str = "", actor: str = "user"
    ) -> bool:
        """Settle a pending request. ``False`` if unknown or already settled
        (the first decision wins; a second client gets ``False``)."""
        pending = self._pending.get(approval_id)
        if pending is None or pending.future.done():
            return False
        pending.future.set_result(
            ApprovalDecision(
                approved=approved,
                outcome="approved" if approved else "denied",
                note=note,
                actor=actor,
            )
        )
        return True

    def cancel_all(self, reason: str = "turn stopped") -> None:
        """Deny everything still pending (the turn is stopping or ending)."""
        for pending in list(self._pending.values()):
            if not pending.future.done():
                pending.future.set_result(
                    ApprovalDecision(False, "cancelled", note=reason, actor="system")
                )

    def new_request(
        self, *, tool_call_id: str, tool_name: str, args: Any
    ) -> ApprovalRequest:
        now = self._clock()
        note = self._notes.pop(tool_call_id, {})
        return ApprovalRequest(
            id=uuid.uuid4().hex[:12],
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args=_args_text(args),
            reason=note.get("reason") or "a guard requires approval for this tool",
            preview=note.get("preview"),
            created_at=now,
            expires_at=now + self.timeout,
            conversation_id=self.conversation_id,
        )

    async def ask(self, requests: list[ApprovalRequest]) -> dict[str, ApprovalDecision]:
        """Publish ``requests``, wait for decisions (or the timeout), return them."""
        loop = asyncio.get_running_loop()
        for req in requests:
            self._pending[req.id] = _Pending(req, loop.create_future())
            self._record("request", req)
            _span(
                REQUEST_SPAN,
                {
                    "approval_id": req.id,
                    "tool": req.tool_name,
                    "reason": req.reason,
                    "conversation_id": req.conversation_id,
                },
            )
            self._emit(ApprovalRequested(req))
        futures = [self._pending[r.id].future for r in requests]
        decisions: dict[str, ApprovalDecision] = {}
        try:
            await asyncio.wait(futures, timeout=self.timeout)
        except asyncio.CancelledError:
            self.cancel_all("turn aborted")
            raise
        finally:
            for req in requests:
                fut = self._pending.pop(req.id).future
                if fut.done():
                    decision = fut.result()
                else:
                    fut.cancel()
                    decision = ApprovalDecision(False, "timeout", actor="system")
                decisions[req.id] = decision
                self._settled(req, decision)
        return decisions

    def _settled(self, req: ApprovalRequest, decision: ApprovalDecision) -> None:
        latency_ms = round((self._clock() - req.created_at) * 1000)
        self._record("decision", req, decision=decision, latency_ms=latency_ms)
        _span(
            DECISION_SPAN,
            {
                "approval_id": req.id,
                "tool": req.tool_name,
                "outcome": decision.outcome,
                "actor": decision.actor,
                "latency_ms": latency_ms,
            },
        )
        self._emit(
            ApprovalResolved(
                approval_id=req.id,
                tool_call_id=req.tool_call_id,
                tool_name=req.tool_name,
                outcome=decision.outcome,
                actor=decision.actor,
                note=decision.note,
            )
        )

    def _record(self, event: str, req: ApprovalRequest, **extra: Any) -> None:
        if self._audit is None:
            return
        record: dict[str, Any] = {"event": event, "ts": self._clock(), **asdict(req)}
        if "decision" in extra:
            record["decision"] = asdict(extra.pop("decision"))
        record.update(extra)
        self._audit.write(record)


def _args_text(args: Any) -> str:
    # Imported here: this module is imported by the guards themselves, so it
    # must not pull in the capability package at import time.
    from paf.capabilities.guardrails.secret_handles import scrub_secrets

    if isinstance(args, str):
        text = args
    else:
        text = json.dumps(args, default=str, ensure_ascii=False)
    return scrub_secrets(text)


# --- binding a channel to the running turn ------------------------------------------------
_CURRENT: ContextVar[ApprovalChannel | None] = ContextVar("paf_approvals", default=None)


def current_channel() -> ApprovalChannel | None:
    return _CURRENT.get()


def approvals_available() -> bool:
    """Is a person attached to the running turn to answer an approval request?"""
    return _CURRENT.get() is not None


@contextmanager
def bind_channel(channel: ApprovalChannel | None) -> Iterator[None]:
    token = _CURRENT.set(channel)
    try:
        yield
    finally:
        try:
            _CURRENT.reset(token)
        except ValueError:  # closed from another context (generator GC): harmless
            pass


# --- the capability -------------------------------------------------------------------------
@dataclass
class ApprovalBroker(AbstractCapability[AgentDepsT]):
    """Answers pydantic-ai's deferred-tool hook by asking the attached person.

    Always installed: it never approves anything on its own. Without a channel
    (no one attached) every approval-required call is denied with a message the
    model can act on, which keeps a run from ending in an unhandled deferred
    request.
    """

    minutes_label: Callable[[float], str] = field(
        default=lambda seconds: f"{max(1, round(seconds / 60))}", repr=False
    )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    async def handle_deferred_tool_calls(
        self, ctx: RunContext[AgentDepsT], *, requests: Any
    ) -> DeferredToolResults | None:
        calls = list(requests.approvals)
        if not calls:
            return None
        channel = current_channel()
        if channel is None:
            return DeferredToolResults(
                approvals={
                    c.tool_call_id: ToolDenied(_UNATTENDED.format(tool=c.tool_name))
                    for c in calls
                }
            )
        reqs = [
            channel.new_request(
                tool_call_id=c.tool_call_id, tool_name=c.tool_name, args=c.args
            )
            for c in calls
        ]
        decisions = await channel.ask(reqs)
        results: dict[str, Any] = {}
        for req in reqs:
            decision = decisions[req.id]
            if decision.approved:
                results[req.tool_call_id] = True
            else:
                results[req.tool_call_id] = ToolDenied(
                    self._denial(req, decision, channel.timeout)
                )
        return DeferredToolResults(approvals=results)

    def _denial(
        self, req: ApprovalRequest, decision: ApprovalDecision, timeout: float
    ) -> str:
        if decision.outcome == "timeout":
            return _TIMEOUT.format(
                tool=req.tool_name, minutes=self.minutes_label(timeout)
            )
        if decision.outcome == "cancelled":
            return _CANCELLED.format(tool=req.tool_name)
        note = f" with the note: {decision.note!r}" if decision.note else ""
        return _DENIED.format(tool=req.tool_name, note=note)
