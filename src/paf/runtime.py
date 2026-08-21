"""paf.runtime — a UI-agnostic agent session.

Owns the full run lifecycle: build the agent from its spec, enable tracing,
run one or more turns (optionally streaming), and flush spans before the
process exits. So callers stop re-implementing

    setup_observability() → build_agent() → agent.run() → shutdown_observability()

and instead get one object that does all four. The async methods are the core;
``run_sync`` is a thin convenience wrapper for scripts, tests, and CLIs that
don't need a running event loop.

The session is a context manager: exiting it always flushes buffered OTel spans
(via ``shutdown_observability``), so a short-lived process can't silently drop
traces the way a bare ``agent.run_sync()`` can.

Streaming and turn-accumulation mirror ``ui/cli_demo.py``: the run is driven by
``agent.run_stream_events`` (the pydantic-ai streaming API), ``TextPartDelta``
events yield the answer token-by-token, and the settled ``AgentRunResult`` is the
autoritative transcript — ``.output`` is the final answer, ``.new_messages()`` is
exactly this turn's messages to append to running history for cross-turn
persistence (``StepPersistence`` keys it via ``conversation_id``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai import (
    Agent,
    AgentRunResult,
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
)
from pydantic_ai.messages import (
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai_harness.step_persistence import continue_run

from paf.capabilities.persistence import _build_store
from paf.observability.instrumentation import (
    _json_dumps,
    root_span,
    setup_observability,
    shutdown_observability,
)
from paf.specs import build_agent


@dataclass(frozen=True)
class StreamEvent:
    """One renderable event from an agent run.

    UI-agnostic: a caller (REPL, web socket, TTS pipeline) maps these onto
    whatever it shows. ``kind`` is one of:

    - ``"text"``      a token chunk of the model's answer (``text``)
    - ``"thinking"``  a reasoning token (suppress or render faintly)
    - ``"tool_call"`` the model invoked a tool (``tool_name``, ``args``)
    - ``"tool_result"`` a tool returned (``tool_name``, ``text``)

    The settled ``AgentRunResult`` lands on ``self.last_result`` when the run
    completes, and ``self.history`` accumulates the turn's messages — callers
    never need to reconstruct the transcript from these events.
    """

    kind: Literal["text", "thinking", "tool_call", "tool_result"]
    text: str = ""
    tool_name: str = ""
    args: str = ""


class Session:
    """An agent session: build once, run many turns, flush on exit.

    Parameters
    ----------
    agent_name:
        Spec base name to build (`"assistant"` → `specs/assistant.yaml`).
    profile:
        Capability allowlist for this agent kind, resolved to
        ``custom_capability_types``. Defaults to ``ASSISTANT_PROFILE``. Pass
        ``DEMO_PROFILE`` for the lean demo/live-testing agent
        (``specs/demo_assistant.yaml``).
    conversation_id:
        Stable id for cross-turn history. ``StepPersistence`` keys off this, so
        two turns with the same id share stored history. Defaults to ``None``
        (the agent derives a per-run id unless persistence is configured).
    trace:
        Enable Langfuse/OTel tracing. Disable for tests or offline runs.

    The session is reusable: call ``stream``/``run``/``run_sync`` repeatedly,
    then ``close()`` (or ``with``) once when done to flush spans.
    """

    def __init__(
        self,
        agent_name: str = "assistant",
        *,
        profile=None,
        conversation_id: str | None = None,
        trace: bool = True,
        resume: bool = True,
        resume_window: int | None = None,
        store=None,
    ) -> None:
        self.agent_name = agent_name
        self.profile = profile
        self.conversation_id = conversation_id
        self.trace = trace
        # ``resume``: seed ``self.history`` from durable storage on the first run
        # (continue_run over the shared flat-file store) so a fresh process picks
        # up where the prior session left off. M1 semantics: continue_run returns
        # the full snapshot, compaction summary included -- no suppression here.
        # Set ``False`` for a genuinely fresh conversation (e.g. tests, probe runs
        # that must NOT inherit prior history).
        self.resume = resume
        # ``resume_window``: cap the number of most-recent messages loaded back
        # into context on resume. ``None`` (default) loads the full snapshot;
        # an int loads only the last N messages and drops older history from the
        # live context (recoverable via the journal / conversation search).
        self.resume_window = resume_window
        self._resumed = False
        self._restored = False
        # Default to the canonical shared flat-file store (same directory
        # build_persistence()/cli use). Injectable for tests / custom roots.
        self._store = store if store is not None else _build_store()
        if trace:
            setup_observability()
        self._agent: Agent[Any, Any] = build_agent(agent_name, profile=profile)
        # Running message history for cross-turn persistence. Populated by
        # ``_ensure_resumed`` from durable storage on the first run (if resume)
        # and then accumulated per-turn.
        self.history: list = []

    # -- introspection -------------------------------------------------------

    @property
    def agent(self) -> Agent[Any, Any]:
        """The built agent."""
        return self._agent

    @property
    def name(self) -> str:
        return self._agent.name or self.agent_name

    # -- resume --------------------------------------------------------------

    @property
    def restored_count(self) -> int:
        """Number of prior messages loaded by ``continue_run`` resume (0 if none)."""
        return len(self.history) if self._restored else 0

    async def restore(self, *, conversation_id: str | None = None) -> int:
        """Load prior history from the store now, returning ``restored_count``.

        Explicit form of the auto-resume — call it before the first run (e.g. to
        show "resumed N prior messages" in a banner). Same semantics as
        ``_ensure_resumed``: ``continue_run`` full snapshot (compaction summary
        included), idempotent, safe to call early or never.
        """
        cid = conversation_id if conversation_id is not None else self.conversation_id
        await self._ensure_resumed(cid)
        return self.restored_count

    async def refresh(self, *, conversation_id: str | None = None) -> int:
        """Re-read prior history from the store, discarding the cached copy.

        Unlike :meth:`restore` — which is one-shot per session (guarded by the
        ``_resumed`` flag) — this forces a fresh ``continue_run`` so a *page
        reload* shows turns that were persisted after the session first resumed.
        Returns the new ``restored_count`` (0 if none).
        """
        cid = conversation_id if conversation_id is not None else self.conversation_id
        self._resumed = False
        self._restored = False
        self.history = []
        await self._ensure_resumed(cid)
        return self.restored_count

    async def _ensure_resumed(self, conversation_id: str | None) -> None:
        """Seed ``self.history`` from the store exactly once (before the first run).

        ``continue_run`` loads the latest settled snapshot for the conversation's
        most recent run and returns it as a message history — M1 semantics, full
        snapshot (compaction summary included). When ``resume_window`` is set, the
        history is trimmed to the last N messages so a fresh process does not load
        the entire conversation into context (older turns stay recoverable via the
        journal / ``search_conversation_history``). Idempotent: only acts when
        resume is enabled, the conversation has a stable id, and this is the first
        run. A conversation with no prior runs stays empty (``restored_count == 0``).
        """
        if self._resumed or not self.resume or conversation_id is None:
            self._resumed = True
            return
        self._resumed = True
        try:
            runs = await self._store.list_runs(conversation_id=conversation_id)
            if not runs:
                return
            # Resume the latest *continuable* run. The last listed run may be an
            # interrupted/in-flight one whose snapshot never settled (e.g. the
            # process was killed mid-turn), which `continue_run` refuses with a
            # LookupError. Falling straight through to it would return EMPTY
            # history even though earlier runs have valid snapshots — so walk
            # backward to the newest run with a continuable snapshot.
            history = []
            for run in reversed(runs):
                try:
                    history = await continue_run(self._store, run_id=run.run_id)
                    break
                except Exception:  # noqa: BLE001 - skip unsettled runs
                    continue
            if not history:
                return
            if self.resume_window is not None and self.resume_window >= 0:
                history = history[-self.resume_window:] if self.resume_window else []
            self.history = history
            self._restored = True
        except Exception:  # noqa: BLE001 - resume is sugar; never break the run
            self._restored = False

    # -- async core ----------------------------------------------------------

    async def stream(
        self,
        prompt: str,
        *,
        conversation_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run one turn, yielding renderable :class:`StreamEvent` objects.

        Driven by ``agent.run_stream_events`` (the official pydantic-ai
        streaming API, as in ``ui/cli_demo.py``). Emits a token for every
        ``TextPartDelta`` (``kind="text"``), a reasoning token for each
        ``ThinkingPartDelta`` (``kind="thinking"``), and a ``tool_call`` /
        ``tool_result`` pair whenever the agent invokes a tool.

        On completion (``AgentRunResultEvent``) the settled result lands on
        ``self.last_result`` and the turn's messages are appended to
        ``self.history`` — the two things a caller needs after the stream ends.

        The chosen ``conversation_id`` (explicit > session default) is threaded
        through so persistent storage restores prior history.
        """
        cid = conversation_id if conversation_id is not None else self.conversation_id
        await self._ensure_resumed(cid)

        result: AgentRunResult | None = None
        # Trace hygiene: open ONE root span for the whole turn so Langfuse's
        # trace-list view gets a non-empty Input. pydantic-ai only emits
        # gen_ai.input.messages/output.messages, which Langfuse does not map --
        # we set langfuse.observation.input/output here (SDK-free; see
        # instrumentation.root_span). Input is known up front; output is set once
        # the settled AgentRunResult lands, before the span closes.
        with root_span(
            f"paf.run.{cid or 'default'}",
            **({"langfuse.observation.input": _run_input(prompt, cid)}),
        ) as _span:
            async with self._agent.run_stream_events(
                prompt,
                message_history=self.history,
                conversation_id=cid,
            ) as event_stream:
                async for event in event_stream:
                    if isinstance(event, PartStartEvent):
                        # First chunk of a part arrives in .part; emit it or the
                        # leading text is silently dropped (cli_demo tags this).
                        # Tag by part kind, not by content, so a thinking part's
                        # opening token isn't mistaken for the answer (the model can
                        # stream the first reasoning token before switching over).
                        part = event.part
                        delta = _part_leading_text(part)
                        if delta:
                            kind = "thinking" if isinstance(part, ThinkingPart) else "text"
                            yield StreamEvent(kind=kind, text=delta)
                    elif isinstance(event, PartDeltaEvent):
                        if isinstance(event.delta, TextPartDelta):
                            yield StreamEvent(kind="text", text=event.delta.content_delta)
                        elif isinstance(event.delta, ThinkingPartDelta):
                            yield StreamEvent(
                                kind="thinking", text=event.delta.content_delta
                            )
                        # ToolCallPartDelta: args stream in; the full args are shown
                        # on the FunctionToolCallEvent line instead (as cli_demo does).
                    elif isinstance(event, FunctionToolCallEvent):
                        yield StreamEvent(
                            kind="tool_call",
                            tool_name=str(event.part.tool_name or ""),
                            args=(
                                event.part.args
                                if isinstance(event.part.args, str)
                                else str(event.part.args)
                            ),
                        )
                    elif isinstance(event, FunctionToolResultEvent):
                        yield StreamEvent(
                            kind="tool_result",
                            tool_name=str(event.part.tool_name or ""),
                            text=(
                                event.part.content
                                if isinstance(event.part.content, str)
                                else str(event.part.content or "")
                            ),
                        )
                    elif isinstance(event, AgentRunResultEvent):
                        result = event.result
                        # Set the trace output NOW, while the root span is still
                        # open -- the settled answer is the only complete output.
                        _span.set_attribute("langfuse.observation.output", _json_dumps(_run_output(result)))
        assert result is not None, "run_stream_events ended before AgentRunResultEvent"
        self.last_result = result
        # ACKNOWLEDGE (cli_demo): result.new_messages() is ONLY this turn; older
        # history lives in self.history. Accumulate, don't overwrite.
        self.history = [*self.history, *result.new_messages()]
    async def run(
        self,
        prompt: str,
        *,
        conversation_id: str | None = None,
    ) -> str:
        """Run one turn and return the complete final text (async).

        Thin convenience over :meth:`stream` that keeps only the answer text
        (``kind == "text"``) and discards tool/thinking events.
        """
        parts: list[str] = []
        async for event in self.stream(prompt, conversation_id=conversation_id):
            if event.kind == "text":
                parts.append(event.text)
        return "".join(parts)

    # -- thin sync convenience -----------------------------------------------

    def run_sync(self, prompt: str, *, conversation_id: str | None = None) -> str:
        """Run one turn synchronously; returns the final text.

        Thin wrapper for scripts/tests that don't manage their own loop: creates
        a short-lived loop, runs the async core inside it, returns the result.
        """
        return asyncio.run(self.run(prompt, conversation_id=conversation_id))

    # -- observability lifecycle ---------------------------------------------

    def close(self) -> None:
        """Flush buffered spans and disable tracing before process exit."""
        if self.trace:
            shutdown_observability()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _part_leading_text(part: Any) -> str:
    """Pull the initial text out of a stream part's ``.content`` field.

    Mirrors cli_demo's ``_stream_part_content``: the first chunk of a part sits
    in ``.content`` (a ``str`` or list of ``str``), later chunks arrive as
    ``TextPartDelta.content_delta``. Non-text parts return ``""``.
    """
    content = getattr(part, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(chunk) for chunk in content)
    return ""


def _run_input(prompt: str, cid: str | None) -> dict:
    """Serialize a turn's input for the Langfuse trace root span.

    ``langfuse.observation.input`` is what Langfuse's OTLP mapper shows in the
    trace-list Input column. That column is the *user's* input for that turn --
    the current prompt -- NOT the accumulated history. pydantic-ai's own
    `gen_ai.input.messages` on the agent/generation spans already carries full
    per-message context (and reads coherently when you click into the span), so
    dumping the whole transcript here would just bloat the list view and
    duplicate what's a click away. cid is included for at-a-glance grouping.
    """
    return {"prompt": prompt, "conversation_id": cid}


def _run_output(result: Any) -> dict:
    """Serialize a turn's settled output for the Langfuse trace root span."""
    return {"output": result.output if result is not None else None}
