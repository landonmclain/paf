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
from typing import Any

from pydantic_ai import (
    Agent,
    AgentRunResult,
    AgentRunResultEvent,
    PartDeltaEvent,
    PartStartEvent,
)
from pydantic_ai.messages import TextPartDelta

from paf.observability.instrumentation import (
    setup_observability,
    shutdown_observability,
)
from paf.specs import build_agent


class Session:
    """An agent session: build once, run many turns, flush on exit.

    Parameters
    ----------
    agent_name:
        Spec base name to build (``"assistant"`` → ``specs/assistant.yaml``).
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
        conversation_id: str | None = None,
        trace: bool = True,
    ) -> None:
        self.agent_name = agent_name
        self.conversation_id = conversation_id
        self.trace = trace
        if trace:
            setup_observability()
        self._agent: Agent[Any, Any] = build_agent(agent_name)
        # Running message history for cross-turn persistence. Callers may seed
        # this (e.g. resumed via StepPersistence) before the first run.
        self.history: list = []

    # -- introspection -------------------------------------------------------

    @property
    def agent(self) -> Agent[Any, Any]:
        """The built agent."""
        return self._agent

    @property
    def name(self) -> str:
        return self._agent.name or self.agent_name

    # -- async core ----------------------------------------------------------

    async def stream(
        self,
        prompt: str,
        *,
        conversation_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Run one turn, yielding the answer's text deltas as they arrive.

        Mirrors ``ui/cli_demo.py``'s event loop: ``run_stream_events`` feed text
        deltas (``TextPartDelta``) out of the model. Hand a caller a simple
        string-stream to forward to a TTY/renderer; the full settled result
        lands on ``self.last_result`` when the turn completes.

        The chosen ``conversation_id`` (explicit > session default) is threaded
        through so persistent storage restores prior history. The turn's new
        messages are appended to ``self.history``.
        """
        cid = conversation_id if conversation_id is not None else self.conversation_id

        result: AgentRunResult | None = None
        async with self._agent.run_stream_events(
            prompt,
            message_history=self.history,
            conversation_id=cid,
        ) as event_stream:
            async for event in event_stream:
                if isinstance(event, PartStartEvent):
                    # First chunk of a part arrives in .part; emit it or the
                    # leading text is silently dropped (cli_demo timestamps this).
                    delta = _part_leading_text(event.part)
                    if delta:
                        yield delta
                elif isinstance(event, PartDeltaEvent) and isinstance(
                    event.delta, TextPartDelta
                ):
                    yield event.delta.content_delta
                elif isinstance(event, AgentRunResultEvent):
                    result = event.result

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
        """Run one turn and return the complete final text (async)."""
        parts: list[str] = []
        async for text in self.stream(prompt, conversation_id=conversation_id):
            parts.append(text)
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
