"""Structured, off-path thread-beat summarization over a conversation turn span."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from paf.capabilities._turns import user_anchor_indices


class Beat(BaseModel):
    """One chronological, receipt-backed event from the summarized span."""

    at: str = Field(description="The exact {run_id}:{turn_id} source ref.")
    what: str = Field(description="What happened in this span, without speculation.")
    why: str = Field(description="Why it mattered in the thread, only if grounded in the span.")


class BeatsRecord(BaseModel):
    """The durable structured output of one thread summarization pass."""

    turns_covered: list[str] = Field(default_factory=list)
    beats: list[Beat] = Field(default_factory=list)
    open: list[str] = Field(default_factory=list)
    resumed_from: str = ""


SummarizeSpan = Callable[[list, str], Awaitable[BeatsRecord]]


def _ref_for(messages: list, idx: int) -> str | None:
    """``{run_id}:{turn_id}`` ref for ``messages[idx]`` (greatest user anchor <= idx)."""
    msg = messages[idx]
    run_id = getattr(msg, "run_id", None)
    if run_id is None:
        return None
    anchors = user_anchor_indices(messages)
    tid = next((a for a in range(len(anchors) - 1, -1, -1) if anchors[a] <= idx), None)
    return f"{run_id}:{tid}" if tid is not None else None


@dataclass(frozen=True)
class SummarizerPrompt:
    """Prompt material passed to the background summarizer agent.

    Input is the CONVERSATION TURN SPAN (a list of persisted ``ModelMessage``),
    rendered via the curated extraction — user prompts + turn-closing final
    answers only, with thinking / mid-step tool narration dropped (see
    ``paf.capabilities._window._first_text``). The store is the lossless source
    of truth, so it does not matter whether a turn is still in context: whatever
    is unsummarized since the watermark gets summarized here.
    """

    span: list
    conversation_id: str | None

    def render(self) -> str:
        # NOTE: iterate the CURATED text once so the prompt stays bounded — a
        # tool-heavy turn collapses from ~15KB of narration to a short user +
        # final-answer pair. Non-durable messages (thinking/intermediate/tool
        # returns) render as nothing and are skipped for the model.
        from paf.capabilities._window import _first_text

        lines: list[str] = []
        for i, msg in enumerate(self.span):
            text = _first_text(msg)
            if not text:
                continue  # thinking / tool narration / retry — not durable
            ref = _ref_for(self.span, i)
            role = type(msg).__name__
            lines.append(f"[{ref or 'unanchored'}] {role}: {text}")
        messages = "\n".join(lines)
        return (
            "Summarize this conversation turn-span as chronological episodic beats. "
            "Do not infer facts or current state beyond these messages. Every beat must cite "
            "an exact source ref from the span. Return the requested structured BeatsRecord.\n\n"
            f"Conversation: {self.conversation_id}\n"
            f"Messages:\n{messages}"
        )


async def summarize_span(span: list, conversation_id: str | None) -> BeatsRecord:
    """Run the real structured summarizer agent over a conversation turn span."""
    from pydantic_ai import Agent

    from paf.capabilities.memory._consolidate import (
        _consolidation_model_settings,
        consolidate_model,
    )

    agent = Agent(
        consolidate_model(),
        name="thread-summarizer",
        output_type=BeatsRecord,
        model_settings=_consolidation_model_settings(),
        retries=2,
    )
    result = await agent.run(SummarizerPrompt(span, conversation_id).render())
    return result.output
