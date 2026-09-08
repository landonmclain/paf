"""Structured, off-path thread-beat summarization."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from paf.capabilities.thread_summarizer._capture import EvictionBlock


class Beat(BaseModel):
    """One chronological, receipt-backed event from an eviction block."""

    at: str = Field(description="The exact {run_id}:{turn_id} source ref.")
    what: str = Field(description="What happened in this block, without speculation.")
    why: str = Field(description="Why it mattered in the thread, only if grounded in the block.")


class BeatsRecord(BaseModel):
    """The durable structured output of one thread summarization pass."""

    turns_covered: list[str] = Field(default_factory=list)
    beats: list[Beat] = Field(default_factory=list)
    open: list[str] = Field(default_factory=list)
    resumed_from: str = ""


SummarizeBlock = Callable[[EvictionBlock], Awaitable[BeatsRecord]]


@dataclass(frozen=True)
class SummarizerPrompt:
    """Prompt material passed to the background summarizer agent."""

    block: EvictionBlock

    def render(self) -> str:
        messages = "\n".join(
            f"[{message.ref or 'unanchored'}] {message.role}: {message.text or ''}"
            for message in self.block.messages
        )
        return (
            "Summarize ONLY this evicted conversation block as chronological episodic beats. "
            "Do not infer facts or current state beyond these messages. Every beat must cite "
            "an exact source ref from the block. Return the requested structured BeatsRecord.\n\n"
            f"Block sequence: {self.block.seq}\n"
            f"Conversation: {self.block.conversation_id}\n"
            f"Messages:\n{messages}"
        )


async def summarize_block(block: EvictionBlock) -> BeatsRecord:
    """Run the real structured summarizer agent for one eviction block."""
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
    result = await agent.run(SummarizerPrompt(block).render())
    return result.output
