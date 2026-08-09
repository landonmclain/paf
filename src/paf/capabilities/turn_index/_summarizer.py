"""Real LLM turn summarizer: a pydantic-ai structured-output Agent.

This is the piece that turns the one-line-per-turn index from a commodity into
something actually *searchable* and *recognizable* (M2's first feature; the
"turn index" read path keeps the one-liner in context and gates `read_turn`).

The motivation, from discussion (Landon 2026-08-08):

  * it must be a REAL pydantic-ai LLM call (not a heuristic), with structured
    output so the summary is a *record*, not a sentence;
  * the summary serves two masters -- in-context recognition AND searchability.
    A human-grade one-liner favors the first; the words used favor the second.
    We resolve it by splitting the record: a `sentence` for context + a
    `key_terms` list for search;
  * a "databases but not the type of database" failure mode is acceptable while
    the turn is fresh in context, but the *searchable* vocabulary must be there;
  * retries are needed on TWO axes that pydantic-ai gives you only one of:
      - schema/validation (pydantic's built-in `retries=`) -- malformed output,
      - LENGTH (token/char budget) -- pydantic's built-in retry resends the
        *same* prompt, so it can't make the output shorter. That loop is OURS.

Scope (decision 2026-08-08): stateless / order-independent -- NO carry-forward
of the prior summary. Each turn is summarized from its own messages only, so the
index stays composable, reorderable, and background-idempotent. (Carry-forward
is a possible later layer once this proves out.)

This module is `pydantic-ai` + stdlib only -- zero `paf` coupling -- so it stays
extractable alongside the rest of the `turn_index` subtree.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ._extract import build_turn_context

logger = logging.getLogger(__name__)

# The one thing that decides "searchable or not": how many searchable terms the
# model is allowed to list. Generous enough to carry the vocabulary, small enough
# to stay cheap in context. (These go to the index store, NOT the read-path
# injection by default -- see summary sentence below.)
MAX_KEY_TERMS = 6

# The one-line-in-context budget. Deterministic char cap, not token cap: LLMs are
# bad at self-estimating tokens but broadly OK at "N characters". Retried against.
DEFAULT_BUDGET_CHARS = 160


class TurnSummary(BaseModel):
    """Structured one-line-per-turn summary (searchable record, not a sentence).

    Fields mirror the two-service split:
      * `sentence` -- the human-grade one-liner that goes IN CONTEXT. Bounded by
        `budget_chars`. Must read naturally and let the model *recognize* the turn.
      * `key_terms` -- the SEARCH vocabulary the sentence may not fully carry (the
        "Postgres vs 'the database'" fix). What a future `search_index` /
        `conversation_search` would match against.
      * `outcome` -- a cheap categorical that makes the index filterable and is
        exactly the signal a model scanning a long run wants ("blocked" / "error").
    """

    sentence: str = Field(
        description=(
            'A natural one-line summary of what happened in this turn, no more than '
            f'{DEFAULT_BUDGET_CHARS} characters. Readable on its own, specific enough '
            'to recognize this turn in a list.'
        )
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description=(
            f'Up to {MAX_KEY_TERMS} concrete nouns/names/vocabulary from this turn that a '
            'future search would use -- real entity names (APIs, databases, tools, people, '
            'concepts, error codes). NOT generic words. These make the turn searchable even '
            '"when the sentence phrases it abstractly."'
        ),
    )
    outcome: Literal['resolved', 'blocked', 'error', 'decided', 'awaiting'] = Field(
        description=(
            "The turn's end state: 'resolved' it worked, 'blocked' hit a wall to revisit, "
            "'error' something failed, 'decided' a choice was made, 'awaiting' waiting on an "
            'external action/response.'
        )
    )


class TurnSummarizer:
    """A real pydantic-ai structured-output summarizer with a char-budget retry.

    Wraps an `Agent(..., output_type=TurnSummary, retries=...)`. Two retry layers:

      1. schema/validation -- pydantic-ai's builtin `retries=`, resends the SAME
         prompt on malformed/against-schema output. Cheap, free.
      2. LENGTH -- a loop WE own. If the model returns a `sentence` over
         `budget_chars`, the builtin retry is useless (same prompt -> same length),
         so we re-run with an explicit corrective system-prompt that states the
         overage and the cap. Bounded to `max_shorten_retries`. The retry prompt
         MUTATES the original prompt (via `output` context) -- pydantic-ai
         `Agent.run(output=...)`-style carry is not required; we just inject a
         one-shot corrective instruction and rebuild the model each retry.

    Concrete model is injectable (constructor `model=`), so the index can use a
    cheap/fast secondary model independent of the chat model. Defaults to the
    same OpenRouter DeepSeek family used for the main agent to stay zero-config.

    NOT a capability: a plain leaf helper, spec-declarable only in that
    `TurnIndexCapability.from_spec` passes a flat `summarizer:` dict straight to
    the constructor.
    """

    def __init__(
        self,
        model: str = 'openrouter:deepseek/deepseek-v4-flash-0731:nitro',
        *,
        budget_chars: int = DEFAULT_BUDGET_CHARS,
        max_key_terms: int = MAX_KEY_TERMS,
        retries: int = 2,
        max_shorten_retries: int = 2,
        max_output_tokens: int = 400,
    ) -> None:
        self.model = model
        self.budget_chars = budget_chars
        self.max_key_terms = max_key_terms
        self.max_shorten_retries = max_shorten_retries

        self._agent = Agent(
            model,
            output_type=TurnSummary,
            retries=retries,  # schema/validation retries (builtin)
            system_prompt=(
                "You write ONE-LINE summaries of a single agent turn for a searchable turn index. "
                f"Rules:\n"
                f"- `sentence`: a natural one-liner, HARD cap {budget_chars} characters, that lets a "
                f"human (or another model) instantly RECOGNIZE this turn in a list. Prefer concrete "
                f"verbs and real names over abstraction.\n"
                f"- `key_terms`: up to {max_key_terms} concrete searchable terms the sentence may not "
                f"fully carry -- real entity names, APIs, databases, tools, concepts, error codes. "
                f"'Postgres' not 'the database'; 'OAuth' not 'auth'. These exist so a keyword search "
                f"finds this turn even when the sentence phrases it abstractly.\n"
                f"- `outcome`: the end state -- resolved / blocked / error / decided / awaiting.\n"
                f"- Do NOT invent facts. Only summarize what is actually in the turn."
            ),
        )
        self._base_system = self._agent.system_prompt  # keep for the corrective retry

    @property
    def budget_chars(self) -> int:
        return self._budget_chars

    @budget_chars.setter
    def budget_chars(self, value: int) -> None:
        self._budget_chars = value

    async def summarize(
        self,
        context: str,
        *,
        max_output_tokens: int | None = None,
    ) -> TurnSummary:
        """Summarize a rendered turn context into a structured `TurnSummary`.

        `context` is the flat text from `build_turn_context(...)`. Runs the agent;
        if the resulting `sentence` is over budget, shortens it with up to
        `max_shorten_retries` corrective re-runs.
        """
        result = await self._agent.run(context)
        summary = result.output
        trimmed = 0
        while len(summary.sentence) > self._budget_chars and trimmed < self.max_shorten_retries:
            trimmed += 1
            logger.info(
                'turn summary %d chars over %d budget (attempt %d); shortening',
                len(summary.sentence), self._budget_chars, trimmed,
            )
            summary = await self._shorten(summary)
        # Ensure the contract holds for the fields we promised.
        summary.key_terms = summary.key_terms[: self.max_key_terms]
        if len(summary.sentence) > self._budget_chars:
            summary.sentence = summary.sentence[: self._budget_chars]  # hard floor, never store over-budget
        return summary

    async def _shorten(self, previous: TurnSummary) -> TurnSummary:
        """Re-run the summarizer demanding a shorter `sentence` than `previous`."""
        corrective = (
            f'Your previous summary was {len(previous.sentence)} characters, over the {self._budget_chars} '
            f'cap. Rewrite the SAME turn as a shorter one-liner (keep it specific), keeping the same '
            f'key_terms and outcome:\n\nPREVIOUS:\n{previous.sentence}'
        )
        # A fresh agent with an appended corrective instruction. `system_prompt`
        # here is the original + the corrective nudge, driving down length.
        agent = Agent(
            self.model,
            output_type=TurnSummary,
            retries=self.max_shorten_retries,
            system_prompt=f'{self._base_system}\n\n{corrective}',
        )
        return _extract_output(await agent.run(previous.sentence))


def _extract_output(result) -> TurnSummary:
    """Unwrap an `AgentRunResult` into its `.output` (the `TurnSummary`).

    `run()` returns an `AgentRunResult[T]`; `.output` is `T`. Kept as a single
    seam in case the result type changes across pydantic-ai versions.
    """
    return result.output


__all__ = [
    'DEFAULT_BUDGET_CHARS',
    'MAX_KEY_TERMS',
    'TurnSummarizer',
    'TurnSummary',
]