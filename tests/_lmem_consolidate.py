"""LongMemEval seed-time consolidation helpers (benchmark harness only).

These fold a *given* conversation into durable memory via the writable owner
agent, either in one pass (:func:`consolidate_conversation`) or in
token-bounded, message-boundary-aligned chunks (:func:`consolidate_conversation
_chunked`). They exist for the large-haystack benchmark path
(``tests/run_lmem.py``) — the live eviction-slice owner
(:func:`paf.capabilities.memory._consolidate.consolidate`) is a different, trim
call. Keeping them here, not in the production module, matches the rule that
agent model/settings/guidance come from specs and the harness constants stay
hardcoded (no env knobs).

The chunked variant matches the LIVE owner's shape — "current memory + a small
increment => merge" — repeated over bounded chunks instead of one giant pass.
The owner runs once per chunk against the SAME store; with ``inject_memory=True``
memory is re-rendered in-context each request, so chunk N sees the facts chunk
N-1 appended and merges rather than re-derives. Threads stay intact because
chunks are cut on MESSAGE boundaries, never mid-message. This preserves
single-mention thematic clusters that the one-shot path drops under output-budget
pressure (LongMemEval seeds up to ~443 messages).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic_ai.exceptions import (
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)

from paf.capabilities.memory._consolidate import (
    CONSOLIDATION_PROMPT,
    _build_owner,
    resident_consolidate_text,
)
from paf.degeneration_guard import run_guarded

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

logger = logging.getLogger(__name__)


async def consolidate_conversation(
    messages: list["ModelMessage"],
    *,
    store,
    agent_name: str,
    namespace: str = "",
    journal=None,
) -> None:
    """Fold a *full given* conversation into memory via the owner agent.

    Seed-time variant of :func:`paf.capabilities.memory._consolidate.consolidate`:
    it runs the owner over the ENTIRE ``messages`` list (no eviction-cutoff
    no-op) so a seeded haystack conversation can be consolidated into memory
    *before* a recall question, regardless of live sliding-window state (which
    never sees direct-seeded history). The call site is responsible for passing
    only pre-question content (honest-memory guard) — this function folds
    exactly what it is handed.

    ``journal`` (optional :class:`JournalStore`) switches the owner to the
    append-only model (``append_memory`` only); ``MEMORY.md`` is rendered from
    the journal at recall time.
    """
    owner, _owner_memory = _build_owner(
        store,
        agent_name=agent_name,
        namespace=namespace,
        journal=journal,
    )
    text = await resident_consolidate_text(messages)
    if not text:
        return
    result = await run_guarded(
        owner,
        CONSOLIDATION_PROMPT.format(evicted_text=text),
    )
    logger.info(
        "consolidated %d msgs (full conversation): %s",
        len(messages),
        result.output,
    )


# Rough chars-per-token used only to bound chunk size. paf has no tokenizer
# dependency; a chars/4 heuristic keeps every chunk comfortably within a single
# owner output budget without an exact token count (the budget is a heuristic
# bound on prompt size, not a hard token gate).
_CHARS_PER_TOKEN = 4

# Token budget per consolidation chunk (hardcoded — no env knob).
_CHUNK_MAX_TOKENS = 4000


def _split_chunks(text: str, *, max_tokens: int) -> list[str]:
    """Split rendered durable text into message-boundary aligned chunks.

    ``text`` is the ``resident_consolidate_text`` output: one ``User:`` /
    ``Assistant:`` line per message (content may itself span lines). Chunks are
    cut on whole-message boundaries only — we accumulate messages until adding
    the next would exceed the char budget (``max_tokens * _CHARS_PER_TOKEN``),
    then close the chunk. This preserves multi-turn threads (a theme that spans
    the boundary stays on one side or the predicted seams are handled by the
    owner's newest-wins merge) and never splits a single message.
    """
    if not max_tokens or max_tokens <= 0:
        return [text] if text else []
    budget = max_tokens * _CHARS_PER_TOKEN
    lines = text.split("\n")

    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0

    def _flush() -> None:
        nonlocal current, current_chars
        if current and any(line.strip() for line in current):
            chunks.append("\n".join(current))
        current = []
        current_chars = 0

    for line in lines:
        # A new message starts when the line begins a User/Assistant label:
        # ``User <ref> [<date>]: <text>`` (role + space + turn ref) or the
        # legacy ``User: <text>`` form. Lines inside a message's content do not
        # carry the label.
        is_msg_start = (
            line.startswith("User:")
            or line.startswith("User ")
            or line.startswith("Assistant:")
            or line.startswith("Assistant ")
        )
        line_chars = len(line) + 1  # +1 for the newline

        # If starting a new message would overflow the budget AND we already
        # have content in this chunk, close the chunk first. Empty lines with no
        # message yet are skipped.
        if is_msg_start and current and (current_chars + line_chars) > budget:
            _flush()

        current.append(line)
        current_chars += line_chars

    _flush()
    return chunks


async def consolidate_conversation_chunked(
    messages: list["ModelMessage"],
    *,
    store,
    agent_name: str,
    namespace: str,
    chunk_max_tokens: int | None = None,
    journal=None,
) -> None:
    """Consolidate a conversation into durable memory in token-bounded chunks.

    Same contract as :func:`consolidate_conversation` (renders via
    ``resident_consolidate_text``, persists through the sole writable owner)
    but splits the render into message-boundary chunks and runs the owner once
    per chunk against the SAME store, with memory injected in-context so each
    later chunk merges into the accumulated result instead of re-deriving.
    For large seeds this preserves single-mention thematic clusters that the
    one-shot path drops under output-budget pressure. The owner is built with
    ``inject_memory=True`` only for this path; the live/eviction ``consolidate``
    keeps its tool-only owner unchanged.

    ``journal`` (optional :class:`JournalStore`) switches the owner to the
    three-layer model: it appends immutable facts via ``append_memory`` (its
    ONLY tool) instead of in-place editing, and ``MEMORY.md`` is rendered from
    the journal later (at recall time), not here.
    """
    owner, _owner_memory = _build_owner(
        store,
        agent_name=agent_name,
        namespace=namespace,
        inject_memory=True,
        journal=journal,
    )
    text = await resident_consolidate_text(messages)
    if not text:
        return
    budget = chunk_max_tokens or _CHUNK_MAX_TOKENS
    chunks = _split_chunks(text, max_tokens=budget)
    for i, chunk in enumerate(chunks, start=1):
        try:
            result = await run_guarded(
                owner,
                CONSOLIDATION_PROMPT.format(evicted_text=chunk),
            )
        except (UsageLimitExceeded, UnexpectedModelBehavior, ModelHTTPError) as exc:
            # Mirror the recall-side resilience: a single owner failure must not
            # abort the whole benchmark run. Prior chunks are already written to
            # the store; log and stop the chunk loop so the run finishes with a
            # partial (but not crashed) consolidation.
            logger.warning(
                "consolidation stopped at chunk %d/%d after owner error "
                "(type=%s): %s. Prior chunks remain persisted.",
                i,
                len(chunks),
                type(exc).__name__,
                exc,
            )
            return
        logger.info(
            "consolidated chunk %d/%d (%d chars) of %d msgs: %s",
            i,
            len(chunks),
            len(chunk),
            len(messages),
            result.output,
        )
