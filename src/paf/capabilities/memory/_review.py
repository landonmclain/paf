"""Memory reviewer — periodic, off-band curation pass.

Mirrors the owner (``_consolidate.py``) but ONE layer up: the owner **captures**
durable facts into the journal; the reviewer **curates** what's already there.
It is cadence-fired (separate review watermark, like the owner's
``consolidate_every``) and curates the **whole rendered memory**, not just
pending suggestions:

- resolves the assistant's pending ``memory_review`` proposals (approve /
  reject / re-tier), AND
- proactively archives stale/wrong facts and changes tiers on its own.

Its read input is the **rendered view only** (``read_render``) — it never reads
the journal or index directly, addressing facts by their stable ``#N`` journal
line id (the same address the assistant and the sidecar use). It writes
verdicts via ``memory_review`` into the same append-only ``memory-overrides.jsonl``
sidecar the assistant writes to; the render pass applies them deterministically.

Structure mirrors ``_consolidate.py``: a ``Memory`` capability configured as the
REVIEWER (``overrides`` set + ``reviewer=True`` -> ``ReviewerToolset`` exposing
ONLY ``memory_review``), wrapped in an ``Agent``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from ._consolidate import _consolidation_model_settings, consolidate_model

if TYPE_CHECKING:
    from ._capability import Memory
logger = logging.getLogger(__name__)


def review_model() -> str:
    """The reviewer model — same as the owner's by decision."""
    return consolidate_model()


def _build_reviewer_memory(
    store,
    overrides,
    *,
    agent_name: str,
    namespace: str,
) -> Memory:
    """Build the reviewer's Memory capability (curation-only, no append).

    ``overrides`` set + ``reviewer=True`` => ``get_toolset`` returns
    ``ReviewerToolset`` exposing ONLY ``memory_review``. The reviewer has no
    ``append_memory`` (adding facts is the owner's job) and no journal read.
    Extracted so the invariants are testable without constructing an ``Agent``.
    """
    from ._capability import Memory  # deferred to avoid circular import

    return Memory(
        store=store,
        agent_name=agent_name,
        namespace=namespace,
        inject_memory=False,  # reads the render via read_render, not injection
        guidance="",  # single reviewer prompt is REVIEW_PROMPT (the user message)
        writable=False,
        expose_tools=True,  # carries the reviewer toolset (memory_review)
        overrides=overrides,
        reviewer=True,
    )


def _build_reviewer(
    store,
    overrides,
    *,
    agent_name: str,
    namespace: str,
):
    """Construct the reviewer agent (curator: memory_review only)."""
    reviewer_memory = _build_reviewer_memory(
        store,
        overrides,
        agent_name=agent_name,
        namespace=namespace,
    )
    reviewer = Agent(
        review_model(),
        name="memory-reviewer",
        capabilities=[reviewer_memory],
        model_settings=_consolidation_model_settings(),
        retries=2,
    )
    return reviewer, reviewer_memory


REVIEW_PROMPT = """\
You are the memory reviewer. You curate this user's durable memory — you do NOT
add new facts (that's the owner's job). Your input is the current rendered
memory below, tiered (## t1 / ## t2 / ## t3), one fact per line, each ending in
its journal line id (#N). Some lines carry a `*(archive pending review)*`
marker — those are assistant suggestions awaiting your verdict.

{rendered}

Your ONE job is to keep this memory tight and non-redundant so real, distinct
facts don't get crowded out. The tier list has a hard size limit: everything
above it is silently dropped from what the agent sees. Every duplicate line you
leave standing is a slot a genuinely distinct fact should have.

ARCHIVE — collapse duplicates aggressively, then handle contradiction:
- When several lines say the same thing in different words (same subject and
  same core claim, differing only in phrasing or incidental detail), keep ONE
  and archive the rest. This is your most important action. Do NOT require
  identical wording — restatements of the same fact are still duplicates. For
  example, five lines all describing the same writing-voice rule in slightly
  different words are one fact, not five.
    - keep the MOST THOROUGH line (richest detail);
    - if two are equally complete, keep the most recent (newest date);
    - archive every other duplicate.
- Archive a fact directly contradicted or factually superseded by a newer fact
  in this view (e.g. "rm is denied" superseded by "rm works").

Do NOT archive a fact merely for being old, historical, process-related, or
settled — that is a TIER_CHANGE, never an archive. Do NOT archive one-of-a-kind
facts that carry genuinely distinct nuance; only collapse true restatements.

TIER_CHANGE (ONE step down, t1->t2 or t2->t3 only): a settled live decision or
goal demotes as it stops being the current focus. Demote sparingly — never turn
t3 into a graveyard by dumping every old design fact there. Never up-tier except
for a clearly central, durable constraint.

REJECT: only for a pending-marked line you disagree with (the fact is still true
and not a duplicate). If it IS a duplicate of another line, archive it instead.

Work through the list top to bottom, collapse every duplicate group to its one
best line, and archive contradictions. Leave genuinely distinct facts alone.
"""


async def review(
    directory: str,
    *,
    overrides,
    agent_name: str | None = None,
) -> int:
    """Run ONE reviewer curation pass over the rendered memory.

    Reads the current render (``read_render`` — full/unbounded view, no
    journal/index dependency), runs the reviewer agent with ``memory_review``,
    and returns the number of verdicts it issued. The sidecar writes + eager
    re-render happen inside each ``memory_review`` tool call.

    ``agent_name`` is threaded through to the re-render so the reviewer's
    post-archive write_render resolves the agent's real spec tier_caps (e.g.
    demo_assistant's t1:50) instead of falling back to the built-in defaults.
    When omitted, ``memory-reviewer`` is used (the reviewer's own namespace).
    """
    from ._render import read_render
    from ._view import JournalViewStore

    rendered = read_render(directory)
    if not rendered.strip():
        return 0  # nothing to curate yet

    reviewer, _mem = _build_reviewer(
        JournalViewStore(directory),
        overrides,
        agent_name=agent_name or "memory-reviewer",
        namespace="",
    )
    result = await reviewer.run(REVIEW_PROMPT.format(rendered=rendered))
    return reviewer_verdict_count(result)


def reviewer_verdict_count(result) -> int:
    """Count ``memory_review`` calls the reviewer made (verdicts applied)."""
    from pydantic_ai.messages import ToolCallPart

    count = 0
    for msg in getattr(result, "all_messages", lambda: [])():
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart) and getattr(part, "tool_name", "") == (
                "memory_review"
            ):
                count += 1
    return count
