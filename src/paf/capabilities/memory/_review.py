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

REVIEWER_GUIDANCE = """\
You are the memory reviewer. You curate this user's durable memory — you do NOT
add new facts (that's the owner's job). You review what is already stored and
decide what stays, what gets archived, and where things should live.

Your input is the current rendered memory: a tiered list (## t1 / ## t2 / ## t3)
of facts, each on a line ending with its journal line id (#N). Some lines carry
a `*(archive pending review)*` marker — those are the assistant's suggestions
awaiting your verdict.

Use the `memory_review` tool to apply verdicts by line id:

- `archive` (journal_id) — the fact is stale, wrong, superseded, or no longer
  true: drop it. Use this for pending-marked lines you agree with, and for any
  other fact you judge should no longer be shown.
- `tier_change` (journal_id, tier) — the fact is still valid but belongs at a
  different permanence level. Move it (t1 => t2/t3 to deprioritize, t2 => t1 if
  it's now central). This is the reviewer's judgment call; the assistant can't
  do it.
- `reject` (journal_id) — keep the fact exactly as-is AND clear a pending
  marker you disagree with (the suggestion was wrong; the fact stays).

Curate the WHOLE list, not just the pending-marked lines:
- Facts that are wrong, contradicted, or no longer relevant -> archive.
- Stale/less-central facts that are still true -> tier_change down.
- Core identity/person facts should stay t1. Don't archive things you aren't
  fairly confident are wrong or stale — memory is better with a little
  redundancy than missing a live constraint.

Be decisive but conservative. Only archive when you're reasonably confident the
fact is stale or wrong. Prefer tier_change (keep, deprioritize) over archive
when the fact is merely aged but plausibly still true.
"""


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
        guidance=REVIEWER_GUIDANCE,
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
Below is the user's current durable memory, tiered (## t1 / ## t2 / ## t3),
one fact per line, each ending in its journal line id (#N). Some lines carry a
`*(archive pending review)*` marker — those are assistant suggestions awaiting
your verdict.

{rendered}

Curate it now with `memory_review`. Review the WHOLE list, not just the pending
lines. For each fact you judge stale/wrong/superseded, `memory_review(journal_id,
op="archive", reason=...)`. For facts still valid but mis-tiered,
`memory_review(journal_id, op="tier_change", tier=..., reason=...)`. For pending
suggestions you disagree with, `memory_review(journal_id, op="reject", reason=...)`.
Be conservative: only archive when you are reasonably confident the fact is wrong
or stale; prefer `tier_change` (keep, deprioritize) when it's merely aged.
"""


async def review(
    directory: str,
    *,
    overrides,
) -> int:
    """Run ONE reviewer curation pass over the rendered memory.

    Reads the current render (``read_render`` — full/unbounded view, no
    journal/index dependency), runs the reviewer agent with ``memory_review``,
    and returns the number of verdicts it issued. The sidecar writes + eager
    re-render happen inside each ``memory_review`` tool call.
    """
    from ._render import read_render
    from ._view import JournalViewStore

    rendered = read_render(directory)
    if not rendered.strip():
        return 0  # nothing to curate yet

    reviewer, _mem = _build_reviewer(
        JournalViewStore(directory),
        overrides,
        agent_name="memory-reviewer",
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
