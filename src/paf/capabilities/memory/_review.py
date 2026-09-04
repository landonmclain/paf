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

THE FIRST RULE IS CONSERVATION. Memory is cheap to keep and expensive to lose.
A fact you archive is gone from the injected view forever unless someone
deliberately re-appends it. When in doubt, DO NOTHING to that fact. Aggressive
archiving is a bug, not a feature — missing a live constraint is far worse than
carrying a little harmless redundancy. Most facts should be left exactly as-is.

Only ARCHIVE a fact when it falls into one of these HIGH-CONFIDENCE classes:
  1. DIRECTLY CONTRADICTED — a newer fact in the same view states the opposite
     of this one (e.g. "rm is denied" vs "rm works"). Archive the wrong one.
  2. FACTUALLY SUPERSEDED — a newer fact fully replaces this one (e.g. "needs a
     way to persist the server" superseded by "server runs in tmux now").
  3. VERBATIM / NEAR-IDENTICAL DUPLICATE — two lines state the SAME fact almost
     word-for-word (the same claim, same subject). Archive the weaker copy only.
Do NOT archive merely because a fact is "aged", "redundant", "historical",
"process-related", "moot", "planned-then-executed", or "less central". Those are
TIER_CHANGE (or nothing), never archive. Redundancy across facts that each carry
some distinct nuance is expected and fine.

Use TIER_CHANGE sparingly and only down one step at a time (t1->t2 or t2->t3):
  - A fact that was a live decision/goal but is now settled-and-recorded may
    move ONE tier down (t1->t2, t2->t3). It is still a real fact; it just is no
    longer the current focus.
  - Do NOT move every old design fact to t3 — that turns t3 into a graveyard and
    destroys the tier ladder's meaning. A few well-placed demotions per pass is
    plenty. Most things you think about demoting, you should instead leave.
  - Never move a fact UP tier (t2->t1) reflexively; only do so for a clearly
    central, durable constraint.

DO NOT LET THE SAME FACT LIVE AT MULTIPLE TIERS. If you notice one fact is
expressed in more than one line (e.g. a t1/t2 line and a t3 line saying the same
thing), that is the one situation where you may resolve it — keep the single
most complete/central copy and archive the merely-repeated weaker copies. Left
alone, split copies make the tiers inconsistent.

For pending-marked lines (`*(archive pending review)*`):
  - AGREE (it truly is contradicted/superseded/duplicate) -> archive.
  - DISAGREE (it's still true, just aged) -> `reject` (keep as-is, clear marker).
  Err toward `reject`: a pending marker is the assistant guessing; you are the
  approver. If you would not archive the fact on your own, reject the suggestion.

Be decisive on clear cases and passive on everything else. A pass that archives
NOTHING and changes ONE tier is a perfectly good pass. Restraint is the metric.
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

You are under NO obligation to do anything. Producing zero verdicts is a
perfectly normal, good pass — the memory may simply be healthy as-is. Memory is
cheap to keep and expensive to lose; only act on the HIGH-CONFIDENCE cases below.

Act ONLY on facts that clearly fall into one of these classes, otherwise leave
the fact untouched:

- archive: the fact is directly contradicted by a newer fact in this view, or
  factually superseded by one, or is a verbatim/near-identical duplicate of
  another line (keep the more complete copy). Do NOT archive for age,
  redundancy, being historical/process/moot, or being less central.
- tier_change (ONE step down, t1->t2 or t2->t3 only): a settled live decision
  is being demoted; demote sparingly, a few per pass at most. Never up-tier
  except for a clearly central durable constraint. Do not let the same fact
  live at multiple tiers — if you see split copies, keep the best one and
  archive the weak repeats.
- reject: a pending suggestion you disagree with (the fact is still true, just
  aged). Err toward reject over archive.

For every fact that is not a clear HIGH-CONFIDENCE case: do nothing. A pass that
archives nothing and changes nothing is a good, expected pass. Restraint is the
metric.
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
