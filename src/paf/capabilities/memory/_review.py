"""Memory reviewer — periodic, off-band curation pass.

Mirrors the owner (``_consolidate.py``) but ONE layer up: the owner **captures**
durable facts into the journal; the reviewer **curates** what's already there.
It is cadence-fired (separate review watermark, like the owner's
``consolidate_every``) and curates the **whole rendered memory**, not just
pending suggestions:

- resolves the assistant's pending ``memory_review`` proposals (approve /
  reject / re-tier), AND
- proactively archives stale/wrong facts, changes tiers, and merges
  complementary fragments of one fact into a single canonical line (its one
  write path, via ``memory_merge`` -> the journal) on its own.

Its read input is the **rendered view only** (``read_render``) — it never reads
the journal or index directly, addressing facts by their stable ``#N`` journal
line id (the same address the assistant and the sidecar use). It writes
verdicts via ``memory_review`` into the same append-only ``memory-overrides.jsonl``
sidecar the assistant writes to (and the merged record appended to the journal);
the render pass applies them deterministically.

Structure mirrors ``_consolidate.py``: a ``Memory`` capability configured as the
REVIEWER (``overrides`` set + ``reviewer=True`` -> ``ReviewerToolset`` exposing
``memory_review`` + ``memory_merge``), wrapped in an ``Agent``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic_ai import Agent, ModelSettings

from paf.degeneration_guard import run_guarded

if TYPE_CHECKING:
    from ._capability import Memory
logger = logging.getLogger(__name__)


def _build_reviewer_memory(
    store,
    overrides,
    *,
    agent_name: str,
    namespace: str,
    journal=None,
) -> Memory:
    """Build the reviewer's Memory capability (curation + merge write).

    ``overrides`` set + ``reviewer=True`` => ``get_toolset`` returns
    ``ReviewerToolset`` exposing ``memory_review`` (apply verdicts) and, when
    ``journal`` is provided, ``memory_merge`` (the reviewer's one write path:
    consolidate complementary facts into a canonical record). The reviewer still
    has no standalone ``append_memory`` (authoring net-new facts is the owner's
    job); merge is constrained to consolidating lines it names as sources.
    Extracted so the invariants are testable without constructing an ``Agent``.
    """
    from ._capability import Memory  # deferred to avoid circular import

    return Memory(
        store=store,
        agent_name=agent_name,
        namespace=namespace,
        inject_memory=False,  # reads the render via read_render, not injection
        guidance="",  # single reviewer prompt is the spec instructions (the user message)
        expose_tools=True,  # carries the reviewer toolset (memory_review + merge)
        overrides=overrides,
        reviewer=True,
        journal=journal,
    )


def _reviewer_spec_config() -> tuple[str, ModelSettings, int, int]:
    """Resolve the reviewer leaf spec (model, model_settings, retries, max_item_chars).

    The ``memory_reviewer`` leaf spec is the source of truth
    (Spec-Driven-Background-Agents) for the reviewer's model, model_settings,
    retries, and max_item_chars — there is no code fallback. A missing or
    model-less spec is a hard error: an agent without a spec'd model isn't an
    agent.
    """
    from paf.specs import resolve_background_spec

    spec, raw = resolve_background_spec("memory_reviewer")
    if not spec.model:
        raise RuntimeError(
            "memory_reviewer spec declares no model — an agent without a "
            "spec'd model isn't an agent"
        )
    model: str = spec.model
    settings = spec.model_settings
    retries = raw.get("retries")
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 1:
        retries = 2  # DEFAULT_REVIEWER_RETRIES
    max_chars = raw.get("max_item_chars")
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
        max_chars = DEFAULT_MAX_ITEM_CHARS
    from typing import cast as _cast

    from pydantic_ai import ModelSettings as _MS

    return model, _cast(_MS, settings), retries, max_chars


DEFAULT_MAX_ITEM_CHARS = 1200


def _reviewer_prompt() -> str:
    """The reviewer's operating prompt — the spec `instructions` (sole source).

    The ``memory_reviewer`` spec's ``instructions`` block is the reviewer's
    operating manual, resolved at run time and formatted with the current
    ``{rendered}`` memory by :func:`review`. There is no code fallback: a
    missing or instructions-less spec is a hard error (the manual must be
    spec'd, not hardcoded).
    """
    from paf.specs import resolve_background_spec

    spec, _raw = resolve_background_spec("memory_reviewer")
    instructions = spec.instructions
    if not instructions:
        raise RuntimeError(
            "memory_reviewer spec declares no instructions — the reviewer's "
            "operating manual must be spec'd, not hardcoded"
        )
    from typing import cast as _cast

    return (
        _cast(str, instructions)
        if not isinstance(instructions, list)
        else str(instructions)
    )


def _build_reviewer(
    store,
    overrides,
    *,
    agent_name: str,
    namespace: str,
    journal=None,
):
    """Construct the reviewer agent (curator: memory_review + memory_merge)."""
    reviewer_memory = _build_reviewer_memory(
        store,
        overrides,
        agent_name=agent_name,
        namespace=namespace,
        journal=journal,
    )
    model, model_settings, retries, _max_chars = _reviewer_spec_config()
    reviewer = Agent(
        model,
        name="memory-reviewer",
        capabilities=[reviewer_memory],
        model_settings=model_settings,
        retries=retries,
    )
    return reviewer, reviewer_memory


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
    from ._journal import JournalStore
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
        # The reviewer's one write path: a writable journal so `memory_merge`
        # can append the canonical consolidated record before archiving sources.
        journal=JournalStore(directory),
    )
    result = await run_guarded(reviewer, _reviewer_prompt().format(rendered=rendered))
    return reviewer_verdict_count(result)


def reviewer_verdict_count(result) -> int:
    """Count curation actions the reviewer made (``memory_review`` + ``memory_merge``)."""
    from pydantic_ai.messages import ToolCallPart

    count = 0
    for msg in getattr(result, "all_messages", lambda: [])():
        for part in getattr(msg, "parts", []):
            if isinstance(part, ToolCallPart) and getattr(part, "tool_name", "") in (
                "memory_review",
                "memory_merge",
            ):
                count += 1
    return count
