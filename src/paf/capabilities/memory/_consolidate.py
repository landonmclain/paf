"""Memory consolidation — the writable owner sub-agent.

Backs the **background** watermark consolidation queue
(``paf/capabilities/memory/_background.py``), which folds spans of conversation
into the memory store off the eviction hot path. Eviction
(``TurnBoundSlidingWindow``) no longer calls this — it is trim-only.

Ownership model
---------------
The memory owner is the sole writer: it carries the writable memory capability
and persists durable facts from each evicted slice into the store. The main
assistant agent reads memory only, so it cannot contaminate its own durable
memory with incidental writes.

Model selection
---------------
The owner's model, model_settings, and operating guidance come from the
``memory_extractor`` leaf spec
(``specs/background_agents/memory_extractor.yaml``) — the source of truth for
this agent (Spec-Driven-Background-Agents). There is no code/env fallback: a
missing or model-less spec is a hard error — an agent without a spec'd model
isn't an agent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic_ai import Agent, ModelSettings

from paf.degeneration_guard import run_guarded

from ._toolset import ExtractionContext

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from ._capability import Memory

logger = logging.getLogger(__name__)


CONSOLIDATION_PROMPT = """\
PROJECT: {project}

MESSAGES:
{evicted_text}
"""


def render_consolidation_prompt(evicted_text: str, project: str | None = None) -> str:
    """The owner's user message: the conversation's project, then the slice.

    Per-call content lives here (not in the spec instructions) so the
    instructions stay static and cacheable. ``project`` is ``None`` for an
    untagged conversation (rendered ``none``).
    """
    return CONSOLIDATION_PROMPT.format(
        project=project or "none", evicted_text=evicted_text
    )


# Resolved lazily to avoid a circular import at module load: _consolidate
# imports Memory (for the owner's capability construct).
_MODULE_MEMORY: Memory | None = None


def _build_owner_memory(
    store,
    *,
    agent_name: str,
    namespace: str,
    inject_memory: bool = False,
    journal=None,
    guidance: str = "",
    extraction: ExtractionContext | None = None,
) -> Memory:
    """Build the single-writer Memory capability the owner carries.

    ``inject_memory`` defaults to ``False`` —
    the eviction-slice owner reads the store through tools (never via injected
    context); pass ``inject_memory=True`` for the benchmark chunked seed
    variant (``tests/_lmem_consolidate.consolidate_conversation_chunked``) so
    each chunk pass sees the
    cumulative rendered notebook in-context (the machinery in
    ``Memory.before_model_request`` re-reads the store every request), enabling
    later chunks to append instead of re-derive. ``journal`` is an optional
    :class:`JournalStore`: when set, the owner's toolset exposes ONLY
    ``append_memory`` (append-only, no in-place edits) and the rendered
    ``MEMORY.md`` is derived from the journal at recall time. ``guidance`` is the
    operating manual (defaults to empty; a caller passes the spec-resolved
    instructions). Extracted so the ownership invariants are
    testable WITHOUT constructing an ``Agent`` (which resolves the model/provider
    and requires an API key).
    """
    from ._capability import Memory  # deferred to avoid circular import

    return Memory(
        store=store,
        agent_name=agent_name,
        namespace=namespace,
        inject_memory=inject_memory,
        guidance=guidance,
        journal=journal,
        extraction=extraction,
    )


def _owner_spec_config() -> tuple[str, ModelSettings, str]:
    """Resolve the extractor leaf spec (model, model_settings, guidance).

    The ``memory_extractor`` leaf spec is the source of truth
    (Spec-Driven-Background-Agents) for the owner's model, model_settings, and
    operating guidance — there is no code/env fallback. A missing or
    model-less spec is a hard error: an agent without a spec'd model isn't an
    agent.
    """
    from paf.specs import resolve_background_spec

    spec, _raw = resolve_background_spec("memory_extractor")
    if not spec.model:
        raise RuntimeError(
            "memory_extractor spec declares no model — an agent without a "
            "spec'd model isn't an agent"
        )
    model: str = spec.model
    settings = spec.model_settings
    guidance = spec.instructions
    if not guidance:
        raise RuntimeError(
            "memory_extractor spec declares no instructions — the owner's "
            "operating manual must be spec'd, not hardcoded"
        )
    # Narrow the loosely-typed AgentSpec fields (TemplateStr/dict) — the same
    # cast pattern the writer sub-agent uses; these are plain str/ModelSettings
    # in practice (YAML block scalar + model_settings dict).
    from typing import cast as _cast

    from pydantic_ai import ModelSettings as _MS

    return (
        model,
        _cast(_MS, settings),
        _cast(str, guidance) if not isinstance(guidance, list) else str(guidance),
    )


def _build_owner(
    store,
    *,
    agent_name: str,
    namespace: str,
    inject_memory: bool = False,
    journal=None,
    extraction: ExtractionContext | None = None,
):
    """Construct the sole-writer owner agent (read-via-tools).

    Returns ``(owner, owner_memory)`` — the owner ``Agent`` plus its writable
    ``Memory`` capability (see :func:`_build_owner_memory`).

    Shared by the eviction-slice consolidation (:func:`consolidate`) and the
    benchmark full-conversation seed variants
    (``tests/_lmem_consolidate``) so the
    ownership invariants live in ONE place: memory is read-owned either through
    tools (``inject_memory=False``, the eviction/live default) or in-context
    (``inject_memory=True``, the chunked seed variant) and carries the spec's
    operating guidance.
    """
    model, model_settings, owner_guidance = _owner_spec_config()
    owner_memory = _build_owner_memory(
        store,
        agent_name=agent_name,
        namespace=namespace,
        inject_memory=inject_memory,
        journal=journal,
        guidance=owner_guidance,
        extraction=extraction,
    )
    owner = Agent(
        model,
        name="memory-owner",
        capabilities=[owner_memory],
        model_settings=model_settings,
        retries=3,
    )
    return owner, owner_memory


async def resident_consolidate_text(messages: list[ModelMessage]) -> str | None:
    """Render a message list for the owner prompt; None if there's nothing to fold.

    Thin wrapper over :func:`consolidation_view` for callers that only need the
    text (see there for the full contract).
    """
    text, _ref_dates = await consolidation_view(messages)
    return text


async def consolidation_view(
    messages: list[ModelMessage],
) -> tuple[str | None, dict[str, str]]:
    """Render a message list for the owner prompt, plus each label's turn date.

    Returns ``(text, ref_dates)``: ``text`` is the labeled slice (``None`` if
    there's nothing to fold); ``ref_dates`` maps every ``run_id:turn_id``
    label in it to that TURN's date (``""`` when the turn carries none). The
    date is a property of the turn, not the message — an assistant reply
    inherits its turn's user-message date — so ``append_memory`` can stamp
    ``born_on`` in code instead of asking the model to copy it.

    Deliberately a *dedicated* owner view rather than reusing the search / read_turn
    renderers (`_render.py` / `_reader.py`) or the harness token estimator
    (`_collect_text`): the owner extracts durable facts, so it sees ONLY the
    durable User and Assistant text — labeled, in order. Everything that is not a
    fact source is dropped: assistant thinking/reasoning, tool calls AND tool
    results (their I/O is transient, and results can echo memory content back),
    system prompts (the instruction channel, not conversation), retries, native /
    tool-search / capability-load traffic, files, and compaction artifacts.

    Each emitted message is labeled with its ``{run_id}:{turn_id}`` turn address
    (the SAME convention :func:`search_conversation_history` labels and
    :func:`read_turn` resolves — turn_id = ordinal of the message's user-prompt
    anchor) and its turn's date when known. The owner copies the label into
    ``append_memory``'s ``source_ref`` (receipt-checked against ``ref_dates``),
    so the journal record is read_turn-resolvable; ``born_on`` is stamped from
    ``ref_dates`` in code.

    Injected artifacts (`<memory>` / `<turn-index>` user-prompt blocks tag their
    ``TextContent`` with a marker) are stripped via
    :func:`paf.capabilities._markers.injected_artifact_markers` so a snapshotted
    memory or index block never reaches the owner.
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextContent,
        TextPart,
        UserPromptPart,
    )

    from paf.capabilities._markers import injected_artifact_markers
    from paf.capabilities._turns import user_anchor_indices

    markers = injected_artifact_markers()

    def _is_injected(item: object) -> bool:
        metadata = getattr(item, "metadata", None)
        return metadata is not None and any(
            str(metadata).startswith(marker) for marker in markers
        )

    def _user_text(part: UserPromptPart) -> str | None:
        """The durable text of a user prompt, stripped of injected blocks.

        Returns None when nothing durable remains (e.g. an all-injected ``<memory>``
        snapshot), so the message is omitted rather than shown empty.
        """
        if isinstance(part.content, str):
            return part.content or None
        texts: list[str] = []
        for item in part.content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, TextContent) and not _is_injected(item):
                texts.append(item.content)
        joined = " ".join(texts).strip()
        return joined or None

    # Precompute each message's `{run_id}:{turn_id}` ref (same walk as the
    # conversation-search toolset: turn_id = ordinal of the message's greatest
    # user-prompt anchor <= its index). Messages before any anchor get None.
    runs = [getattr(m, "run_id", None) for m in messages]
    anchors = user_anchor_indices(messages)
    turn_refs: list[str | None] = []
    for idx, _m in enumerate(messages):
        tid = None
        for a in range(len(anchors) - 1, -1, -1):
            if anchors[a] <= idx:
                tid = a
                break
        run_id = runs[idx]
        turn_refs.append(
            f"{run_id}:{tid}" if run_id is not None and tid is not None else None
        )

    # Pass 1: collect the durable lines, and each turn's date (metadata
    # ``source_date`` first, else the inline ``[date: ...]`` prefix the harness
    # puts on user messages).
    # (role, message index, text, the message's own date)
    entries: list[tuple[str, int, str, str]] = []
    ref_dates: dict[str, str] = {}
    for idx, message in enumerate(messages):
        md = getattr(message, "metadata", None) or {}
        ref = turn_refs[idx]
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    content = _user_text(part)
                    if content:
                        date = md.get("source_date") or _inline_date(content)
                        if ref is not None and date and not ref_dates.get(ref):
                            ref_dates[ref] = str(date)
                        entries.append(
                            ("User", idx, _strip_inline_date(content), str(date or ""))
                        )
                # Everything else a request can carry (system prompt, tool return,
                # retry, thinking, native / tool-search / capability-load traffic,
                # file) is not durable user text — dropped.
        elif isinstance(message, ModelResponse):
            if ref is not None and md.get("source_date") and not ref_dates.get(ref):
                ref_dates[ref] = str(md["source_date"])
            for part in message.parts:
                if isinstance(part, TextPart):
                    entries.append(
                        (
                            "Assistant",
                            idx,
                            part.content,
                            str(md.get("source_date") or ""),
                        )
                    )
                # Tool calls, thinking, compaction — all dropped.

    # Pass 2: label every line with its turn address and the TURN's date.
    lines: list[str] = []
    for role, idx, text, own_date in entries:
        ref = turn_refs[idx]
        if ref is not None:
            ref_dates.setdefault(ref, "")
        # The turn's date when addressable; a message with no turn address
        # (no run_id) still shows its own date.
        date = ref_dates.get(ref, "") if ref is not None else own_date
        label = f"{role} {ref}" + (f" [{date}]" if date else "") + ":"
        lines.append(f"{label} {text}")

    if not lines:
        return None, ref_dates
    return "\n".join(lines), ref_dates


def _inline_date(text: str) -> str | None:
    """Extract the date from a leading ``[date: ...]`` prefix, else ``None``.

    The harness (benchmark and production) prefixes the message date as
    ``[date: <iso>]``. When a message carries no ``metadata['source_date']``,
    the consolidation owner reads this inline value (via the label) so it can
    fill ``born_on`` with real provenance instead of inventing one.
    """
    if text.startswith("[date: "):
        end = text.find("]")
        if end != -1:
            return text[len("[date: ") : end].strip() or None
    return None


def _strip_inline_date(text: str) -> str:
    """Strip a leading ``[date: ...]`` inline prefix (used for search/read_turn).

    The benchmark harness prefixes a ``[date: ...]`` token (first turn of each
    session) and the production harness prepends one to EVERY user message, so
    the model always has an explicit time anchor. The owner view resolves the
    date into the label (metadata ``source_date`` first, else the inline
    prefix) and drops the redundant prefix from the fact TEXT so it never leaks
    into the stored ``fact`` — the date lives in ``born_on``, not in prose.
    """
    if text.startswith("[date: "):
        end = text.find("]")
        if end != -1:
            return text[end + 1 :].lstrip()
    return text


async def consolidate(
    messages: list[ModelMessage],
    *,
    store,
    agent_name: str,
    namespace: str,
    keep_tail: int,
    journal=None,
    project: str | None = None,
) -> None:
    """Fold the about-to-be-evicted slice into the store via the owner agent.

    ``keep_tail`` is the window's own ``keep_messages`` bound: the slice
    consolidated is exactly what :func:`.find_safe_cutoff` says the window will
    drop, so consolidation is always *on* the eviction boundary. When nothing
    is about to be evicted, this is a no-op.

    ``journal`` (optional :class:`JournalStore`) switches the owner to the
    append-only model (``append_memory`` only); ``MEMORY.md`` is rendered from
    the journal at recall time. When set, ``store`` is still required (the
    owner's ``Memory`` holds it) but the owner writes through the journal, not
    the store.
    """
    from pydantic_ai_harness.compaction._shared import find_safe_cutoff

    cutoff = find_safe_cutoff(messages, keep_tail)
    if cutoff <= 0:
        return  # nothing about to be evicted

    evicted = messages[:cutoff]
    evicted_text, ref_dates = await consolidation_view(evicted)
    owner, _owner_memory = _build_owner(
        store,
        agent_name=agent_name,
        namespace=namespace,
        journal=journal,
        extraction=ExtractionContext(ref_dates=ref_dates, project=project),
    )
    if not evicted_text:
        # No durable user/assistant text in the span (pure tool/empty traffic).
        # Run the owner anyway over a minimal synthesized view so it always
        # gets a pass (the watermark only advances on a real fold, so a
        # genuinely empty span costing the owner one run is harmless — it
        # either writes a real fact or folds 0 and the material is retried
        # next cadence rather than silently skipped).
        logger.info(
            "consolidate: span of %d msgs has no durable text; running owner "
            "defensively over an empty view (folds only if it writes)",
            len(evicted),
        )
        evicted_text = (
            "No new durable user or assistant text was captured in this span."
        )
    result = await run_guarded(
        owner,
        render_consolidation_prompt(evicted_text, project),
    )
    logger.info("consolidated %d evicted msgs: %s", len(evicted), result.output)
