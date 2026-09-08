"""Memory consolidation — the writable owner sub-agent.

Backs the **background** watermark consolidation queue
(``paf/capabilities/memory/_background.py``), which folds spans of conversation
into the memory store off the eviction hot path. Eviction
(``TurnBoundSlidingWindow``) no longer calls this — it is trim-only.

Ownership model
---------------
The assistant carries ``writable=False`` — its memory toolset exposes
``read_memory`` / ``search_memory`` only, so it can never write. The owner
agent (this module) carries ``writable=True`` and is the sole writer of memory.
This separation prevents the main agent from contaminating its own durable
memory with incidental or repeated writes while still letting durable facts
from each evicted slice reach the store.

Model selection
---------------
The owner model is read from ``Settings.consol_model`` (``CONSOL_MODEL`` env /
`.env`, aliased automatically by pydantic-settings) with a cheap default,
mirroring how the benchmark harness sources its model knobs. It is deliberately
*not* hardcoded in the source and *not* the spec'd ``assistant.yaml`` model
(that model is the subject; the owner is a separate leaf agent). Override per
run: ``CONSOL_MODEL=openrouter:deepseek/deepseek-v4-flash-0731`` (routing is
pinned via :func:`_consolidation_model_settings`, not a ``:nitro`` suffix).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, cast

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.exceptions import (
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)

from paf.config import settings

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from ._capability import Memory

logger = logging.getLogger(__name__)

# Default: the subject feed model (deepseek, OpenRouter-qualified). Overridable
# via Settings.consol_model (CONSOL_MODEL env / .env) — see module docstring.
# Not the spec'd subject, but the same provider form so it resolves without a
# bare-name inference. Must be `openrouter:`-prefixed (a bare `model:variant`
# string can't resolve to a provider). Routing (which provider serves it) is
# pinned via the `_consolidation_model_settings` provider object, NOT the
# `:nitro` suffix — the suffix means OpenRouter `provider.sort: throughput`,
# which bounces across providers and gives run-to-run nondeterminism.
DEFAULT_CONSOL_MODEL = "openrouter:deepseek/deepseek-v4-flash-0731"

CONSOLIDATION_GUIDANCE = """\
You are the consolidation owner for this user's durable memory. You append
records to an append-only journal — you never edit, move, or delete an existing
record.

Your input is a slice of conversation messages, each labeled with its turn
address (`run_id:turn_id`) and the date it was said. As you read, persist what
survives the salience gate below — and tag each durable fact with a permanence
tier.

APPEND AS YOU SEE IT. Read the chunk top to bottom. The moment you identify a
durable fact, call `append_memory` for it — do not collect facts and dump them
at the end, and do not make multiple passes over the text. Capture each fact in
order as you encounter it, so the journal stays faithful to the conversation's
sequence.

For each durable fact, call `append_memory(type, fact, tier, born_on, source_ref)`
exactly as the tool describes:
- `type` — one of the six semantic kinds (`identity`, `preference`, `fact`,
  `event`, `task`, `current`); pick the kind that best fits.
- `fact` — a COMPLETE, self-contained, canonically-worded statement, subject
  inline. "The user prefers dark roast coffee" — NOT "likes dark roast best".
  If the conversation shows the subject, state it fully; a fact you cannot state
  completely and unambiguously is not durable.
- `tier` — the permanence tier, decided FIRST (below).
- `born_on` — the ISO date the fact was stated, from the message's label. If the
  label carries no date, leave `born_on` empty — NEVER invent a date; a
  fabricated date is worse than none.
- `source_ref` — the message's `{run_id}:{turn_id}` label, so the fact can be
  traced and verified.

TIER — how permanent / how central to knowing the user. Decide this first.
- `t1` STABLE PERSON: identity, preferences, likes/dislikes, constraints/don'ts,
  skills, key durable context, milestones/dated life events. Supersede-only.
- `t2` GOALS / CURRENT WORK: what the user is trying to do now, active projects,
  current objective, pending decisions. Keep while active; supersede when it shifts.
- `t3` EPHEMERAL: one-off events, transient details, technical minutiae, passing
  remarks. Store but low priority; may age out of the injected view.

SALIENCE GATE — decide whether a fact is durable, from the three signals below.
Each true signal is weight. The gate uses CONCRETE signals, not a vague "how
important is this".
- User stated it directly ("I prefer", "I want", "I'm worried about", "I decided",
  "always remember to...")   → +2. This alone is usually durable.
- Will matter NEXT time (cross-session value, a standing rule, a constraint) → +2
- Self-relevant, and about something consequential (identity, work, health,
  family, finances, an active project) → +2

- Any ONE strong signal → durable. It does not need to accumulate two.
- A passing remark / small talk / single-use instruction → skip. Not durable.
- On genuine uncertainty, store at `t3` — never delete. A low tier only lowers
  priority; it never loses the fact.

There is NO `decision` type — the only valid types are the six above. When a
"user decided to do X" moment appears, file it under a valid kind instead of
inventing `decision`: a standing choice or preference → `preference`; a
completed, one-time action → `event`; an upcoming/planned action → `task`; a
settled choice that is really just a durable state about the user → `fact`.
Never pass `decision` as the type — the record will be rejected and the fact
silently lost.

You do not dedupe, merge, update, or delete — the journal is append-only. If the
same subject appears again (even with a newer value), append a new record;
newest-wins and dedup are resolved by the render pass, never by you. There is no
bulk or multi-append tool; you write exactly one fact per call.

Never invent facts not present in the messages. Record them plainly, with no
notes about how or when you recorded them. Never claim you remembered something
unless you actually appended it this turn.

Reply with one short line: what you appended (with tiers), or "no changes" if
nothing was durable.
"""

CONSOLIDATION_PROMPT = """\
Here are messages from a conversation. Remember anything durable from them:

{evicted_text}
"""

# Imported lazily to avoid a circular import at module load: _consolidate
# imports Memory (for the owner's writable=True construct). Resolve lazily
# inside the function.
_MODULE_MEMORY: Memory | None = None


def consolidate_model() -> str:
    """Resolve the owner model from Settings (CONSOL_MODEL / .env), defaulting to deepseek."""
    return settings.consol_model or DEFAULT_CONSOL_MODEL


def _consolidation_model_settings() -> ModelSettings:
    """Build the owner's `model_settings`, pinning OpenRouter routing + sampler.

    Mirrors the benchmark's `_openrouter_model_settings` (`tests/framework.py`)
    exactly (deliberately duplicated, not imported — paf must not depend on the
    test harness): sampling determinism (`top_p: 1.0` + optional fixed seed)
    combined with the OpenRouter `provider` object via `extra_body`. When
    `settings.consol_provider_order` is empty/None, no routing pin is applied
    (plain request). The provider-order + quantization hard filter resolves the
    owner off deepseek fp8 boxes deterministically instead of OpenRouter's
    default throughput sort, which bounces across providers and gives
    run-to-run nondeterminism (the bug this pin closes).
    """
    out: dict = {"temperature": 0}
    out["top_p"] = 1.0
    if settings.consol_seed is not None:
        out["seed"] = settings.consol_seed
    pinned_quantizations = [
        q for q in settings.consol_provider_quantizations if q
    ]
    if settings.consol_provider_order:
        provider: dict = {"order": list(settings.consol_provider_order)}
        if pinned_quantizations:
            provider["quantizations"] = pinned_quantizations
        # allow_fallbacks defaults True on OpenRouter — keep, so a pinned
        # primary dropping still fails over to the listed fallbacks (never to
        # an out-of-quantization box thanks to the hard filter above).
        extra_body: dict = dict(out.get("extra_body") or {})
        extra_body["provider"] = provider
        out["extra_body"] = extra_body
    return cast(ModelSettings, out)


def _build_owner_memory(
    store, *, agent_name: str, namespace: str, inject_memory: bool = False, journal=None
) -> Memory:
    """Build the single-writer Memory capability the owner carries.

    ``writable=True`` (sole writer). ``inject_memory`` defaults to ``False`` —
    the existing eviction-slice owner reads the store through tools (never via
    injected context); pass ``inject_memory=True`` for the chunked seed variant
    (:func:`consolidate_conversation_chunked`) so each chunk pass sees the
    cumulative rendered notebook in-context (the machinery in
    ``Memory.before_model_request`` re-reads the store every request), enabling
    later chunks to append instead of re-derive. ``journal`` is an optional
    :class:`JournalStore` (M2 three-layer model): when set, the owner's toolset
    exposes ONLY ``append_memory`` (append-only, no in-place edits) and the
    rendered ``MEMORY.md`` is derived from the journal at recall time.
    Fixed ``CONSOLIDATION_GUIDANCE``. Extracted so the ownership invariants are
    testable WITHOUT constructing an ``Agent`` (which resolves the model/provider
    and requires an API key).
    """
    from ._capability import Memory  # deferred to avoid circular import

    return Memory(
        store=store,
        agent_name=agent_name,
        namespace=namespace,
        inject_memory=inject_memory,
        guidance=CONSOLIDATION_GUIDANCE,
        writable=True,
        journal=journal,
    )


def _build_owner(
    store, *, agent_name: str, namespace: str, inject_memory: bool = False, journal=None
):
    """Construct the single-writer owner agent (writable=True, read-via-tools).

    Returns ``(owner, owner_memory)`` — the owner ``Agent`` plus its writable
    ``Memory`` capability (see :func:`_build_owner_memory`).

    Shared by the eviction-slice consolidation (:func:`consolidate`) and the
    full-conversation seed variant (:func:`consolidate_conversation`) so the
    ownership invariants live in ONE place: the owner is ``writable=True``
    (the sole memory writer) with memory read-owned either through tools
    (``inject_memory=False``, the eviction/live default) or in-context
    (``inject_memory=True``, the chunked seed variant) and carries the fixed
    ``CONSOLIDATION_GUIDANCE`` operating manual.
    """
    owner_memory = _build_owner_memory(
        store,
        agent_name=agent_name,
        namespace=namespace,
        inject_memory=inject_memory,
        journal=journal,
    )
    owner = Agent(
        consolidate_model(),
        name="memory-owner",
        capabilities=[owner_memory],
        model_settings=_consolidation_model_settings(),
        retries=3,
    )
    return owner, owner_memory


async def resident_consolidate_text(messages: list[ModelMessage]) -> str | None:
    """Render a message list for the owner prompt; None if there's nothing to fold.

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
    anchor) and, when the message carries a ``source_date`` in its metadata, its
    date. This is what lets the owner fill ``source_ref`` / ``born_on`` correctly:
    it copies the label verbatim into ``append_memory`` so the journal record is
    read_turn-resolvable and date-grounded.

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
        turn_refs.append(f"{run_id}:{tid}" if run_id is not None and tid is not None else None)

    def _label(role: str, idx: int, inline_date: str | None = None) -> str:
        ref = turn_refs[idx]
        date = ""
        md = getattr(messages[idx], "metadata", None) or {}
        src_date = md.get("source_date")
        if src_date:
            date = f" [{src_date}]"
        elif inline_date:
            # No metadata ``source_date`` (production path — the harness prepends
            # a ``[date: ...]`` prefix instead of stamping metadata). Derive the
            # date from that inline prefix so ``born_on`` is still real
            # provenance, never a model-invented value.
            date = f" [{inline_date}]"
        return f"{role} {ref}{date}:"

    lines: list[str] = []
    for idx, message in enumerate(messages):
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    content = _user_text(part)
                    if content:
                        lines.append(
                            f"{_label('User', idx, _inline_date(content))} "
                            f"{_strip_inline_date(content)}"
                        )
                # Everything else a request can carry (system prompt, tool return,
                # retry, thinking, native / tool-search / capability-load traffic,
                # file) is not durable user text — dropped.
        elif isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, TextPart):
                    lines.append(f"{_label('Assistant', idx)} {part.content}")
                # Tool calls, thinking, compaction — all dropped.

    if not lines:
        return None
    return "\n".join(lines)


async def consolidate(
    messages: list[ModelMessage],
    *,
    store,
    agent_name: str,
    namespace: str,
    keep_tail: int,
    journal=None,
) -> None:
    """Fold the about-to-be-evicted slice into the store via the owner agent.

    ``keep_tail`` is the window's own ``keep_messages`` bound: the slice
    consolidated is exactly what :func:`.find_safe_cutoff` says the window will
    drop, so consolidation is always *on* the eviction boundary. When nothing
    is about to be evicted, this is a no-op.

    ``journal`` (optional :class:`JournalStore`) switches the owner to the M2
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
    owner, _owner_memory = _build_owner(
        store, agent_name=agent_name, namespace=namespace, journal=journal
    )
    evicted_text = await resident_consolidate_text(evicted)
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
        evicted_text = "No new durable user or assistant text was captured in this span."
    result = await owner.run(
        CONSOLIDATION_PROMPT.format(evicted_text=evicted_text),
    )
    logger.info("consolidated %d evicted msgs: %s", len(evicted), result.output)


async def consolidate_conversation(
    messages: list[ModelMessage],
    *,
    store,
    agent_name: str,
    namespace: str = "",
    journal=None,
) -> None:
    """Fold a *full given* conversation into memory via the owner agent.

    Seed-time variant of :func:`consolidate`: it runs the owner over the
    ENTIRE ``messages`` list (no eviction-cutoff no-op) so a seeded haystack
    conversation can be consolidated into memory *before* a recall question,
    regardless of live sliding-window state (which never sees direct-seeded
    history). The call site is responsible for passing only pre-question
    content (honest-memory guard) — this function folds exactly what it is
    handed.

    ``journal`` (optional :class:`JournalStore`) switches the owner to the M2
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
    result = await owner.run(
        CONSOLIDATION_PROMPT.format(evicted_text=text),
    )
    logger.info(
        "consolidated %d msgs (full conversation): %s",
        len(messages),
        result.output,
    )


# ---------------------------------------------------------------------------
# Chunked seed consolidation (benchmark / large-haystack path)
#
# `consolidate_conversation` folds a whole conversation into memory in ONE
# owner.run. For a large haystack (LongMemEval seeds up to ~443 messages) the
# owner's finite single-shot output budget forces it to cover the vivid/repeated
# clusters and DROP single-mention thematic clusters (e.g. a "video editing"
# preference mentioned once while a dense "photography" theme dominates).
#
# The chunked variant below matches the LIVE (eviction-slice) owner's shape —
# "current memory + a small increment => merge" — repeated over bounded chunks,
# instead of one giant pass. The owner runs once per chunk against the SAME
# store; with `inject_memory=True` memory is re-rendered in-context each request
# (see `Memory.before_model_request`), so chunk N sees the facts chunk N-1
# appended and merges rather than re-derives. Threads stay intact because chunks
# are cut on MESSAGE boundaries, never mid-message.
# ---------------------------------------------------------------------------

# Rough chars-per-token used only to bound chunk size. paf has no tokenizer
# dependency; a chars/4 heuristic keeps every chunk comfortably within a single
# owner output budget without an exact token count (the budget is a heuristic
# bound on prompt size, not a hard token gate). Overridable via
# `LME_CONSOL_CHUNK_TOKENS` (see `_chunk_token_budget`).
_CHARS_PER_TOKEN = 4


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


def _chunk_token_budget() -> int:
    """Token budget per consolidation chunk; env `LME_CONSOL_CHUNK_TOKENS` (default 4000)."""
    raw = os.environ.get("LME_CONSOL_CHUNK_TOKENS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("LME_CONSOL_CHUNK_TOKENS=%r not an int; using 4000", raw)
    return 4000


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
            line.startswith("User:") or line.startswith("User ")
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
    messages: list[ModelMessage],
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

    ``journal`` (optional :class:`JournalStore`) switches the owner to the M2
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
    budget = chunk_max_tokens or _chunk_token_budget()
    chunks = _split_chunks(text, max_tokens=budget)
    for i, chunk in enumerate(chunks, start=1):
        try:
            result = await owner.run(
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
