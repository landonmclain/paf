"""Pre-eviction memory consolidation — the writable owner sub-agent.

Backs :class:`MemoryConsolidationWindow` (``paf/capabilities/memory/_window.py``),
which folds the about-to-be-evicted slice of conversation into the memory store
*before* the sliding window drops it.

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
from typing import TYPE_CHECKING, cast

from pydantic_ai import Agent, ModelSettings

from paf.config import settings

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from ._capability import Memory

from ._format import MEMORY_FORMAT

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
You maintain this user's memory and are its only writer. You create, update,
delete, and organize memory files. Nothing else does.

Your input is a slice of recent conversation messages. Persist what is genuinely
worth remembering long-term: decisions, preferences, conventions, lessons,
commitments and pending actions, and stable facts about the user or project.
Skip ordinary small talk and single-use instructions.

NEWER ALWAYS WINS: memory holds the current truth. When an older message
contradicts what memory already records, update memory to the newest value —
do not re-assert the older one, and do not leave conflicting entries. Produce
the single current value.

Update in place; prefer a targeted edit over a duplicate append. Delete an
entry when it is done or superseded.

Write bias — record EVERYTHING durable and every user-stated somatic
(preferences, names, plans, decisions, pending actions); skip only truly
incidental small talk. Pruning a kept entry is cheaper than missing it at
compaction, so err toward writing it.

{format_spec}

AFTER YOU WRITE — self-check the format. Re-read MEMORY.md as it now stands and
verify it still conforms to the format above before you consider the consolidation
done:
- exactly ONE of each section header (## Facts, ## Preferences, ## Pending, ## Notes)
  in that order, and nothing after the last one;
- any lingering fact/pending from a PRIOR consolidation that the new merge
  displaced is folded back under the correct section header — never leave a fact
  stranded under a dangling or duplicate section header, never nest a second
  header mid-file;
- one fact per line, no stray blank prose under Facts/Preferences/Pending.
If the file does not conform, fix it now with a memory write. Do not finish until
the file matches the format exactly. Your final one-line reply should state what
you changed (or \"no changes\") — and if you had to repair the structure, say so.

Deletion is permitted when an entry is done, superseded, or stale. Update is
the default, but do not hoard dead entries. Do not delete on age alone if the
fact is still true.

Never invent facts not present in the messages or memory you were given.
Record facts plainly — no notes about how or when you recorded them.

Reply with one short line: what you changed, or "no changes" if nothing was durable.
"""

CONSOLIDATION_PROMPT = """\
Here are messages from a conversation. Remember anything durable from them:

{evicted_text}
"""

# Imported lazily to avoid a circular import at module load: _consolidate
# imports Memory (for the owner's writable=True construct) and _window imports
# consolidate. resolve lazily inside the function.
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


def _build_owner_memory(store, *, agent_name: str, namespace: str) -> Memory:
    """Build the single-writer Memory capability the owner carries.

    ``writable=True`` (sole writer), ``inject_memory=False`` (reads the store
    through tools, never injected context), fixed ``CONSOLIDATION_GUIDANCE``.
    Extracted so the ownership invariants are testable WITHOUT constructing an
    ``Agent`` (which resolves the model/provider and requires an API key).
    """
    from ._capability import Memory  # deferred to avoid circular import

    return Memory(
        store=store,
        agent_name=agent_name,
        namespace=namespace,
        inject_memory=False,
        guidance=CONSOLIDATION_GUIDANCE.format(format_spec=MEMORY_FORMAT),
        writable=True,
    )


def _build_owner(store, *, agent_name: str, namespace: str):
    """Construct the single-writer owner agent (writable=True, read-via-tools).

    Returns ``(owner, owner_memory)`` — the owner ``Agent`` plus its writable
    ``Memory`` capability (see :func:`_build_owner_memory`).

    Shared by the eviction-slice consolidation (:func:`consolidate`) and the
    full-conversation seed variant (:func:`consolidate_conversation`) so the
    ownership invariants live in ONE place: the owner is ``writable=True``
    (the sole memory writer) with ``inject_memory=False`` (it reads the store
    through tools, never via injected context) and carries the fixed
    ``CONSOLIDATION_GUIDANCE`` operating manual.
    """
    owner_memory = _build_owner_memory(
        store, agent_name=agent_name, namespace=namespace
    )
    owner = Agent(
        consolidate_model(),
        name="memory-owner",
        capabilities=[owner_memory],
        model_settings=_consolidation_model_settings(),
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

    lines: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    content = _user_text(part)
                    if content:
                        lines.append(f"User: {content}")
                # Everything else a request can carry (system prompt, tool return,
                # retry, thinking, native / tool-search / capability-load traffic,
                # file) is not durable user text — dropped.
        elif isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, TextPart):
                    lines.append(f"Assistant: {part.content}")
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
) -> None:
    """Fold the about-to-be-evicted slice into the store via the owner agent.

    ``keep_tail`` is the window's own ``keep_messages`` bound: the slice
    consolidated is exactly what :func:`.find_safe_cutoff` says the window will
    drop, so consolidation is always *on* the eviction boundary. When nothing
    is about to be evicted, this is a no-op.
    """
    from pydantic_ai_harness.compaction._shared import find_safe_cutoff

    cutoff = find_safe_cutoff(messages, keep_tail)
    if cutoff <= 0:
        return  # nothing about to be evicted

    evicted = messages[:cutoff]
    owner, _owner_memory = _build_owner(
        store, agent_name=agent_name, namespace=namespace
    )
    evicted_text = await resident_consolidate_text(evicted)
    if not evicted_text:
        return
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
) -> None:
    """Fold a *full given* conversation into memory via the owner agent.

    Seed-time variant of :func:`consolidate`: it runs the owner over the
    ENTIRE ``messages`` list (no eviction-cutoff no-op) so a seeded haystack
    conversation can be consolidated into memory *before* a recall question,
    regardless of live sliding-window state (which never sees direct-seeded
    history). The call site is responsible for passing only pre-question
    content (honest-memory guard) — this function folds exactly what it is
    handed.
    """
    owner, _owner_memory = _build_owner(
        store, agent_name=agent_name, namespace=namespace
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
