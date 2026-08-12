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
The owner model is read from env (``CONSOL_MODEL``) with a cheap default,
mirroring how ``SUBJECT_MODEL`` / ``JUDGE_MODEL`` are sourced in the benchmark
harness. It is deliberately *not* hardcoded in the source and *not* the spec'd
``assistant.yaml`` model (that model is the subject; the owner is a separate
leaf agent). Override per run: ``CONSOL_MODEL=google/gemini-2.5-flash-lite:nitro``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from pydantic_ai import Agent

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from ._capability import Memory

logger = logging.getLogger(__name__)

# Default: the cheap flash-lite class. Overridable via CONSOL_MODEL env —
# see module docstring. Not the spec'd subject model.
DEFAULT_CONSOL_MODEL = "google/gemini-2.5-flash-lite:nitro"

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

Update in place; prefer a targeted edit over a duplicate append. Move entries
into the right files; keep MEMORY.md a small curated notebook. Take the time
to be complete and well-organized.

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
    """Resolve the owner model from CONSOL_MODEL env, defaulting to flash-lite."""
    return os.environ.get("CONSOL_MODEL", DEFAULT_CONSOL_MODEL)


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
    from pydantic_ai_harness.compaction._shared import _collect_text, find_safe_cutoff

    cutoff = find_safe_cutoff(messages, keep_tail)
    if cutoff <= 0:
        return  # nothing about to be evicted

    evicted = messages[:cutoff]
    if not evicted:
        return

    from ._capability import Memory  # deferred to avoid circular import

    owner_memory = Memory(
        store=store,
        agent_name=agent_name,
        namespace=namespace,
        inject_memory=False,
        guidance=CONSOLIDATION_GUIDANCE,
        writable=True,
    )
    owner = Agent(
        consolidate_model(),
        name="memory-owner",
        capabilities=[owner_memory],
    )
    evicted_text = "\n".join(f"- {seg}" for seg in _collect_text(evicted))
    result = await owner.run(
        CONSOLIDATION_PROMPT.format(evicted_text=evicted_text),
    )
    logger.info("consolidated %d evicted msgs: %s", len(evicted), result.output)
