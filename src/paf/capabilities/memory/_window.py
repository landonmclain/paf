"""Sliding-window compaction that consolidates before it evicts.

:class:`MemoryConsolidationWindow` subclasses the harness
``SlidingWindowCompaction`` and overrides ``compact`` to fold the about-to-be-
evicted slice into the memory store (via :func:`.consolidate`) before the
window drops it. It is a drop-in replacement for the ``SlidingWindowCompaction``
tier inside :func:`paf.capabilities.compaction.build_compaction`.

Why ``compact`` and not ``before_model_request``
------------------------------------------------
``TieredCompaction._escalate`` drives tiers directly through their ``compact``
method and explicitly *bypasses* each tier's own trigger (see the harness
class docstring: "Each tier's own trigger is bypassed -- TieredCompaction
drives the tiers directly via their ``compact`` method"). So overriding
``before_model_request`` on a tier would be silently dead when the window sits
inside a ``TieredCompaction`` ladder. Overriding ``compact`` is the seam that
actually fires — ``_escalate`` calls ``tier.compact(messages, ctx)`` exactly
when it intends to trim, so consolidation is bound to the real eviction event.

``find_safe_cutoff(...) <= 0`` (nothing about to be evicted) short-circuits to
a no-op, so the window costs nothing on turns where it would not trim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai_harness.compaction import SlidingWindowCompaction
from pydantic_ai_harness.compaction._shared import find_safe_cutoff

from paf.config import PROJECT_ROOT, settings

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ModelMessage

# Same fixed store the assistant's Memory uses (build.py `_build_memory`). The
# owner writes these plain `.md` bodies + CAS sidecar so the assistant reads the
# exact same durable facts it would have written itself. Anchored to PROJECT_ROOT
# through the shared Settings object (MEMORY_DIR / .env), so both always target
# one canonical directory no matter where the process is launched from.
DEFAULT_MEMORY_DIR = str(PROJECT_ROOT / Path(settings.memory_dir))
DEFAULT_MEMORY_AGENT = "assistant"


@dataclass
class MemoryConsolidationWindow(SlidingWindowCompaction):
    """Sliding window that folds the about-to-evict slice into memory first.

    Same trigger/behaviour as ``SlidingWindowCompaction``; additionally it
    invokes the owner sub-agent on the slice the window is about to drop, so
    durable facts survive eviction. The owner is ``writable=True`` (the sole
    memory writer); the assistant's ``Memory`` stays ``writable=False``
    (read-only).

    The owner writes through the M2 **journal** (``memory-journal.jsonl`` in
    ``memory_directory``), then ``compact`` renders that journal to the root
    ``MEMORY.md`` — the exact file the assistant's read-only
    ``JournalViewStore`` injects. This keeps the live/eviction path on the SAME
    store contract as the benchmark/chunked path; the legacy in-place
    ``MarkdownStorage`` (assistant/MEMORY.md) is NOT used here, so a
    consolidation can never write to a store the assistant doesn't read.

    Store config defaults to the same ``data/memory`` scope the assistant's
    ``Memory`` uses, so both share one durable store. Override
    ``memory_directory`` / ``memory_agent`` to target a different scope.
    """

    memory_directory: str = field(default=DEFAULT_MEMORY_DIR, kw_only=True)
    """Directory for the shared memory store (owns ``memory-journal.jsonl`` + rendered ``MEMORY.md``)."""

    memory_agent: str = field(default=DEFAULT_MEMORY_AGENT, kw_only=True)
    """Agent segment scoping the shared memory store (must match the assistant's)."""

    memory_namespace: str = field(default="", kw_only=True)
    """Optional namespace prefix for the shared memory scope."""

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext,
    ) -> list[ModelMessage]:
        """Consolidate, organize, render, then trim.

        Flow (chosen with Landon 2026-08-21): the **owner** wraps the
        about-to-evict slice into journal records (append-only
        ``memory-journal.jsonl``); the **organizer** reconciles the new journal
        tail into the curated ``memory-index.json`` (dedup/supersede); and the
        **index renderer** writes the root ``MEMORY.md`` from the index — the
        deduped view the assistant's read-only ``JournalViewStore`` injects.
        The journal render is NOT used here: the index is the curated source.
        """
        cutoff = find_safe_cutoff(messages, self.keep_messages)
        if cutoff > 0:
            from ._consolidate import consolidate
            from ._journal import JournalStore
            from ._organize import IndexStore, run_organizer
            from ._render import write_index_render

            journal = JournalStore(self.memory_directory)
            await consolidate(
                messages,
                store=self._build_memory_store(),
                agent_name=self.memory_agent,
                namespace=self.memory_namespace,
                keep_tail=self.keep_messages,
                journal=journal,
            )
            # Organize the new journal tail into the curated index, then
            # materialize MEMORY.md from THAT (not the raw journal, which
            # repeats facts as it grows). Same atomic-write guarantee.
            index_store = IndexStore(self.memory_directory)
            await run_organizer(journal, index_store=index_store)
            write_index_render(
                index_store,
                str(Path(self.memory_directory) / "MEMORY.md"),
            )
        return await super().compact(messages, ctx)

    def _build_memory_store(self):
        """Construct the shared store for the owner's Memory handle.

        Semantically inert for writes once ``journal`` is passed (the owner
        appends through the journal, not the store), but still required by
        ``consolidate``'s signature.
        """
        from paf.capabilities.memory._markdown import MarkdownStorage

        return MarkdownStorage(self.memory_directory)
