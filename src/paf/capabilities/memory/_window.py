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
    store contract as the benchmark/chunked path.

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

    # Monotonic rate-limit gate: the highest ``find_safe_cutoff`` (about-to-
    # evict prefix) that has already been folded into durable memory by this
    # window instance. The harness drives ``compact`` once per model request
    # (before every tool round-trip), so a long tool-heavy turn can cross the
    # token trigger several times and call ``compact`` repeatedly. Without a
    # gate, each call would re-run the full owner->organizer->render LLM chain
    # on the same/overlapping evictable slice (witnessed: 5 consolidations in
    # one turn). Consolidation is made monotonic: we only fold when the
    # about-to-evict prefix extends BEYOND what we already folded — i.e. when
    # genuinely new material has accumulated. Re-invoking on a cutoff at or
    # below the last folded point is a no-op (still trims, just skips memory).
    _consolidated_cutoff: int | None = field(default=None, init=False, repr=False)

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext,
    ) -> list[ModelMessage]:
        """Consolidate, render, then trim.

        Flow (chosen with Landon 2026-08-21, render source updated 2026-08-25):
        the **owner** wraps the about-to-evict slice into journal records
        (append-only ``memory-journal.jsonl``), and the **deterministic journal
        renderer** writes the root ``MEMORY.md`` straight from that journal —
        the view the assistant's read-only ``JournalViewStore`` injects. The
        journal is the sole source; the organizer's curated index and the
        consolidation second pass are NO LONGER WIRED here (see below).

        Consolidation is rate-limited to once per *eviction generation*: it
        fires only when the about-to-evict prefix is longer than the last one
        folded (fresh material), never on repeated calls over the same slice.

        .. note:: The organizer (``run_organizer``) and consolidation second
           pass (``run_consolidation``) were UNHOOKED 2026-08-25 — they were
           context-blind by construction and drove index churn, so ``MEMORY.md``
           is now rendered deterministically from the append-only journal
           instead of from a curated LLM-built index. Dedup/supersede/
           tier-sorting will return as a deterministic index *derivation*
           (pure code), not an LLM pass. The removed code is preserved in
           ``archive/organizer/`` (kept importable for revival).
        """
        cutoff = find_safe_cutoff(messages, self.keep_messages)
        if cutoff > 0 and (
            self._consolidated_cutoff is None or cutoff > self._consolidated_cutoff
        ):
            from ._consolidate import consolidate
            from ._journal import JournalStore
            from ._render import write_render

            journal = JournalStore(self.memory_directory)
            await consolidate(
                messages,
                store=self._build_memory_store(),
                agent_name=self.memory_agent,
                namespace=self.memory_namespace,
                keep_tail=self.keep_messages,
                journal=journal,
            )
            # Materialize MEMORY.md deterministically from the append-only
            # journal (the sole source). The organizer and consolidation second
            # pass are UNHOOKED (2026-08-25): they were context-blind by
            # construction and drove index churn; the journal render is pure,
            # deterministic, and needs no model. Dedup/supersede/tier-sorting
            # will be re-introduced later as a deterministic index derivation,
            # not an LLM pass.
            write_render(journal, str(Path(self.memory_directory) / "MEMORY.md"))
            self._consolidated_cutoff = cutoff
        return await super().compact(messages, ctx)

    def _build_memory_store(self):
        """Construct the store handle for the owner's `Memory` (journal-view).

        Semantically inert for writes once ``journal`` is passed (the owner
        appends through the journal, not the store), but still required by
        ``consolidate``'s signature and read by the chunked/inject variant.
        A ``JournalViewStore`` of the memory directory — the read-only view of
        the journal-rendered ``MEMORY.md`` — is the correct M2 backend (same as
        the assistant's ``Memory`` in ``build.py``).
        """
        from ._view import JournalViewStore

        return JournalViewStore(self.memory_directory)
