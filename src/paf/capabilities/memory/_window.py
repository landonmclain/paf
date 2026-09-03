"""Turn-bounded sliding-window compaction — never evicts the active turn.

Two windows ship here:

* :class:`TurnBoundSlidingWindow` — the **assistant's** eviction tier. Trim-only
  and turn-bounded: it drops the oldest *settled* messages (strictly before the
  most recent real user prompt) and does **no memory work**. This is the only
  window ``build_compaction`` assembles; there is no ``consolidate`` toggle.
* :class:`MemoryConsolidationWindow` — **benchmark-only** (LongMemEval
  ``tests/framework.py`` / ``run_lmem``). Same turn-bounded base, but it ALSO
  folds the about-to-evict settled slice into the durable memory journal before
  trimming — the harness's build pass turns evicted evidence into a durable
  notebook. It is deliberately NOT wired into the assistant path.

Turn-bounding is inherited by both: ``compact`` slices at the most recent real
user prompt, so neither window can ever consume or degrade the active turn.

Why ``compact`` and not ``before_model_request``
------------------------------------------------
``TieredCompaction._escalate`` drives tiers directly through their ``compact``
method and explicitly *bypasses* each tier's own trigger (see the harness
class docstring: "Each tier's own trigger is bypassed -- TieredCompaction
drives the tiers directly via their ``compact`` method"). So overriding
``before_model_request`` on a tier would be silently dead when the window sits
inside a ``TieredCompaction`` ladder. Overriding ``compact`` is the seam that
actually fires — ``_escalate`` calls ``tier.compact(messages, ctx)`` exactly
when it intends to trim, so the gate is bound to the real eviction event.
``find_safe_cutoff(...) <= 0`` (nothing about to be evicted) short-circuits to
a no-op, so a window costs nothing on turns where it would not trim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai_harness.compaction import SlidingWindowCompaction
from pydantic_ai_harness.compaction._shared import find_safe_cutoff

from paf.config import PROJECT_ROOT, settings

if TYPE_CHECKING:
    from pydantic_ai import RunContext

# Mid-turn eviction gate (Landon 2026-09-02, Option A): the eviction tier must
# never consume the ACTIVE user turn. `compact()` is driven by
# TieredCompaction._escalate before every model request, so it fires mid-turn —
# exactly when the active prompt is vulnerable to find_safe_cutoff's back-walk
# (run 01a05ff8: the live "implement it" prompt was evicted while its response
# survived). The gate bounds every eviction decision to the completed prefix
# strictly BEFORE the most recent real user prompt, so the current turn is
# structurally unreachable by the window tier.
#
# The "<memory>" injected context parts carry a non-str `content` (a list), so
# they are deliberately NOT turn starts — this mirrors runtime2._is_turn_start
# so the gate and the resume_window renderer see the same turns.
def _is_real_user_prompt(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart) and isinstance(part.content, str)
        for part in (getattr(message, "parts", None) or [])
    )


# Same fixed store the memory-owner uses (build.py `_build_memory`). The owner
# writes these plain `.md` bodies + CAS sidecar so the subject reads the exact
# same durable facts it would have written itself. Anchored to PROJECT_ROOT
# through the shared Settings object (MEMORY_DIR / .env), so both always target
# one canonical directory no matter where the process is launched from.
DEFAULT_MEMORY_DIR = str(PROJECT_ROOT / Path(settings.memory_dir))
DEFAULT_MEMORY_AGENT = "assistant"


class TurnBoundSlidingWindow(SlidingWindowCompaction):
    """Turn-bounded, trim-only sliding window — never evicts the active turn.

    The assistant's eviction tier. Same trigger/behaviour as
    ``SlidingWindowCompaction``; additionally it bounds every eviction decision
    to the settled prefix strictly before the most recent real user prompt,
    re-appending the active turn verbatim.

    It performs **no memory work** — durable-fact capture is owned solely by the
    background watermark worker (``MemoryConsolidationQueue`` in
    :mod:`._background`), never by eviction. There is no ``consolidate`` flag;
    this window does not know memory exists.

    When ``keep_messages`` protects fewer messages than the tool-pair walk's
    healthy minimum, ``find_safe_cutoff`` may still back-walk *within the settled
    prefix* — but it can never reach the active turn, because that turn is sliced
    out and re-appended by this gate.
    """

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext,
    ) -> list[ModelMessage]:
        """Trim only the settled prefix, never consuming the active turn.

        **Mid-turn gate (2026-09-02, Option A):** ``compact`` is driven by
        ``TieredCompaction._escalate`` *before every model request*, so it fires
        mid-turn. To forbid eviction of the ACTIVE user turn, every decision is
        bounded to the completed prefix before the most recent real user prompt:
        the base window is invoked on that prefix alone, with the active turn
        re-appended verbatim. A history that is entirely a live turn (or has
        nothing evictable in completed turns) is returned untouched — the window
        tier never touches it.

        ClearToolResults / ClampOversizedMessages are separate tiers and
        continue to run mid-turn.
        """
        # === Mid-turn gate: locate the active turn's start ===
        anchor: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            if _is_real_user_prompt(messages[i]):
                anchor = i
                break
        # Evictable = strictly before the last real user prompt. If the whole
        # history is the active turn (no completed prefix), nothing may drop.
        if anchor is None or anchor == 0:
            return messages
        settled = list(messages[:anchor])  # completed turns only
        active = messages[anchor:]         # current turn — never evicted
        # Recompute the cutoff against the SETTLED prefix only. If nothing is
        # evictable there (short completed history), skip entirely — even the
        # active tail must not be pushed through the base, which would re-derive
        # an un-clamped cutoff over the full list.
        cutoff = find_safe_cutoff(settled, self.keep_messages)
        if cutoff <= 0:
            return messages
        # Run the base window over the settled prefix ONLY, then reattach the
        # active turn verbatim. The base's find_safe_cutoff / find_token_cutoff
        # can therefore never reach the active prompt.
        trimmed_settled = await super().compact(settled, ctx)
        return [*trimmed_settled, *active]


@dataclass
class MemoryConsolidationWindow(TurnBoundSlidingWindow):
    """Turn-bounded window that ALSO folds the evicted slice into memory.

    **Benchmark-only** (LongMemEval ``tests/framework.py`` / ``run_lmem``): it
    adds the inline consolidation fold on top of :class:`TurnBoundSlidingWindow`
    — folding the about-to-evict SETTLED slice into the durable journal before
    trimming. This is the harness's build pass: evicted evidence becomes a
    durable notebook. It is intentionally NOT wired into the assistant path,
    which uses trim-only ``TurnBoundSlidingWindow``.

    The owner is ``writable=True`` (the sole memory writer); the subject's
    ``Memory`` stays ``writable=False`` (read-only). The owner writes through the
    M2 **journal** (``memory-journal.jsonl``), then renders ``MEMORY.md`` from it.

    Store config defaults to the same ``data/memory`` scope the subject's
    ``Memory`` uses, so both share one durable store. Override
    ``memory_directory`` / ``memory_agent`` to target a different scope.
    """

    memory_directory: str = field(default=DEFAULT_MEMORY_DIR, kw_only=True)
    """Directory for the shared memory store (owns ``memory-journal.jsonl`` + rendered ``MEMORY.md``)."""

    memory_agent: str = field(default=DEFAULT_MEMORY_AGENT, kw_only=True)
    """Agent segment scoping the shared memory store (must match the subject's)."""

    memory_namespace: str = field(default="", kw_only=True)
    """Optional namespace prefix for the shared memory scope."""

    # Monotonic rate-limit gate: the highest ``find_safe_cutoff`` (about-to-
    # evict prefix) that has already been folded into durable memory by this
    # window instance. Without a gate, repeated compact() calls would re-run the
    # full owner->render chain on the same/overlapping evictable slice.
    # Consolidation is made monotonic: only fold when the about-to-evict prefix
    # extends BEYOND what we already folded.
    _consolidated_cutoff: int | None = field(default=None, init=False, repr=False)

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext,
    ) -> list[ModelMessage]:
        """Consolidate the settled slice, render, then trim — never consuming the active turn.

        Splits at the most recent real user prompt (exactly as the base), folds
        the about-to-evict SETTLED prefix into the journal, re-renders
        ``MEMORY.md`` from the journal, then trims only the settled prefix and
        re-appends the active turn verbatim.
        """
        # Reuse the base gate to find the settled/active split.
        anchor: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            if _is_real_user_prompt(messages[i]):
                anchor = i
                break
        if anchor is None or anchor == 0:
            return messages
        settled = list(messages[:anchor])  # completed turns only
        cutoff = find_safe_cutoff(settled, self.keep_messages)
        if cutoff <= 0:
            return messages
        if cutoff > 0 and (
            self._consolidated_cutoff is None or cutoff > self._consolidated_cutoff
        ):
            from ._consolidate import consolidate
            from ._index import update_index
            from ._journal import JournalStore
            from ._render import write_render

            journal = JournalStore(self.memory_directory)
            await consolidate(
                settled,
                store=self._build_memory_store(),
                agent_name=self.memory_agent,
                namespace=self.memory_namespace,
                keep_tail=self.keep_messages,
                journal=journal,
            )
            write_render(journal, str(Path(self.memory_directory) / "MEMORY.md"))
            update_index(self.memory_directory)
            self._consolidated_cutoff = cutoff
        # Delegate the trim to the turn-bounded base (settled only, active
        # re-appended verbatim).
        return await super().compact(messages, ctx)

    def _build_memory_store(self):
        """Build the store handle for the owner's `Memory` (journal-view).

        Semantically inert for writes once ``journal`` is passed (the owner
        appends through the journal, not the store), but still required by
        ``consolidate``'s signature and read by the chunked/inject variant.
        """
        from ._view import JournalViewStore

        return JournalViewStore(self.memory_directory)
