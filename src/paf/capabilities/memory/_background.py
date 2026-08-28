"""Background memory-consolidation consumer — the ``MemoryConsolidationQueue``.

Wires the framework-level work-queue engine (:class:`paf.background.BackgroundQueueService`,
committed ``29faa67``) to the M2 memory store, fired by the *watermark* trigger
(:mod:`._watermark`) instead of eviction.

Where ``MemoryConsolidationWindow.compact`` runs the owner inline because the
window is about to drop messages, this queue runs the owner **out-of-band** —
the turn returns immediately and a background worker drains the job. The turn
never blocks on the owner's LLM work. The eviction hook is intentionally left
in place (see design): the watermark is the *driver*, eviction stays the
*safety net*, and the two paths share the one append-only journal — the
queue's FIFO + ``write_lock`` serialize journal writes so the paths never
stomp each other.

The heavy seams are injectable so the service + trigger mechanics can be
unit-tested without a model call or a fake persistence substrate:

- ``span_source`` — given a ``(run_id, turn_id)`` target, returns the unseen
  span's messages to fold. **Supplied by the builder**, bound to the same
  ``HistorySource`` the search capability holds (the queue never reconstructs a
  persistence store from a directory string). Tests supply a synthetic one.
- ``consolidate_span`` — folds a resolved span's messages into the journal.
  Defaults to the real owner (:func:`._consolidate.consolidate`); tests inject a
  lightweight stand-in (still journal-fenced, so correctness is honest).

The consumer writes through the **same** journal + ``write_render`` as the
eviction path and the benchmark — the measured contract is untouched (see the
watermark spec's "does this break the benchmark" section).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from paf.background import BackgroundQueueService, QueueState

from ._journal import JournalStore
from ._watermark import _WATERMARK_FILENAME, MemoryWatermark

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage  # noqa: F401

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsolidationJob:
    """A request to fold the unseen persisted span into durable memory.

    ``run_id`` / ``turn_id`` mark the *current* position; the worker folds
    everything strictly newer than the stored watermark, then advances the
    watermark to ``turn_id``.
    """

    run_id: str
    turn_id: int


#: Type of the span-source seam: resolve "(run_id, up_to_turn_id)" into the
#: ordered messages to fold. This is where the real persistence read sits
#: (``TurnFinder``/``read_turn``); a test supplies a synthetic list.
SpanSource = Callable[[str, int], Awaitable[list["ModelMessage"]]]

#: Type of the consolidate seam: fold a span's messages into the journal.
#: Defaults to the real owner; a test injects a journal-fenced stand-in.
ConsolidateSpan = Callable[
    [list["ModelMessage"], str, MemoryWatermark], Awaitable[int]
]


class MemoryConsolidationQueue(BackgroundQueueService[ConsolidationJob]):
    """Background owner queue fired by the watermark trigger.

    One instance == one queue == one worker per session (the engine's
    composition model). ``process_job`` folds the unseen span since the
    watermark via ``consolidate_span``, advances the watermark file, and
    re-renders ``MEMORY.md``.

    The span to fold is **provided by the caller** (``span_source``) — the
    queue never reconstructs a persistence store from a directory string. The
    agent that builds this queue supplies a ``span_source`` bound to the same
    ``HistorySource`` the search capability already holds (e.g. a
    ``SqliteStepStore`` / ``FileStepStore`` the runtime persists to). This keeps
    the queue a thin, honest consumer: it does not guess at where history
    lives. A test supplies a synthetic span source standing in for persistence.

    ``consolidate_span`` defaults to the real owner (:func:`._consolidate.consolidate`);
    a test injects a journal-fenced stand-in so the service + trigger mechanics
    are proven without an LLM call.

    NOTE: deliberately NOT a ``@dataclass``. Subclassing
    :class:`BackgroundQueueService` requires ``super().__init__()`` to run so
    the engine's ``_states`` dict is created (see ``tests/test_background.py``:
    a dataclass with its own fields generates an ``__init__`` that skips the
    base). We write an explicit ``__init__`` that calls the base, mirroring that
    established consumer pattern.
    """

    name: str = "memory-consolidation"

    def __init__(
        self,
        *,
        memory_directory: str = "",
        agent_name: str = "assistant",
        name: str = "memory-consolidation",
        span_source: SpanSource | None = None,
        consolidate_span: ConsolidateSpan | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.memory_directory = memory_directory
        self.agent_name = agent_name
        self.span_source = span_source
        self.consolidate_span = consolidate_span

    async def process_job(
        self,
        session_id: str,
        job: ConsolidationJob,
        state: QueueState[ConsolidationJob],
    ) -> None:
        del state  # unused; single queue, FIFO serialization is enough
        directory = self.memory_directory
        if not directory:
            log.error("memory_consolidation_queue missing memory_directory; dropping job")
            return

        journal = JournalStore(directory)
        watermark = MemoryWatermark.load(directory, run_id=job.run_id)

        # Resolve the unseen span from persistence (or a synthetic source).
        if self.span_source is None:
            log.error(
                "memory_consolidation_queue has no span_source; cannot fold job "
                "session=%s run=%s turn=%s — the builder must supply one",
                session_id, job.run_id, job.turn_id,
            )
            return
        messages = await self.span_source(job.run_id, job.turn_id)

        # Fold into the journal (append-only; dedup/supersede at render).
        if self.consolidate_span is not None:
            new_records = await self.consolidate_span(messages, directory, watermark)
        else:
            new_records = await self._default_consolidate_span(
                messages, directory, watermark
            )

        # Advance the watermark past what we just consumed.
        watermark.run_id = job.run_id
        watermark.turn_id = job.turn_id
        watermark.save(directory)

        # Materialize MEMORY.md deterministically from the append-only journal —
        # the exact render the eviction path and benchmark produce.
        from ._render import write_render

        write_render(journal, str(Path(directory) / "MEMORY.md"))
        log.info(
            "memory_consolidation_job session=%s folded=%s watermark=%s:%s",
            session_id,
            new_records,
            watermark.run_id,
            watermark.turn_id,
        )

    async def _default_consolidate_span(
        self,
        messages: list[ModelMessage],
        directory: str,
        watermark: MemoryWatermark,
    ) -> int:
        """Real fold: run the owner over ``messages`` into the journal.

        Returns the number of journal records appended (for the log/assert).
        """
        from ._consolidate import consolidate
        from ._view import JournalViewStore

        journal = JournalStore(directory)
        await consolidate(
            messages,
            store=JournalViewStore(directory),
            agent_name=self.agent_name,
            namespace="",
            keep_tail=0,
            journal=journal,
        )
        return journal.count()


def put_watermark_path(directory: str | os.PathLike) -> str:
    """Path of the watermark file within a memory directory (for tests/docs)."""
    return str(Path(directory) / _WATERMARK_FILENAME)
