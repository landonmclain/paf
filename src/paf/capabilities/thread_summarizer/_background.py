"""Background worker for conversation-scoped thread beats (cadence + watermark).

Mirrors the Memory **owner** workflow (:class:`paf.capabilities.memory._background.MemoryConsolidationQueue`)
verbatim: the trigger resolves the unseen conversation turn-span, the worker
summarizes it with the curated summarizer, and advances a per-conversation
``MemoryWatermark`` ONLY on a completed (non-error) pass — the identical
advance-on-normal-return contract. The worker is the sole watermark writer.

The one structural delta from Memory's owner: this is keyed by **conversation**
(``{profile}:{cid}``), not agent. ``resolve_unseen_span`` gets a
``conversation_id`` filter so one conversation's beats never bleed into
another's; the watermark lives under
``data/summaries/<agent>/conversations/<hash(cid)>/``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from paf.background import BackgroundQueueService, QueueState
from paf.capabilities.memory._watermark import MemoryWatermark
from paf.capabilities.thread_summarizer._store import (
    THREAD_WATERMARK_FILENAME,
    ThreadSummaryStore,
    now_iso,
)
from paf.capabilities.thread_summarizer._summarize import (
    BeatsRecord,
    SummarizeSpan,
    summarize_span,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThreadSummarizeJob:
    """A cadence request to summarize one conversation's unsummarized turn-span.

    ``conversation_id`` is the profile-scoped ``{profile}:{cid}``. ``messages``
    optionally carries the **pre-resolved span** the trigger already computed;
    when present the worker summarizes it directly instead of re-resolving
    against the (not-yet-advanced) watermark. ``watermark`` carries the
    resolved **advance target** so the worker, on success, advances the
    per-conversation watermark to exactly that position.
    """

    conversation_id: str
    messages: list | None = None
    watermark: MemoryWatermark | None = None


ThreadSpanSource = Callable[[str, int, str], Awaitable]  # (tip_run, tip_turn, conversation_id) -> ResolvedSpan


class ThreadSummarizerQueue(BackgroundQueueService[ThreadSummarizeJob]):
    """FIFO, single-flight worker that owns a conversation's thread watermark."""

    name = "thread-summarizer"

    def __init__(
        self,
        *,
        capture_root: str,
        span_source: ThreadSpanSource | None = None,
        summarize_span_fn: SummarizeSpan | None = None,
        on_complete: Callable[[str, bool, str], None] | None = None,
        name: str = "thread-summarizer",
    ) -> None:
        super().__init__()
        self.capture_root = capture_root
        self.span_source = span_source
        self.summarize_span_fn = summarize_span_fn or summarize_span
        self.on_complete = on_complete
        self.name = name

    def _watermark_path(self, conversation_id: str) -> Path:
        store = ThreadSummaryStore(self.capture_root, conversation_id)
        return store.directory / THREAD_WATERMARK_FILENAME

    async def process_job(
        self,
        session_id: str,
        job: ThreadSummarizeJob,
        state: QueueState[ThreadSummarizeJob],
    ) -> None:
        del state
        try:
            if session_id != job.conversation_id:
                raise ValueError("queue session_id must equal job conversation_id")

            # Pre-resolved span from the trigger, or re-resolve via span_source
            # (conversation-filtered). The ADVANCE TARGET — where the per-conv
            # watermark moves on success — is the trigger's resolved watermark
            # when supplied, else the resolver's own; NEVER the stale on-disk
            # watermark (that is only the cadence reference).
            if job.messages is not None:
                span_messages = job.messages
                advance_target = job.watermark
            else:
                if self.span_source is None:
                    log.error(
                        "thread_summarizer_queue missing span_source; dropping job "
                        "session=%s — the builder must supply one", session_id,
                    )
                    return
                if job.watermark is None:
                    log.error("thread_summarizer_queue missing watermark for job; dropping session=%s", session_id)
                    return
                resolved = await self.span_source(
                    job.watermark.run_id, job.watermark.turn_id or 0, job.conversation_id
                )
                span_messages = getattr(resolved, "messages", [])
                advance_target = getattr(resolved, "watermark", None)

            if not span_messages:
                # Nothing new since the watermark — a successful empty pass.
                log.info("thread_summarizer_job session=%s success span=0", session_id)
                self._advance(job.conversation_id, advance_target)
                if self.on_complete is not None:
                    self.on_complete(session_id, True, "0 turns summarized")
                return

            beats: BeatsRecord = await self.summarize_span_fn(span_messages, job.conversation_id)
            store = ThreadSummaryStore(self.capture_root, job.conversation_id)
            store.append_summary(
                summary_from_beats(beats, job.conversation_id, advance_target)
            )

            # Advance ONLY after a completed (non-error) pass. A thrown model
            # op leaves the watermark back so this span is retried next cadence.
            self._advance(job.conversation_id, advance_target)
            detail = f"{len(span_messages)} turns summarized"
            log.info(
                "thread_summarizer_job session=%s success summarized=%d watermark=%s",
                session_id, len(span_messages), advance_target,
            )
            if self.on_complete is not None:
                self.on_complete(session_id, True, detail)
        except Exception as exc:
            if self.on_complete is not None:
                self.on_complete(session_id, False, str(exc))
            raise

    def _advance(self, conversation_id: str, target: MemoryWatermark | None) -> None:
        """Advance the per-conversation thread watermark to ``target`` (never backward)."""
        if target is None:
            return
        store = ThreadSummaryStore(self.capture_root, conversation_id)
        current = MemoryWatermark.load(
            store.directory, filename=THREAD_WATERMARK_FILENAME
        )
        current.run_id = target.run_id
        current.turn_id = target.turn_id
        current.consumed_ts = target.consumed_ts
        current.save(store.directory, filename=THREAD_WATERMARK_FILENAME)


def summary_from_beats(
    beats: BeatsRecord, conversation_id: str, target: MemoryWatermark | None
) -> object:
    """Build the durable per-conversation summary record anchored at ``target``.

    The record's ``{run_id, turn_id}`` anchor equals the position the
    per-conversation watermark advanced to on this successful pass, so a
    retried/duplicate pass over the same span is idempotent (see
    :meth:`ThreadSummaryStore.append_summary`).
    """
    from paf.capabilities.thread_summarizer._store import ThreadSummaryRecord

    return ThreadSummaryRecord(
        conversation_id=conversation_id,
        run_id=target.run_id if target else "",
        turn_id=target.turn_id if target else -1,
        summarized_at=now_iso(),
        thread=beats.resumed_from,
        refs=beats.turns_covered,
        beats=[beat.model_dump(mode="json") for beat in beats.beats],
    )


def put_watermark_path(capture_root: str, conversation_id: str) -> str:
    """Path of the per-conversation thread watermark file (for tests/docs)."""
    store = ThreadSummaryStore(capture_root, conversation_id)
    return str(store.directory / THREAD_WATERMARK_FILENAME)
