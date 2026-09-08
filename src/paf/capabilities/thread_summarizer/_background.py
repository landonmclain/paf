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
from paf.capabilities._turns import user_anchor_indices
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

#: Turns per narrative block. The trigger fires only when this many COMPLETE
#: turns have accumulated since the watermark, and the worker summarizes each
#: 10-turn slice as its own block (one model call + one appended record).
BLOCK_TURNS = 10


@dataclass(frozen=True)
class ThreadSummarizeJob:
    """A request to summarize one conversation's unsummarized turn-span.

    ``conversation_id`` is the profile-scoped ``{profile}:{cid}``. ``messages``
    carries the **pre-resolved span** the trigger already computed; ``watermark``
    carries the resolved **advance target** (span end). The worker slices the
    span into ~10-turn blocks, summarizes each, and advances the per-conversation
    watermark incrementally past each successfully-summarized block.
    """

    conversation_id: str
    messages: list | None = None
    watermark: MemoryWatermark | None = None


ThreadSpanSource = Callable[[str, int, str], Awaitable]  # (tip_run, tip_turn, conversation_id) -> ResolvedSpan


def _span_complete_turns(messages: list) -> list[list]:
    """Slice ``messages`` into per-complete-turn lists, ordered.

    Every returned slice is one settled turn (user prompt .. prior to the next
    user anchor); the span is already complete-only (the tip is excluded by
    ``resolve_unseen_span``), so every message belongs to some slice.
    """
    from paf.capabilities._turns import user_anchor_indices

    anchors = user_anchor_indices(messages)
    slices: list[list] = []
    for a, anchor in enumerate(anchors):
        end = anchors[a + 1] if a + 1 < len(anchors) else len(messages)
        slices.append(messages[anchor:end])
    return slices


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

            if job.messages is not None:
                span_messages = job.messages
                span_target = job.watermark
            else:
                if self.span_source is None or job.watermark is None:
                    log.error(
                        "thread_summarizer_queue missing span_source/watermark; dropping "
                        "session=%s", session_id,
                    )
                    return
                resolved = await self.span_source(
                    job.watermark.run_id, job.watermark.turn_id or 0, job.conversation_id
                )
                span_messages = getattr(resolved, "messages", [])
                span_target = getattr(resolved, "watermark", None)

            if not span_messages:
                # Nothing new since the watermark — a successful empty pass (still
                # advances so we don't re-check an empty span every cadence).
                log.info("thread_summarizer_job session=%s success span=0", session_id)
                self._advance(job.conversation_id, span_target)
                if self.on_complete is not None:
                    self.on_complete(session_id, True, "0 turns summarized")
                return

            # Slice into ~10-turn blocks and summarize each as its own record.
            turns = _span_complete_turns(span_messages)
            # Absolute per-turn refs across the WHOLE span (each ref's `turn`
            # is its position in conversation history, not within a slice).
            turn_refs: list[str | None] = [
                self._ref(span_messages, anchor)
                for anchor in user_anchor_indices(span_messages)  # one ref per turn
            ]
            span_runs = [getattr(m, "run_id", None) for m in span_messages]

            store = ThreadSummaryStore(self.capture_root, job.conversation_id)
            summarized = 0
            for start in range(0, len(turns), BLOCK_TURNS):
                block = turns[start : start + BLOCK_TURNS]
                block_msgs: list = []
                for turn in block:
                    block_msgs.extend(turn)
                beats: BeatsRecord = await self.summarize_span_fn(block_msgs, job.conversation_id)
                # Use the span-absolute refs for this block's turn range.
                block_refs = [
                    r for r in turn_refs[start : start + len(block)] if r is not None
                ]
                block_anchor = _block_anchor_from_refs(span_runs, block_refs)
                store.append_summary(
                    summary_from_beats(beats, job.conversation_id, block_anchor, block_refs)
                )
                # Advance positionally past THIS block (idempotent on retry; a
                # later block failing stops here, so only completed blocks are
                # never re-summarized).
                if block_anchor is not None:
                    self._advance(job.conversation_id, block_anchor)
                summarized += len(block)

            # Advance the tail to the resolved span end (covers an inline tip if
            # the last slice stopped shy of span_target).
            self._advance(job.conversation_id, span_target)

            n_blocks = (len(turns) + BLOCK_TURNS - 1) // BLOCK_TURNS
            detail = f"{summarized} turns summarized in {n_blocks} block(s)"
            log.info(
                "thread_summarizer_job session=%s success %s watermark=%s",
                session_id, detail, span_target,
            )
            if self.on_complete is not None:
                self.on_complete(session_id, True, detail)
        except Exception as exc:
            if self.on_complete is not None:
                self.on_complete(session_id, False, str(exc))
            raise

    @staticmethod
    def _ref(messages: list, idx: int) -> str | None:
        from paf.capabilities.thread_summarizer._summarize import _ref_for

        return _ref_for(messages, idx)

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


def _block_anchor_from_refs(span_runs: list[str | None], block_refs: list[str]) -> MemoryWatermark | None:
    """``MemoryWatermark`` at the last turn covered by ``block_refs``.

    ``block_refs`` are conversation-absolute ``{run_id}:{turn_id}`` refs, so the
    last one gives the correct advance position for this block regardless of how
    the block was sliced. Falls back to reversing ``span_runs`` if refs are empty.
    """
    from datetime import UTC, datetime

    ref = block_refs[-1] if block_refs else None
    if ref:
        run_id, _, turn_s = ref.rpartition(":")
        if turn_s.isdigit():
            return MemoryWatermark(
                run_id=run_id,
                turn_id=int(turn_s),
                consumed_ts=datetime.now(UTC),
            )
    # Fallback: last message with a run_id stamp.
    for m in reversed(span_runs):
        if m:
            return MemoryWatermark(
                run_id=m, turn_id=0, consumed_ts=datetime.now(UTC)
            )
    return None


def summary_from_beats(
    beats: BeatsRecord,
    conversation_id: str,
    anchor: MemoryWatermark | None,
    block_refs: list[str] | None = None,
) -> object:
    """Build the durable per-conversation block record from a BeatsRecord.

    The record's ``{run_id, turn_id}`` anchor + ``refs`` identify exactly which
    turns this block summarizes (receipts), so a retried/duplicate pass over the
    same block is idempotent (see :meth:`ThreadSummaryStore.append_summary`).
    """
    from paf.capabilities.thread_summarizer._store import ThreadSummaryRecord

    refs = list(dict.fromkeys(block_refs or beats.turns_covered))
    return ThreadSummaryRecord(
        conversation_id=conversation_id,
        run_id=anchor.run_id if anchor else "",
        turn_id=anchor.turn_id if anchor else -1,
        summarized_at=now_iso(),
        summary=beats.summary,
        refs=refs,
        beats=[beat.model_dump(mode="json") for beat in beats.beats],
    )


def put_watermark_path(capture_root: str, conversation_id: str) -> str:
    """Path of the per-conversation thread watermark file (for tests/docs)."""
    store = ThreadSummaryStore(capture_root, conversation_id)
    return str(store.directory / THREAD_WATERMARK_FILENAME)

