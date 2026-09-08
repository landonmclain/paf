"""Background worker for conversation-scoped thread beats."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from paf.background import BackgroundQueueService, QueueState
from paf.capabilities.thread_summarizer._capture import EvictionBlock
from paf.capabilities.thread_summarizer._store import (
    ThreadSummaryRecord,
    ThreadSummaryStore,
    now_iso,
)
from paf.capabilities.thread_summarizer._summarize import BeatsRecord, summarize_block

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThreadSummarizeJob:
    """A cadence request to consume one conversation's pending eviction blocks."""

    conversation_id: str


SummarizeBlock = Callable[[EvictionBlock], Awaitable[BeatsRecord]]


class ThreadSummarizerQueue(BackgroundQueueService[ThreadSummarizeJob]):
    """FIFO, single-flight worker that owns a conversation's watermark."""

    name = "thread-summarizer"

    def __init__(
        self,
        *,
        capture_root: str,
        summarize_block_fn: SummarizeBlock | None = None,
        on_complete: Callable[[str, bool, str], None] | None = None,
        name: str = "thread-summarizer",
    ) -> None:
        super().__init__()
        self.capture_root = capture_root
        self.summarize_block_fn = summarize_block_fn or summarize_block
        self.on_complete = on_complete
        self.name = name

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

            store = ThreadSummaryStore(self.capture_root, job.conversation_id)
            pending = store.pending_blocks()
            if not pending:
                log.info("thread_summarizer_job session=%s success pending=0", session_id)
                if self.on_complete is not None:
                    self.on_complete(session_id, True, "0 blocks summarized")
                return

            # Every append is idempotent by eviction_seq. If a later block fails,
            # the cursor remains behind and the next cadence retries the gap.
            for block in pending:
                beats = await self.summarize_block_fn(block)
                record = ThreadSummaryRecord(
                    eviction_seq=block.seq,
                    conversation_id=job.conversation_id,
                    summarized_at=now_iso(),
                    thread=beats.resumed_from,
                    refs=[message.ref for message in block.messages if message.ref is not None],
                    beats=[beat.model_dump(mode="json") for beat in beats.beats],
                )
                store.append_summary(record)

            # Sole writer; advance only after every pending block summarized and
            # durably appended. A thrown model/store operation leaves the cursor back.
            store.advance(pending[-1].seq)
            detail = f"{len(pending)} blocks summarized"
            log.info(
                "thread_summarizer_job session=%s success summarized=%d watermark=%s",
                session_id,
                len(pending),
                pending[-1].seq,
            )
            if self.on_complete is not None:
                self.on_complete(session_id, True, detail)
        except Exception as exc:
            if self.on_complete is not None:
                self.on_complete(session_id, False, str(exc))
            raise
