"""Conversation-scoped storage for the thread summarizer.

The worker owns a per-**conversation** thread watermark
(:class:`paf.capabilities.memory._watermark.MemoryWatermark`) persisted below
``conversations/<stable-key>/`` — one watermark per ``{profile}:{cid}``, so two
conversations never share a cursor or cross-summarize. Summaries are appended
to the same per-conversation ``summaries.jsonl``, anchored by the
``{run_id, turn_id}`` position the watermark advanced to on a successful pass,
making a retried pass idempotent.

This module is deterministic and model-free. The worker advances the watermark
ONLY after a completed (non-error) summarize pass; callers may read and append,
but advancement is an explicit post-success operation — mirroring the Memory
owner's advance-on-normal-return contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paf.capabilities._storage._atomic import write_atomic

SUMMARIES_FILENAME = "summaries.jsonl"
#: Per-conversation thread watermark (a ``MemoryWatermark`` in a distinct file so
#: it tracks alongside, but independently of, any memory watermark in the same dir).
THREAD_WATERMARK_FILENAME = "thread-watermark.json"


def thread_watermark_filename() -> str:
    return THREAD_WATERMARK_FILENAME


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def conversation_key(conversation_id: str) -> str:
    """Return a stable filesystem-safe key for a scoped conversation ID."""
    return hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:24]


def conversation_directory(root: str | os.PathLike, conversation_id: str) -> Path:
    """Derived-store directory for ``conversation_id`` below an agent root."""
    return Path(root) / "conversations" / conversation_key(conversation_id)


@dataclass(frozen=True)
class ThreadSummaryRecord:
    """One derived narrative block for one ~10-turn conversation span.

    Anchored by ``{run_id, turn_id}`` — the position the per-conversation
    thread watermark advanced to when this span was successfully summarized —
    so a retried/duplicate pass over the same span is a no-op, mirroring
    Memory's idempotent append by source ref.
    """

    conversation_id: str
    run_id: str
    turn_id: int
    summarized_at: str
    summary: str
    refs: list[str] = field(default_factory=list)
    beats: list[dict[str, Any]] = field(default_factory=list)
    # ISO timestamps of the summarized span's first and last distinct real user
    # prompt (started_at .. ended_at). Wall-clock range of the turns this block
    # covers — distinct from ``summarized_at`` (when the worker ran). Used by the
    # Phase-3 in-context renderer to give the model + a human a recency/span
    # signal. Backward-compatible: absent on legacy records (``from_row``
    # defaults to "").
    started_at: str = ""
    ended_at: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ThreadSummaryRecord:
        return cls(
            conversation_id=str(row["conversation_id"]),
            run_id=str(row.get("run_id", "")),
            turn_id=int(row.get("turn_id", -1)),
            summarized_at=str(row["summarized_at"]),
            summary=str(row.get("summary", "")),
            refs=[str(ref) for ref in row.get("refs", [])],
            beats=list(row.get("beats", [])),
            started_at=str(row.get("started_at", "")),
            ended_at=str(row.get("ended_at", "")),
        )


def _lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [line if line.endswith("\n") else line + "\n" for line in fh]


def _read_summary_records(path: Path) -> list[ThreadSummaryRecord]:
    records: list[ThreadSummaryRecord] = []
    for line in _lines(path):
        try:
            records.append(ThreadSummaryRecord.from_row(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return records


class ThreadSummaryStore:
    """Read/write one conversation's thread beats + watermark (no eviction WAL).

    The conversation-scoped directory holds both the appended ``summaries.jsonl``
    and the per-conversation ``MemoryWatermark``. The shared eviction WAL is no
    longer consumed — the trigger resolves the unsummarized span directly from
    the persisted step store via ``resolve_unseen_span(conversation_id=...)``.
    """

    def __init__(self, root: str | os.PathLike, conversation_id: str) -> None:
        self.root = Path(root)
        self.conversation_id = conversation_id
        self.directory = conversation_directory(root, conversation_id)
        self.summaries_path = self.directory / SUMMARIES_FILENAME

    def existing_summaries(self) -> list[ThreadSummaryRecord]:
        return _read_summary_records(self.summaries_path)

    def last_anchor(self) -> tuple[str, int] | None:
        """``(run_id, turn_id)`` of the most recently-appended summary, if any.

        Used by the worker's catch-up: a block whose would-be anchor is already
        at-or-below this is a previously-covered span (e.g. after a stale or
        cross-process-diverged watermark) and should be skipped, advancing the
        watermark past it rather than hard-failing on "anchor must increase".
        ``None`` when the file is empty (nothing summarized yet).
        """
        recs = _read_summary_records(self.summaries_path)
        if not recs:
            return None
        last = recs[-1]
        if not last.run_id:
            return None
        return (last.run_id, last.turn_id)


    def append_summary(self, record: ThreadSummaryRecord) -> None:
        """Append a summary exactly once, idempotently by its run/turn anchor."""
        if record.conversation_id != self.conversation_id:
            raise ValueError("summary conversation_id does not match store")
        lines = _lines(self.summaries_path)
        existing = _read_summary_records(self.summaries_path)
        for prior in existing:
            if (prior.run_id, prior.turn_id) == (record.run_id, record.turn_id):
                same_content = (
                    prior.conversation_id == record.conversation_id
                    and prior.summary == record.summary
                    and prior.refs == record.refs
                    and prior.beats == record.beats
                )
                if same_content:
                    return
                raise ValueError(
                    f"summary anchor ({record.run_id}:{record.turn_id}) already has different content"
                )
        if existing and (record.run_id, record.turn_id) <= (
            existing[-1].run_id,
            existing[-1].turn_id,
        ):
            raise ValueError("summary run/turn anchor must increase")
        write_atomic(
            self.summaries_path,
            "".join([*lines, json.dumps(record.to_row(), ensure_ascii=False) + "\n"]),
        )
