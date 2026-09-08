"""Conversation-scoped storage for the thread summarizer worker.

The capture WAL is shared by an agent:
``data/summaries/<agent>/evictions.jsonl``.  Each row carries its
``conversation_id``.  Derived summaries and the consumption cursor are scoped
below ``conversations/<stable-key>/`` so two conversations can consume the
shared WAL independently without sharing a watermark.

This module is deliberately deterministic and model-free.  The worker owns the
watermark; callers may read pending blocks and append summaries, but cursor
advancement is an explicit operation performed only after a successful pass.
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
from paf.capabilities.thread_summarizer._capture import (
    EVICTIONS_FILENAME,
    EvictionBlock,
    iter_blocks,
)

WATERMARK_FILENAME = "watermark.json"
SUMMARIES_FILENAME = "summaries.jsonl"


@dataclass
class ThreadWatermark:
    """The last eviction sequence consumed for one conversation."""

    eviction_seq: int = -1

    @classmethod
    def load(cls, directory: str | os.PathLike) -> ThreadWatermark:
        path = Path(directory) / WATERMARK_FILENAME
        if not path.exists():
            return cls()
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            return cls(eviction_seq=int(row.get("eviction_seq", -1)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            # The WAL remains authoritative; a bad cursor only causes replay.
            return cls()

    def save(self, directory: str | os.PathLike) -> None:
        path = Path(directory) / WATERMARK_FILENAME
        payload = json.dumps({"eviction_seq": self.eviction_seq}, ensure_ascii=False) + "\n"
        write_atomic(path, payload)


@dataclass(frozen=True)
class ThreadSummaryRecord:
    """One derived summary for one captured eviction block."""

    eviction_seq: int
    conversation_id: str
    summarized_at: str
    thread: str
    refs: list[str] = field(default_factory=list)
    beats: list[dict[str, Any]] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ThreadSummaryRecord:
        return cls(
            eviction_seq=int(row["eviction_seq"]),
            conversation_id=str(row["conversation_id"]),
            summarized_at=str(row["summarized_at"]),
            thread=str(row.get("thread", "")),
            refs=[str(ref) for ref in row.get("refs", [])],
            beats=list(row.get("beats", [])),
        )


def conversation_key(conversation_id: str) -> str:
    """Return a stable filesystem-safe key for a scoped conversation ID."""
    return hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:24]


def conversation_directory(root: str | os.PathLike, conversation_id: str) -> Path:
    """Derived-store directory for ``conversation_id`` below an agent root."""
    return Path(root) / "conversations" / conversation_key(conversation_id)


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
    """Read the shared eviction WAL and write one conversation's derived files."""

    def __init__(self, root: str | os.PathLike, conversation_id: str) -> None:
        self.root = Path(root)
        self.conversation_id = conversation_id
        self.directory = conversation_directory(root, conversation_id)
        self.wal_path = self.root / EVICTIONS_FILENAME
        self.summaries_path = self.directory / SUMMARIES_FILENAME
        self.watermark_path = self.directory / WATERMARK_FILENAME

    def load_watermark(self) -> ThreadWatermark:
        return ThreadWatermark.load(self.directory)

    def pending_blocks(self) -> list[EvictionBlock]:
        """Return this conversation's WAL blocks after its cursor, in order."""
        watermark = self.load_watermark()
        return [
            block
            for block in sorted(iter_blocks(self.root), key=lambda item: item.seq)
            if block.conversation_id == self.conversation_id
            and block.seq > watermark.eviction_seq
        ]

    def existing_summaries(self) -> list[ThreadSummaryRecord]:
        return _read_summary_records(self.summaries_path)

    def append_summary(self, record: ThreadSummaryRecord) -> None:
        """Append a summary exactly once, atomically and idempotently by seq."""
        if record.conversation_id != self.conversation_id:
            raise ValueError("summary conversation_id does not match store")
        lines = _lines(self.summaries_path)
        existing = _read_summary_records(self.summaries_path)
        for prior in existing:
            if prior.eviction_seq == record.eviction_seq:
                # Retry-safe: timestamps are generated per attempt, but the
                # derived content/provenance must remain identical.
                same_content = (
                    prior.conversation_id == record.conversation_id
                    and prior.thread == record.thread
                    and prior.refs == record.refs
                    and prior.beats == record.beats
                )
                if same_content:
                    return
                raise ValueError(f"summary seq {record.eviction_seq} already has different content")
        if existing and record.eviction_seq <= existing[-1].eviction_seq:
            raise ValueError("summary eviction_seq must increase")
        write_atomic(
            self.summaries_path,
            "".join([*lines, json.dumps(record.to_row(), ensure_ascii=False) + "\n"]),
        )

    def advance(self, eviction_seq: int) -> None:
        """Advance this conversation's cursor atomically; never move backward."""
        watermark = self.load_watermark()
        if eviction_seq < watermark.eviction_seq:
            raise ValueError("watermark cannot move backward")
        watermark.eviction_seq = eviction_seq
        watermark.save(self.directory)


def now_iso() -> str:
    """ISO-8601 UTC timestamp for derived summary records."""
    return datetime.now(UTC).isoformat()
