"""Eviction capture for ThreadSummarizer — the append-only evictions WAL.

When a ``TurnBoundSlidingWindow`` drops the oldest settled messages, this
module records *what was evicted and from which thread* as one immutable,
append-only JSONL block per eviction, per agent:
``data/summaries/<agent>/evictions.jsonl``.

Unlike ``Memory`` (cross-conversation durable-fact extraction) this is the raw
episodic record — ordered, explicit-``conversation_id`` eviction blocks — that
the background ``beats`` worker (a later phase) summarizes. Nothing here does
LLM work; capture is deterministic content + stable provenance refs.

Ref model mirrors ``search``/``memory`` exactly: each captured message carries a
``{run_id}:{turn_id}`` ref (``turn_id`` = ordinal of the message's greatest
user-prompt anchor <= its index, computed over the *pre-eviction* settled list)
so a later ``read_turn``/``run_history`` re-verifies the full original turn.
The block also stamps ``conversation_id`` explicitly — the thread it belongs to
— so the worker never has to guess which conversation's evictions these are.

Crash-safety reuses the shared ``write_atomic`` (staging + ``os.replace``), so
a block is never a torn line (see ``paf.capabilities._storage._atomic`` and
memory's ``_append_block_atomic``, which this mirrors).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from paf.capabilities._storage._atomic import write_atomic

EVICTIONS_FILENAME = "evictions.jsonl"


@dataclass(frozen=True)
class CapturedMessage:
    """One evicted message, identified by a stable ``{run_id}:{turn_id}`` ref.

    ``ref`` is ``None`` only when the message carries no ``run_id`` (never
    persisted / synthetic); such messages are kept for content but cannot be
    re-verified against the store.
    """

    ref: str | None
    role: str
    text: str | None


@dataclass(frozen=True)
class EvictionBlock:
    """One eviction event: the dropped messages of one thread, append-only.

    ``seq`` is the file's record number (next line index — 0-based, mirroring
    memory's "id = next line number" journal convention), stable per file.
    """

    seq: int
    conversation_id: str | None
    evicted_at: str
    keep_messages: int
    messages: list[CapturedMessage] = field(default_factory=list)

    def to_row(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict) -> EvictionBlock:
        return cls(
            seq=int(row["seq"]),
            conversation_id=row.get("conversation_id"),
            evicted_at=row["evicted_at"],
            keep_messages=int(row["keep_messages"]),
            messages=[CapturedMessage(**m) for m in row.get("messages", [])],
        )


def _sequence_of(path: Path) -> list[str]:
    """Existing non-staging lines of ``path``, in order (may be empty)."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [ln if ln.endswith("\n") else ln + "\n" for ln in fh]


def append_eviction_block(
    directory: str | os.PathLike,
    block: EvictionBlock,
) -> None:
    """Append ``block`` to ``<directory>/evictions.jsonl`` atomically.

    One JSON object per line; the whole block (all messages) is written as a
    single atomic unit so a crash never leaves a partial record. The file is
    created on first write (parent dirs made). ``block.seq`` must equal the
    file's next sequence (callers derive it via :func:`next_sequence`); this is
    asserted to catch a misused counter, but the append is still atomic.
    """
    path = Path(directory) / EVICTIONS_FILENAME
    lines = _sequence_of(path)
    assert block.seq == len(lines), (
        f"eviction seq {block.seq} != next line {len(lines)} "
        f"for {path} — counter desync"
    )
    line = json.dumps(block.to_row(), ensure_ascii=False)
    write_atomic(path, "".join([*lines, line + "\n"]))


def next_sequence(directory: str | os.PathLike) -> int:
    """The next record sequence for ``<directory>/evictions.jsonl`` (line count)."""
    return len(_sequence_of(Path(directory) / EVICTIONS_FILENAME))


def now_iso() -> str:
    """ISO-8601 UTC timestamp for ``evicted_at``."""
    return datetime.now(UTC).isoformat()


def iter_blocks(directory: str | os.PathLike) -> list[EvictionBlock]:
    """Read all eviction blocks (oldest first) for tests / worker consumption."""
    path = Path(directory) / EVICTIONS_FILENAME
    out: list[EvictionBlock] = []
    if not path.exists():
        return out
    for ln in _sequence_of(path):
        try:
            out.append(EvictionBlock.from_row(json.loads(ln)))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # skip a corrupt/partial line defensively
    return out
