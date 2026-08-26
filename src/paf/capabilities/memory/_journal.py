"""Append-only durable-facts journal, sitting beside ``MEMORY.md`` (the M2 journal).

This is the **dump** layer of the three-layer memory design (journal ->
(render) -> ``MEMORY.md``): a write-once, append-only JSONL log of *what was
seen and when*, each record a durable fact extracted from a chunk of source
conversation. The relational concerns — dedup, recency-supersession, stale
soft-archive, section assignment — deliberately live in the *render* pass, NOT
here. The journal never edits, compacts, dedups, or reorders history; a fact
seen again (possibly updated) in a later chunk legitimately appends a second
record with a later ``born_on``, and "newest wins" is decided downstream.

Design decisions (all LOCKED with Landon, 2026-08-17):

- **Record shape** — immutable ``{type, fact, born_on, source_ref}``. ``id`` is
  the 1-based JSONL line number (stable forever because the journal is never
  rewritten; no stored id field, no counter -- just position).
- **No ``entity`` field** — the subject lives inline in ``fact`` prose; ``fact``
  is the sole searchable axis (BM25 search over fact text at render/query).
- **Dump semantics, no write-time dedup** — the journal "just extracts what it
  sees in that chunk no matter of what is already there." The ONLY write-side
  safety is atomicity (a chunk's records are appended as one block; a crash
  mid-block never leaves a half-written line).
- **Deliberately free of ``agent-foundation``** — we mirror the *structure* of the
  turn-index store (append-only JSONL, `{run_id}:{turn_id}` ref model) but NOT its
  ``StorageService`` seam. Like ``_atomic`` before it, this module is
  self-contained: atomic line-writes via the local ``_atomic.write_atomic``.
  ``agent-foundation`` was an idea that didn't sit well after the fact; no new
  dependency.
- **Sits beside ``MEMORY.md``** — same directory, same flat-file role. The journal
  is the append-only source; ``MEMORY.md`` (the render) is the presentation the
  answerer reads.

Layout::

    <memory dir>/
      MEMORY.md                # the render (presentation only, produced by the render pass)
      memory-journal.jsonl     # this module: the append-only fact stream

Ref model: ``source_ref`` is the existing turn addressing
``{run_id}:{turn_id}`` — identical to what ``search_conversation_history``
labels and what ``read_turn('{run_id}:{turn_id}')`` resolves. No new addressing.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime

from ._atomic import write_atomic

_JOURNAL_FILENAME = "memory-journal.jsonl"

# Semantic memory types (LOCKED). These are NOT the benchmark's 4 render sections;
# section assignment happens at render time from `type`.
MEMORY_TYPES: frozenset[str] = frozenset(
    {"identity", "preference", "fact", "event", "task", "current"}
)

# Permanence tiers (added 2026-08-25 for the owner importance rubric). Perpendicular
# to `type`: type says *what kind of thing* a fact is (for search), tier says *how
# permanent / how central to knowing the user* (for bounded, tier-sorted injection).
# t1 = stable person (identity/preferences/constraints/skills/milestones; supersede-only)
# t2 = goals / current work (active projects, current objective; superseded when it shifts)
# t3 = ephemeral (one-off events, transient details, technical minutiae; ages out of inject)
MEMORY_TIERS: frozenset[str] = frozenset({"t1", "t2", "t3"})


class MemoryRecord:
    """One immutable journal record: a durable fact, append-only.

    Fields (LOCKED 2026-08-17; ``tier`` added 2026-08-25):
      type       -- one of ``MEMORY_TYPES`` (identity|preference|fact|event|task|current)
      fact       -- the durable statement; subject lives inline (no entity field)
      tier       -- one of ``MEMORY_TIERS`` (t1|t2|t3); permanence for bounded injection.
                    REQUIRED (we start fresh — no legacy rows without a tier).
      born_on    -- ISO datetime of the source message (corpus clock, not wall-clock)
      source_ref -- ``{run_id}:{turn_id}`` turn addressing, read_turn-resolvable

    ``id`` is NOT stored -- it is the record's 1-based JSONL line number, assigned
    by :meth:`JournalStore.append`/``.append_many``.
    """

    __slots__ = ("type", "fact", "tier", "born_on", "source_ref")

    def __init__(
        self,
        *,
        type: str,
        fact: str,
        tier: str,
        born_on: str | datetime | None = None,
        source_ref: str,
    ) -> None:
        if type not in MEMORY_TYPES:
            raise ValueError(
                f"invalid memory type {type!r}; expected one of {sorted(MEMORY_TYPES)}"
            )
        if tier not in MEMORY_TIERS:
            raise ValueError(
                f"invalid memory tier {tier!r}; expected one of {sorted(MEMORY_TIERS)}"
            )
        fact = (fact or "").strip()
        if not fact:
            raise ValueError("fact must be non-empty")
        source_ref = (source_ref or "").strip()
        if not source_ref:
            raise ValueError("source_ref must be non-empty")
        self.type = type
        self.fact = fact
        self.tier = tier
        self.born_on = _normalize_born_on(born_on)
        self.source_ref = source_ref

    # --- serialization -----------------------------------------------------

    def to_row(self) -> dict:
        return {
            "type": self.type,
            "fact": self.fact,
            "tier": self.tier,
            "born_on": self.born_on,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_row(cls, row: dict) -> MemoryRecord:
        return cls(
            type=row["type"],
            fact=row["fact"],
            tier=row["tier"],
            born_on=row["born_on"],
            source_ref=row["source_ref"],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryRecord):
            return NotImplemented
        return (
            self.type == other.type
            and self.fact == other.fact
            and self.tier == other.tier
            and self.born_on == other.born_on
            and self.source_ref == other.source_ref
        )

    def __repr__(self) -> str:
        return (
            f"MemoryRecord(type={self.type!r}, fact={self.fact!r}, tier={self.tier!r}, "
            f"born_on={self.born_on!r}, source_ref={self.source_ref!r})"
        )


def _normalize_born_on(value: str | datetime | None) -> str:
    """Normalize ``born_on`` to a UTC ISO-8601 string (string out for dump/read ease).

    Accepts a ``datetime`` (naive is assumed UTC per the corpus-clock convention)
    or an ISO string. Stored in ISO order so chronological sort is lexical.

    An empty/``None`` value is allowed and stored as ``""``: a fact with no
    derivable date (no label, no metadata, no inline prefix) is recorded with
    empty ``born_on`` rather than a fabricated one. Recency ordering must
    tolerate empty values (they sort last).
    """
    if value is None:
        return ""
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat()
    # String: try to normalize, else keep as-is (ISO already sorts lexically).
    text = value.strip()
    if not text:
        return ""
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat()
    except ValueError:
        return text


def _append_block_atomic(path: str | os.PathLike, lines: list[str]) -> None:
    """Append ``lines`` (each already JSON-serialized, one record per line) atomically.

    Reads the current file, appends the new lines, writes the whole thing back via
    ``write_atomic`` (staging + ``os.replace``). This is the crash-safety guarantee:
    a reader sees either the prior complete journal or the new complete journal,
    never a torn line. **Explicitly not** a bare ``open(..., 'a')`` append: that
    could leave a partial trailing line on crash, and an append-only journal must
    never hold a corrupt line.
    """
    path = os.fspath(path)
    existing: list[str] = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    existing.append(ln if ln.endswith("\n") else ln + "\n")
    parts = existing + lines
    write_atomic(path, "".join(parts))


class JournalStore:
    """Append-only durable-facts journal (one JSON record per line).

    ``directory`` is the memory scope dir — the same one that holds ``MEMORY.md``.
    The journal file is ``<directory>/memory-journal.jsonl``.

    API is deliberately minimal and dump-correct:
      - :meth:`append` -- one record (id = next line number)
      - :meth:`append_many` -- a chunk's records as ONE atomic block
      - :meth:`read_all` -- all records with their 1-based line-number ids
      - :meth:`count` -- number of records

    No ``has``/``latest_for``/idempotency gate: duplicates are expected and
    resolved at render time. Re-reading a resumed chunk legitimately appends new
    records (recency wins at render).
    """

    def __init__(self, directory: str | os.PathLike) -> None:
        self._dir = os.fspath(directory)
        self._path = os.path.join(self._dir, _JOURNAL_FILENAME)
        self._lock = threading.RLock()

    @property
    def path(self) -> str:
        return self._path

    @property
    def filename(self) -> str:
        return _JOURNAL_FILENAME

    def _next_line_number(self) -> int:
        """Next id = current record count + 1 (1-based line number)."""
        return self.count() + 1

    # --- appends -----------------------------------------------------------

    def append(
        self,
        *,
        type: str,
        fact: str,
        tier: str,
        born_on: str | datetime,
        source_ref: str,
    ) -> MemoryRecord:
        """Append one record; returns it. Its id is the new line number."""
        record = MemoryRecord(type=type, fact=fact, tier=tier, born_on=born_on, source_ref=source_ref)
        with self._lock:
            line = json.dumps(record.to_row(), ensure_ascii=False)
            _append_block_atomic(self._path, [line + "\n"])
        return record

    def append_record(self, record: MemoryRecord) -> MemoryRecord:
        """Append an already-constructed :class:`MemoryRecord`."""
        return self.append(
            type=record.type,
            fact=record.fact,
            tier=record.tier,
            born_on=record.born_on,
            source_ref=record.source_ref,
        )

    def append_many(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        """Append ``records`` as ONE atomic block (a chunk's extraction).

        Returns the records (unchanged). All-or-nothing: a crash mid-block leaves
        the journal at the prior complete state — no partial chunk.
        """
        if not records:
            return []
        with self._lock:
            lines = [json.dumps(r.to_row(), ensure_ascii=False) + "\n" for r in records]
            _append_block_atomic(self._path, lines)
        return records

    # --- reads -------------------------------------------------------------

    def read_all(self) -> list[tuple[int, MemoryRecord]]:
        """All records as ``(line_number_id, record)`` pairs, in append order.

        The id is the record's 1-based JSONL line number — stable forever because
        the journal is never rewritten/compacted.
        """
        out: list[tuple[int, MemoryRecord]] = []
        if not os.path.exists(self._path):
            return out
        with self._lock:
            with open(self._path, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    out.append((lineno, MemoryRecord.from_row(row)))
        return out

    def count(self) -> int:
        return len(self.read_all())
