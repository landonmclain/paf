"""Append-only durable-facts journal, sitting beside ``MEMORY.md``.

This is the **source of truth** of the three-layer memory design (journal ->
index -> render): a write-once, append-only JSONL log of *what was seen and
when*, each record a durable fact extracted from a chunk of source
conversation. The relational concerns — dedup, recency-supersession, stale
soft-archive, tier assignment — deliberately live downstream (the index and the
render pass), NOT here. The journal never edits, compacts, dedups, or reorders
history; a fact seen again (possibly updated) in a later chunk legitimately
appends a second record with a later ``born_on``, and "newest wins" is decided
downstream.

Design decisions:

- **Record shape** — immutable ``{type, fact, tier, born_on, source_ref}`` (v1);
  v2 rows add ``{v, scope, project, source}`` (see :class:`MemoryRecord`).
  ``id`` is the 1-based JSONL line number (stable forever because the journal is
  never rewritten; no stored id field, no counter -- just position).
- **No ``entity`` field** — the subject lives inline in ``fact`` prose; ``fact``
  is the sole searchable axis (BM25 search over fact text at render/query).
- **Dump semantics, no write-time dedup** — the journal extracts what it sees in
  a chunk regardless of what is already there. The ONLY write-side safety is
  atomicity (a chunk's records are appended as one block; a crash mid-block
  never leaves a half-written line).
- **Self-contained** — atomic line-writes via the shared ``_storage``
  ``_append_jsonl_atomic`` helper (read + append + ``os.replace``); no
  storage-service seam of its own.
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

from paf.capabilities._storage import _append_jsonl_atomic

_JOURNAL_FILENAME = "memory-journal.jsonl"

# Semantic memory types (v2). The extractor writes only these; the list is
# first-match ordered in its prompt and ends in "skip" — there is no catch-all.
WRITE_TYPES: frozenset[str] = frozenset(
    {"identity", "rule", "preference", "decision", "context", "event"}
)
# v1 types still present in old journal rows (including the removed legacy
# reviewer's memory_merge rows). Readable forever; never written by the extractor.
LEGACY_TYPES: frozenset[str] = frozenset(
    {"identity", "preference", "fact", "event", "task", "current"}
)
# Every type a journal row may carry (validation on read / reconstruct).
MEMORY_TYPES: frozenset[str] = WRITE_TYPES | LEGACY_TYPES

# Permanence tiers. v1 rows carry a model-chosen tier; v2 rows carry a tier
# DERIVED in code from type + scope (see derive_tier) — a bridge so the current
# tier-grouped render keeps working until the scope-aware render replaces it.
# t1 = stable person / user-wide; t2 = project knowledge; t3 = events.
MEMORY_TIERS: frozenset[str] = frozenset({"t1", "t2", "t3"})

# v2: does the fact matter in a conversation about a DIFFERENT project?
#   global  -> yes: about the user everywhere (identity, personal preferences, setup)
#   project -> no: belongs to the conversation's project (``project`` names it;
#              None = "unfiled", filed later once the conversation is tagged)
MEMORY_SCOPES: frozenset[str] = frozenset({"global", "project"})

# v2: how the fact came to be known.
MEMORY_SOURCES: frozenset[str] = frozenset({"stated", "observed", "inferred"})

RECORD_VERSION = 2


def derive_tier(type: str, scope: str) -> str:
    """The tier bridge for v2 rows: events age out (t3); global facts are
    always-on (t1); project facts are project knowledge (t2)."""
    if type == "event":
        return "t3"
    return "t1" if scope == "global" else "t2"


class MemoryRecord:
    """One immutable journal record: a durable fact, append-only.

    Fields (all rows):
      type       -- one of ``MEMORY_TYPES``; v2 rows use ``WRITE_TYPES``
      fact       -- the durable statement; subject lives inline (no entity field)
      tier       -- one of ``MEMORY_TIERS``; model-chosen on v1 rows, derived
                    (``derive_tier``) on v2 rows. REQUIRED.
      born_on    -- ISO datetime of the source message, stored as UTC
      source_ref -- ``{run_id}:{turn_id}`` turn addressing, read_turn-resolvable

    v2-only fields (``None`` on v1 rows):
      scope      -- one of ``MEMORY_SCOPES`` (global | project)
      project    -- project slug (code-stamped from the conversation); ``None``
                    for global facts and for unfiled project facts
      source     -- one of ``MEMORY_SOURCES`` (stated | observed | inferred)
      v          -- row version: 1 for legacy rows, 2 for rows with the fields above

    ``id`` is NOT stored -- it is the record's 1-based JSONL line number, assigned
    by :meth:`JournalStore.append`/``.append_many``.
    """

    __slots__ = (
        "type",
        "fact",
        "tier",
        "born_on",
        "source_ref",
        "scope",
        "project",
        "source",
        "v",
    )

    def __init__(
        self,
        *,
        type: str,
        fact: str,
        tier: str,
        born_on: str | datetime | None = None,
        source_ref: str,
        scope: str | None = None,
        project: str | None = None,
        source: str | None = None,
        v: int | None = None,
    ) -> None:
        if type not in MEMORY_TYPES:
            raise ValueError(
                f"invalid memory type {type!r}; expected one of {sorted(MEMORY_TYPES)}"
            )
        if tier not in MEMORY_TIERS:
            raise ValueError(
                f"invalid memory tier {tier!r}; expected one of {sorted(MEMORY_TIERS)}"
            )
        if scope is not None and scope not in MEMORY_SCOPES:
            raise ValueError(
                f"invalid scope {scope!r}; expected one of {sorted(MEMORY_SCOPES)}"
            )
        if source is not None and source not in MEMORY_SOURCES:
            raise ValueError(
                f"invalid source {source!r}; expected one of {sorted(MEMORY_SOURCES)}"
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
        self.scope = scope
        self.project = project or None
        self.source = source
        self.v = v if v is not None else (RECORD_VERSION if scope is not None else 1)

    # --- serialization -----------------------------------------------------

    def to_row(self) -> dict:
        if self.v < RECORD_VERSION:
            return {
                "type": self.type,
                "fact": self.fact,
                "tier": self.tier,
                "born_on": self.born_on,
                "source_ref": self.source_ref,
            }
        # v2: always write scope/project (null = global or unfiled) explicitly.
        return {
            "v": self.v,
            "type": self.type,
            "fact": self.fact,
            "scope": self.scope,
            "project": self.project,
            "source": self.source,
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
            scope=row.get("scope"),
            project=row.get("project"),
            source=row.get("source"),
            v=row.get("v", 1),
        )

    def with_tier(self, tier: str) -> MemoryRecord:
        """A copy at a different tier (reviewer demotion), keeping every field."""
        row = self.to_row()
        row["tier"] = tier
        return MemoryRecord.from_row(row)

    def with_fields(
        self,
        *,
        type: str | None = None,
        scope: str | None = None,
        project: str | None = None,
    ) -> MemoryRecord:
        """A re-filed copy (``reclassify`` override): new type/scope/project,
        tier re-derived; the fact, date, and source stay. A legacy row comes
        out v2-shaped (it now has a scope)."""
        row = self.to_row()
        row["type"] = type or self.type
        row["scope"] = scope or self.scope
        row["project"] = project if row["scope"] == "project" else None
        row["source"] = self.source
        if row["scope"]:
            row["tier"] = derive_tier(row["type"], row["scope"])
            row["v"] = RECORD_VERSION
        return MemoryRecord.from_row(row)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryRecord):
            return NotImplemented
        return self.to_row() == other.to_row()

    def __repr__(self) -> str:
        return f"MemoryRecord({self.to_row()!r})"


def _normalize_born_on(value: str | datetime | None) -> str:
    """Normalize ``born_on`` to a UTC ISO-8601 string (string out for dump/read ease).

    Accepts a ``datetime`` (naive is assumed UTC per the corpus-clock convention)
    or an ISO string. Offset-bearing values (the runtime stamps messages in the
    user's zone, e.g. ``-05:00``) are CONVERTED to UTC: recency ordering sorts
    ``born_on`` lexically, which is only chronological if every value shares
    one offset.

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
        return dt.astimezone(UTC).isoformat()
    # String: try to normalize, else keep as-is (ISO already sorts lexically).
    text = value.strip()
    if not text:
        return ""
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC).isoformat()
    except ValueError:
        return text


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

    # --- appends -----------------------------------------------------------

    def append(
        self,
        *,
        type: str,
        fact: str,
        tier: str,
        born_on: str | datetime,
        source_ref: str,
        scope: str | None = None,
        project: str | None = None,
        source: str | None = None,
    ) -> MemoryRecord:
        """Append one record; returns it. Its id is the new line number.

        Passing ``scope`` makes it a v2 row (scope/project/source); without it
        the row keeps the v1 shape (the legacy reviewer's merges)."""
        record = MemoryRecord(
            type=type,
            fact=fact,
            tier=tier,
            born_on=born_on,
            source_ref=source_ref,
            scope=scope,
            project=project,
            source=source,
        )
        return self.append_record(record)

    def append_record(self, record: MemoryRecord) -> MemoryRecord:
        """Append an already-constructed :class:`MemoryRecord`."""
        with self._lock:
            _append_jsonl_atomic(self._path, [record.to_row()])
        return record

    def append_many(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        """Append ``records`` as ONE atomic block (a chunk's extraction).

        Returns the records (unchanged). All-or-nothing: a crash mid-block leaves
        the journal at the prior complete state — no partial chunk.
        """
        if not records:
            return []
        with self._lock:
            _append_jsonl_atomic(self._path, [r.to_row() for r in records])
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
