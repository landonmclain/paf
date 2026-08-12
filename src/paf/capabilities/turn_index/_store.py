"""Append-only turn index store, backed by the `agent_foundation` storage seam.

The turn index is *derived* data: it is NOT part of pydantic-ai's run
history. It lives beside it, in its own append-only store, and each entry is
keyed to a full turn in pydantic-ai persistence *by reference* -- the
(conversation_id, run_id, turn_id) triple that `SnapshotHistorySource` can
resolve back to the original messages.

This store used to be a raw-`Path` JSONL writer. It is now a **thin adapter over
the `agent_foundation.storage.StorageService` seam** (M2 wiring, 2026-08-08):
it routes every append through `service.append(kind, scan_root, record)` and
every read through `service.read_rows(...)`, so the turn index is just one
`kind` ("turn_index") under whatever storage medium the consumer picked. The
ref model, schema, and call sites are unchanged; only the I/O went through the
seam. This is what lets a background worker (Background-Queue-Storage-Spec §4b)
share the same store instance as the capability -- one seam, one home for the
journal, regardless of medium.

Schema (structured, 2026-08-08): each row carries the flat `summary` (the
in-context one-liner, backward-compatible) PLUS optional structured fields
`key_terms` and `outcome` produced by the real LLM summarizer. `search_index`
matches across `summary` AND `key_terms` -- the whole point of splitting the
record (the "Postgres vs 'the database'" case: the sentence phrases it
abstractly, `key_terms` carries the searchable vocabulary).

The ref model changed (2026-08-08, per-turn indexing):

  OLD  (run_id, message_index) -- a message_index keyed into `run_history`
        rots as the run grows: the k-th user turn is at a different position
        once later messages arrive, so a ref written early becomes stale.

  NEW  (run_id, turn_id) -- `turn_id` is the ORDINAL of the turn's user-prompt
        anchor within the run (`0` for the first user turn, `1` for the next,
        ...). The same messages always yield the same ordinals, so the ref is
        stable for the life of the run AND `read_turn` re-finds the anchor by
        walking `run_history(run_id)` to the turn_id-th user prompt. Idiom is
        `read_turn('{run_id}:{turn_id}')`.

Design contract (append-only):
  - Entries are never edited or deleted. A wrong summary is *corrected* by
    appending a new entry carrying the corrected text under the same key; the
    stale entry stays for the audit trail. Consumers resolve a key to its
    latest entry.
  - Idempotent by construction: re-summarizing an already-indexed turn yields
    a content-identical duplicate unless the (LLM-derived) text drifted.
    Consumers may dedup on content hash.
  - Appends are single atomic line-writes (via `StorageService.append`) -- no
    partial record can corrupt the store, and it stays valid for a dump/diff
    even mid-run.

Layout (now through the seam's flat-file medium; see `FlatFileStorage`):

    <base>/<hash(scan_root)>/turn_index.jsonl     # append-only journal; one row per turn
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agent_foundation.storage import FlatFileStorage, StorageService

# The `kind` this store reads/writes under the storage seam. Stable per consumer
# family, so a SQLite/object-store backend can route rows the same way.
_KIND = "turn_index"

# Default logical root the journal keys under, when the consumer supplies a
# directory (back-compat with `TurnIndexStore(directory=...)` / paf.config).
# Only used to derive a stable scan_root for the seam; the actual on-disk path
# is `<FlatFileStorage base>/<hash(this)>/turn_index.jsonl`.
_DEFAULT_ROOT = Path(".turn-index")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _content_hash(conversation_id: str | None, run_id: str, turn_id: int, summary: str) -> str:
    """Stable content hash of a single index entry (used for idempotent dedup)."""
    return hashlib.sha256(
        f'{conversation_id}|{run_id}|{turn_id}|{summary}'.encode()
    ).hexdigest()


def _scan_root_for(directory: str | Path | None) -> Path:
    """Normalize the legacy directory arg into a stable seam scan_root."""
    return Path(directory) if directory is not None else _DEFAULT_ROOT


@dataclass(kw_only=True)
class TurnIndexEntry:
    """One append-only one-line summary, keyed to a full turn in pyai persistence.

    `summary` is the flat in-context one-liner (always present, backward-compatible).
    `key_terms` / `outcome` are the OPTIONAL structured fields from the real LLM
    summarizer; `key_terms` are what `search_index` also matches, so a turn whose
    sentence phrases things abstractly is still found by its vocabulary.
    """

    conversation_id: str | None
    run_id: str
    turn_id: int
    ts: str
    summary: str
    hash: str
    key_terms: list[str] | None = None
    outcome: str | None = None
    session_id: str | None = None

    def key(self) -> tuple[str | None, str, int]:
        return (self.conversation_id, self.run_id, self.turn_id)

    @property
    def ref(self) -> str:
        """Stable dereference handle, idiom `read_turn('{run_id}:{turn_id}')`."""
        return f'{self.run_id}:{self.turn_id}'


class TurnIndexStore:
    """Append-only turn index, backed by a `StorageService` seam.

    This is the capability's OWN persistence (the derived index). It never
    stores full turns -- the (conversation, run, turn_id) key resolves through
    `HistorySource` back into pydantic-ai persistence.

    Accepts either an explicit `StorageService` (preferred -- the shared seam,
    so a background worker and the capability write the same journal) or a
    directory `Path`/`str` (back-compat: auto-wraps a `FlatFileStorage` and
    keys under that base). `scan_root` for the seam defaults to the legacy
    directory when given, else `data/.turn-index`-style default; rows carry a
    `conversation_id` field so per-conversation scoping is preserved in the
    adapter, and `read_all(None)` spans conversations.
    """

    def __init__(
        self,
        storage: StorageService | str | Path = _DEFAULT_ROOT,
        *,
        root: str | Path | None = None,
    ) -> None:
        if isinstance(storage, (str, Path)):
            base = Path(storage)
            self.service: StorageService = FlatFileStorage(base_dir=base)
            self.scan_root = _scan_root_for(root)
        else:
            self.service = storage
            self.scan_root = _scan_root_for(root)

    # --- seam plumbing ----------------------------------------------------

    def _row(self, entry: TurnIndexEntry) -> dict:
        row: dict = {
            'conversation_id': entry.conversation_id,
            'run_id': entry.run_id,
            'turn_id': entry.turn_id,
            'ts': entry.ts,
            'summary': entry.summary,
            'hash': entry.hash,
        }
        if entry.key_terms:
            row['key_terms'] = entry.key_terms
        if entry.outcome:
            row['outcome'] = entry.outcome
        if entry.session_id:
            row['session_id'] = entry.session_id
        return row

    def _rows(self) -> list[TurnIndexEntry]:
        """Parse every appended row back into entries (all conversations)."""
        entries = []
        for data in self.service.read_rows(_KIND, self.scan_root):
            entries.append(
                TurnIndexEntry(
                    conversation_id=data.get('conversation_id'),
                    run_id=data['run_id'],
                    turn_id=data['turn_id'],
                    ts=data['ts'],
                    summary=data['summary'],
                    hash=data['hash'],
                    key_terms=data.get('key_terms'),
                    outcome=data.get('outcome'),
                    session_id=data.get('session_id'),
                )
            )
        return entries

    # --- append ------------------------------------------------------------

    def append(
        self,
        *,
        conversation_id: str | None,
        run_id: str,
        turn_id: int,
        summary: str,
        key_terms: list[str] | None = None,
        outcome: str | None = None,
        session_id: str | None = None,
    ) -> TurnIndexEntry:
        """Append one entry atomically (via the seam's single-row append)."""
        entry = TurnIndexEntry(
            conversation_id=conversation_id,
            run_id=run_id,
            turn_id=turn_id,
            ts=_utcnow().isoformat(),
            summary=summary,
            hash=_content_hash(conversation_id, run_id, turn_id, summary),
            key_terms=key_terms,
            outcome=outcome,
            session_id=session_id,
        )
        self.service.append(_KIND, self.scan_root, self._row(entry))
        return entry

    # --- reads -------------------------------------------------------------

    def read_all(self, conversation_id: str | None = None) -> list[TurnIndexEntry]:
        """Read every entry across all conversations (default), or one conversation."""
        entries = self._rows()
        entries.sort(key=lambda e: e.ts)
        if conversation_id is None:
            return entries
        return [e for e in entries if e.conversation_id == conversation_id]

    def latest_for(self, conversation_id: str | None, run_id: str, turn_id: int) -> TurnIndexEntry | None:
        """Resolve a key to its most recent entry (append-only correctness view)."""
        matches = [
            e
            for e in self._rows()
            if e.conversation_id == conversation_id and e.run_id == run_id and e.turn_id == turn_id
        ]
        if not matches:
            return None
        return matches[-1]

    def has(self, conversation_id: str | None, run_id: str, turn_id: int) -> bool:
        """Whether a turn already has an index entry (idempotency for the writer)."""
        return self.latest_for(conversation_id, run_id, turn_id) is not None

    def count(self, conversation_id: str | None = None) -> int:
        return len(self.read_all(conversation_id))
