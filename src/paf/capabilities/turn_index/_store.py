"""Append-only flat-file index store for turn summaries.

The turn index is *derived* data: it is NOT part of pydantic-ai's run
history. It lives beside it, in its own append-only store, and each entry is
keyed to a full turn in pydantic-ai persistence *by reference* -- the
(conversation_id, run_id, turn_id) triple that `SnapshotHistorySource` can
resolve back to the original messages.

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
  - Appends are single atomic line-write (open 'a', one line) -- no partial
    record can corrupt the store, and it stays valid for a dump/diff even
    mid-run.

Layout (flat-file-first, inspectable/editable/diffable, no opaque DB)::

    <root>/
      <conversation_id>/
        index.jsonl     # append-only journal; one JSON object per line
        meta.json       # convenience counters only (rewritten in place); NOT authoritative

Each `index.jsonl` line::

    {"conversation_id", "run_id", "turn_id", "ts", "summary", "hash"}
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _by_conversation(root: Path, conversation_id: str | None) -> Path:
    """Return the directory for a conversation, creating it.

    `None`/empty conversations sort under a reserved `__default__` namespace
    so unlabelled runs still get a home without colliding with a real id.
    """
    segment = conversation_id if conversation_id else '__default__'
    safe = segment.replace('/', '_').replace('\\', '_')
    directory = root / safe
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _content_hash(conversation_id: str | None, run_id: str, turn_id: int, summary: str) -> str:
    """Stable content hash of a single index entry (used for idempotent dedup)."""
    return hashlib.sha256(
        f'{conversation_id}|{run_id}|{turn_id}|{summary}'.encode()
    ).hexdigest()


@dataclass(kw_only=True)
class TurnIndexEntry:
    """One append-only one-line summary, keyed to a full turn in pyai persistence."""

    conversation_id: str | None
    run_id: str
    turn_id: int
    ts: str
    summary: str
    hash: str

    def key(self) -> tuple[str | None, str, int]:
        return (self.conversation_id, self.run_id, self.turn_id)

    @property
    def ref(self) -> str:
        """Stable dereference handle, idiom `read_turn('{run_id}:{turn_id}')`."""
        return f'{self.run_id}:{self.turn_id}'


class TurnIndexStore:
    """Append-only JSONL store of one-line turn summaries.

    This is the capability's OWN persistence (the derived index). It never
    stores full turns -- the (conversation, run, turn_id) key resolves through
    `HistorySource` back into pydantic-ai persistence.
    """

    def __init__(self, directory: str | Path = '.turn-index') -> None:
        """Append-only one-line-turn index store.

        Defaults to `.turn-index` for a standalone store, but `paf` always passes
        the configured `data/.turn-index` explicitly (see `paf.config.turn_index_dir`).
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, conversation_id: str | None) -> Path:
        return _by_conversation(self.directory, conversation_id) / 'index.jsonl'

    def _meta_path(self, conversation_id: str | None) -> Path:
        return _by_conversation(self.directory, conversation_id) / 'meta.json'

    def append(
        self,
        *,
        conversation_id: str | None,
        run_id: str,
        turn_id: int,
        summary: str,
    ) -> TurnIndexEntry:
        """Append one entry atomically (single line-append). Returns the entry."""
        entry = TurnIndexEntry(
            conversation_id=conversation_id,
            run_id=run_id,
            turn_id=turn_id,
            ts=_utcnow().isoformat(),
            summary=summary,
            hash=_content_hash(conversation_id, run_id, turn_id, summary),
        )
        path = self._path(conversation_id)
        line = json.dumps(
            {
                'conversation_id': conversation_id,
                'run_id': run_id,
                'turn_id': turn_id,
                'ts': entry.ts,
                'summary': summary,
                'hash': entry.hash,
            },
            ensure_ascii=False,
        )
        with path.open('a', encoding='utf-8') as fh:
            fh.write(line + '\n')
            fh.flush()
            os.fsync(fh.fileno())
        self._bump_meta(conversation_id)
        return entry

    def read_all(self, conversation_id: str | None = None) -> list[TurnIndexEntry]:
        """Read every entry across all conversations (default), or one conversation."""
        entries: list[TurnIndexEntry] = []
        if conversation_id is None:
            for sub in self.directory.iterdir():
                if sub.is_dir():
                    entries.extend(self._read_file(sub / 'index.jsonl'))
            entries.sort(key=lambda e: e.ts)
            return entries
        return self._read_file(self._path(conversation_id))

    def latest_for(self, conversation_id: str | None, run_id: str, turn_id: int) -> TurnIndexEntry | None:
        """Resolve a key to its most recent entry (append-only correctness view)."""
        entries = [
            e
            for e in self._read_file(self._path(conversation_id))
            if e.run_id == run_id and e.turn_id == turn_id
        ]
        if not entries:
            return None
        return entries[-1]

    def has(self, conversation_id: str | None, run_id: str, turn_id: int) -> bool:
        """Whether a turn already has an index entry (idempotency for the writer)."""
        return self.latest_for(conversation_id, run_id, turn_id) is not None

    def count(self, conversation_id: str | None = None) -> int:
        return len(self.read_all(conversation_id))

    def _read_file(self, path: Path) -> list[TurnIndexEntry]:
        if not path.exists():
            return []
        entries: list[TurnIndexEntry] = []
        with path.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entries.append(
                    TurnIndexEntry(
                        conversation_id=data['conversation_id'],
                        run_id=data['run_id'],
                        turn_id=data['turn_id'],
                        ts=data['ts'],
                        summary=data['summary'],
                        hash=data['hash'],
                    )
                )
        return entries

    def _bump_meta(self, conversation_id: str | None) -> None:
        """Rewrite meta.json (convenience counters only; the journal is authoritative)."""
        path = self._meta_path(conversation_id)
        entries = self._read_file(self._path(conversation_id))
        meta = {
            'conversation_id': conversation_id,
            'count': len(entries),
            'updated': _utcnow().isoformat(),
        }
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, path)