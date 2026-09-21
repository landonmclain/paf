"""Derived, deterministic memory index: journal -> index-of-surviving-facts.

The three-layer memory design has the append-only journal (source of truth),
the render (``MEMORY.md``, what the answerer reads), and this **index** — a
projection that answers *"which journal line is the live copy of each fact, and
which lines are its deduped predecessors"* without re-scanning the whole
journal every time.

Why it exists: the journal is write-once with dump semantics ("no write-time
dedup"), so repeated facts legitimately append N copies. There was no way to
say "these 6 lines are really one fact; line 48 is the live one" short of
re-reading every record. This index is that verdict, and it is:

- **Derived** — regenerable from the journal alone, so it can't drift into a
  parallel source of truth. If you delete it, the next fold regenerates it.
- **Deterministic** — pure code, no model, no LLM pass.
- **Incremental** — a watermark records the last journal line folded in. Each
  fold processes only the newly-appended lines, each via an O(1) lookup keyed
  on the normalized fact, NOT a re-scan of all history. The one exception is
  the very first build over a journal that already has history, which is a
  full pass (:func:`build_index`).

Keep-winner rule: within a dedup group, the winner is the record with the
**highest tier** (t1 > t2 > t3), breaking ties by **most-recent ``born_on``**,
finally by append order (higher line# = newer).

Dedup scoping: folds dedup on **exact** and **normalized-token-set** equality
(catches verbatim + re-ordered repeats at O(1)). Paraphrase bloat is NOT
auto-merged here — the reviewer merges duplicates deliberately (see
``memory_merge``).

Layout: the index lives beside ``MEMORY.md`` and the journal::

    <memory dir>/
      MEMORY.md                # the render (presentation only)
      memory-journal.jsonl     # append-only fact stream (source of truth)
      memory-index.json        # THIS module: derived index of surviving facts
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime

from paf.capabilities._storage._atomic import write_atomic

from ._journal import JournalStore, MemoryRecord

_INDEX_FILENAME = "memory-index.json"
_SCHEMA = 1

# Tier priority for the keep-winner rule: higher index = "wins" a dedup group.
_TIER_PRIORITY = {"t1": 3, "t2": 2, "t3": 1}

# Stopwords too generic to discriminate facts. Anything below 3 chars is also
# dropped.
_STOP = frozenset(
    "a an the and or but of to in for on with is are was were be been "
    "has have had do does did it its this that these those about into from "
    "using uses use used user via by as at not no yes i you we they he she".split()
)


# --------------------------------------------------------------------------
# derived keys (the incremental O(1) lookup)
# --------------------------------------------------------------------------


def _tokens(fact: str) -> set[str]:
    """Lowercased alnum tokens minus stopwords, dropping length<=2 (deterministic)."""
    raw = re.findall(r"[a-z0-9]+", fact.lower())
    return {t for t in raw if t not in _STOP and len(t) > 2}


def fact_keys(fact: str) -> dict[str, str]:
    """Return the lookup keys for a fact: exact and normalized-token-set.

    ``exact`` is the raw lowercased/whitespace-collapsed statement — catches
    byte-identical repeats. ``tokens`` is the sorted stopword-filtered token
    set — catches the same statement re-worded/re-ordered (word order shuffled,
    articles swapped) even when punctuation/case differs. Both are O(1) hashes.
    """
    norm = " ".join(fact.split()).lower()
    tokset = _tokens(fact)
    return {
        "exact": hashlib.sha256(norm.encode("utf-8")).hexdigest(),
        "tokens": hashlib.sha256(" ".join(sorted(tokset)).encode("utf-8")).hexdigest(),
    }


def _winner_better(
    candidate_line: int,
    cand_tier: str,
    cand_born,
    cur_line: int,
    cur_tier: str,
    cur_born,
) -> bool:
    """Keep-winner rule: highest tier, then most-recent born_on, then newest.

    Operates on plain (tier, born_on) so it needs no reconstructed record.
    """
    if _TIER_PRIORITY[cand_tier] != _TIER_PRIORITY[cur_tier]:
        return _TIER_PRIORITY[cand_tier] > _TIER_PRIORITY[cur_tier]
    cb = _born_ts(cand_born)
    ub = _born_ts(cur_born)
    if cb != ub:
        return cb > ub
    # tie -> the later-append record (higher line#) is the newest copy
    return candidate_line > cur_line


def _born_ts(born_on: str | datetime | None):
    """Normalized comparable timestamp for recency tie-break (None -> epoch-)."""
    if born_on is None:
        return datetime.min.replace(tzinfo=UTC)
    if isinstance(born_on, datetime):
        return born_on
    try:
        return datetime.fromisoformat(born_on)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


# --------------------------------------------------------------------------
# the index document
# --------------------------------------------------------------------------


class MemoryIndex:
    """In-memory view of the derived index (one winner per surviving fact)."""

    __slots__ = ("entries", "watermark", "_by_key")

    def __init__(self, entries: list[dict] | None = None, watermark: int = 0):
        self.entries: list[dict] = entries if entries is not None else []
        self.watermark: int = watermark
        self._by_key: dict[str, dict] = {}
        self._rebuild_key_index()

    # -- serialization -----------------------------------------------------

    @classmethod
    def load(cls, directory: str | os.PathLike) -> MemoryIndex:
        """Load ``memory-index.json`` if present; else a fresh (empty) index."""
        path = os.path.join(os.fspath(directory), _INDEX_FILENAME)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                entries = data.get("entries", [])
                # tolerate a future bumped schema by only reading what we know
                watermark = data.get("watermark", len(entries))
                return cls(entries=entries, watermark=watermark)
            except (json.JSONDecodeError, OSError, ValueError):
                # corrupt/missing -> regenerate from scratch next fold
                return cls()
        return cls()

    def save(self, directory: str | os.PathLike) -> None:
        """Atomically write the index to ``<directory>/memory-index.json``."""
        doc = {
            "schema": _SCHEMA,
            "watermark": self.watermark,
            "entries": self.entries,
        }
        write_atomic(
            os.path.join(os.fspath(directory), _INDEX_FILENAME),
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        )

    # -- lookups -----------------------------------------------------------

    def _rebuild_key_index(self) -> None:
        self._by_key = {}
        for e in self.entries:
            for k in (e.get("key_exact"), e.get("key_tokens")):
                if k:
                    self._by_key[k] = e

    def _lookup(self, keys: dict[str, str]) -> dict | None:
        for k in (keys["exact"], keys["tokens"]):
            hit = self._by_key.get(k)
            if hit is not None:
                return hit
        return None

    # -- incremental -------------------------------------------------------

    def fold(
        self, line_number: int, record: MemoryRecord, directory: str | os.PathLike
    ) -> None:
        """Fold ONE new journal line into the index (O(1)-ish, incremental).

        If a surviving winner matches this record's key, the group absorbs it
        (keep-winner rule may promote/move the winner). Otherwise it becomes a
        new winning fact. Advances ``watermark`` to ``line_number``.
        """
        keys = fact_keys(record.fact)
        winner = self._lookup(keys)
        if winner is None:
            self.entries.append(self._entry(line_number, record, deduped=[]))
            self._by_key[keys["exact"]] = self.entries[-1]
            self._by_key[keys["tokens"]] = self.entries[-1]
        else:
            self._absorb(winner, line_number, record)
        self.watermark = line_number
        # persist the fold immediately so a crash never loses a consumed span
        self.save(directory)

    def _absorb(self, winner: dict, line_number: int, record: MemoryRecord) -> None:
        if _winner_better(
            line_number,
            record.tier,
            record.born_on,
            int(winner["current_line"]),
            winner["tier"],
            winner.get("born_on"),
        ):
            # the new record wins -> old winner becomes a deduped predecessor
            old_line = int(winner["current_line"])
            winner["deduped_lines"] = [old_line, *winner.get("deduped_lines", [])]
            self._set_winner(winner, line_number, record)
        else:
            # existing winner stands -> new record is just another deduped copy
            deduped = winner.setdefault("deduped_lines", [])
            if line_number not in deduped:
                deduped.append(line_number)

    def _set_winner(self, entry: dict, line_number: int, record: MemoryRecord) -> None:
        entry["current_line"] = line_number
        entry["tier"] = record.tier
        entry["tier_rank"] = _TIER_PRIORITY[record.tier]
        entry["born_on"] = record.born_on
        entry["source_ref"] = record.source_ref

    @staticmethod
    def _entry(line_number: int, record: MemoryRecord, deduped: list[int]) -> dict:
        keys = fact_keys(record.fact)
        return {
            "key_exact": keys["exact"],
            "key_tokens": keys["tokens"],
            "current_line": line_number,
            "deduped_lines": deduped,
            "tier": record.tier,
            "tier_rank": _TIER_PRIORITY[record.tier],
            "born_on": record.born_on,
            "source_ref": record.source_ref,
        }


# --------------------------------------------------------------------------
# first build (full pass over the whole journal)
# --------------------------------------------------------------------------


def build_index(directory: str | os.PathLike) -> int:
    """Full (re)build of the index from the entire journal. Returns entry count.

    Reads every journal record and folds them in append order via the
    exact/token dedup rule. Idempotent: safe to re-run any time (cheap,
    deterministic), e.g. to repair a deleted/corrupt index.
    """
    journal = JournalStore(directory)
    records = journal.read_all()  # list[(line_no, MemoryRecord)] in append order
    idx = MemoryIndex(watermark=0)

    # --- pass 1: fold every record via the incremental rule (exact/token) ---
    for line_no, rec in records:
        idx.fold(
            line_no, rec, directory
        )  # saves per fold (a bit chatty, fine for a build)

    # Fuzzy cluster collapse (pass 2) was REMOVED (2026-09-21): it conflated
    # distinct facts that merely share tokens — e.g. two different personal-best
    # 5K times (27:12 vs 25:50) were treated as one duplicate and the newer value
    # was dropped from the render, losing the gold answer from injected memory.
    # Only the exact/token dedup above (pass 1) is kept; paraphrase bloat is left
    # to the reviewer to merge deliberately.

    idx.watermark = records[-1][0] if records else 0
    idx.save(directory)
    return len(idx.entries)


# --------------------------------------------------------------------------
# top-level entry points (used by the write path and tests)
# --------------------------------------------------------------------------


def update_index(directory: str | os.PathLike) -> int:
    """Incrementally fold any new journal lines into the index. Returns entries.

    Reads the index watermark; if it's behind the journal tail, folds only the
    new lines (no full re-scan). If the index file is absent or corrupt, falls
    back to a full rebuild. Safe to call after every render. First-ever call on
    a journal that already has history performs the one-time full build.
    """
    journal = JournalStore(directory)
    records = journal.read_all()
    tail = records[-1][0] if records else 0

    idx = MemoryIndex.load(directory)
    if not idx.entries:
        # no existing index -> full build
        return build_index(directory)
    if idx.watermark >= tail:
        return len(idx.entries)  # nothing new

    # incremental: fold only lines strictly after the watermark
    for line_no, rec in records:
        if line_no > idx.watermark:
            idx.fold(line_no, rec, directory)
    return len(idx.entries)
