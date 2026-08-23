"""Memory organizer: the index-maintaining sibling of the consolidation owner.

The three-layer memory design (journal -> *maintenance* -> index -> render)
splits two LLM sub-agents that used to be conflated:

- **Consolidation owner** (``_consolidate.py``) — the *dump* writer. Folds an
  evicted conversation slice into the **append-only journal** (one immutable
  record per durable fact). It never edits, moves, or deletes anything.
- **Memory organizer** (THIS module) — the *maintenance* writer. Reads the new
  journal tail ``(watermark, new_end]`` and reconciles it into the **curated
  index** (``memory-index.json``): classifying each journal record as
  NEW / CONFIRM / SUPERSEDE against the current entries, applying the bump-on-
  change and prune contracts, and advancing the watermark.

The index is the canonical organized view — hand-editable JSON, inspectable and
diffable. The injected ``MEMORY.md`` is a *pure render* of the index (produced
elsewhere by ``render_index``); the index itself is data, not presentation.

Ownership model
---------------
Same split as the consolidation owner: the assistant carries ``writable=False``
(relevant tools are read-only / it has none for the index). The organizer agent
carries the writable index toolset and is the **sole writer** of the index.
The journal stays owner-only: no organizer tool ever touches the journal.

Model selection
---------------
The organizer model is read from ``Settings.organize_model`` (``ORGANIZE_MODEL``
env / .env) and **defaults to the same model as the consolidation owner**
(``consol_model`` / its cheap default) — a separate leaf agent that can later be
moved to a higher-level maintenance model by setting ``ORGANIZE_MODEL``. It is
deliberately not hardcoded and not the spec'd ``assistant.yaml`` subject model,
mirroring the owner (see ``_consolidate.consolidate_model``).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import cast

from pydantic_ai import Agent, ModelRetry, ModelSettings

from paf.config import settings

from ._atomic import read_json, write_json_atomic
from ._journal import MEMORY_TYPES, JournalStore, MemoryRecord, _normalize_born_on
from ._toolset import MemoryWriteResult

logger = logging.getLogger(__name__)

# Index file (sits beside `memory-journal.jsonl` in the same memory-scope dir).
_INDEX_FILENAME = "memory-index.json"
# v4 adds a persisted `next_id` allocator cursor to the wrapper. The locked
# contract "entry ids are never reused" is only honor-able if the monotonic
# counter survives PRUNES (which remove entries, so a derived base recomputed
# from the current entry list would reuse a pruned id). v3 reused ids after a
# prune — a real contract violation — so the cursor is now stored.
_INDEX_VERSION = 4

# Minimum number of new journal lines that justify a run-end auto-trigger of the
# organizer (see Phase 2 trigger). Tunable via `LME_ORGANIZE_BATCH` env.
_DEFAULT_ORGANIZE_BATCH = 25


# ---------------------------------------------------------------------------
# Schema: IndexEntry + wrapper
# ---------------------------------------------------------------------------


@dataclass
class IndexEntry:
    """One curated index entry — the organized view of a durable fact.

    Field contracts (LOCKED with Landon, 2026-08-17):

    - ``id``       -- stable, never reused (``mem-N``, N monotonic).
    - ``type``     -- one of ``MEMORY_TYPES`` (identity|preference|fact|event|
                      task|current).
    - ``subject``  -- index-anchored topic (coarse entity/theme). The organizer
                      reuses an existing subject string when a journal record
                      concerns it, else mints a new one; grouping is exact-string.
    - ``fact``     -- the curated statement (deduped/superseded to current value).
    - ``born_on``  -- original date the fact was first stated (provenance only,
                      never changes).
    - ``updated_on`` -- date the CURRENT value last changed (bump-on-change);
                      THE recency axis.
    - ``source_ref`` -- current single ``{run_id}:{turn_id}`` read_turn handle.
    - ``source``   -- original journal LINE numbers backing this entry (lineage);
                      grows append-only within an entry; provenance axis.
    """

    id: str
    type: str
    subject: str
    fact: str
    born_on: str
    updated_on: str
    source_ref: str
    source: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Copy the source list so a caller's list is never aliased/mutated.
        object.__setattr__(self, "source", list(self.source))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "subject": self.subject,
            "fact": self.fact,
            "born_on": self.born_on,
            "updated_on": self.updated_on,
            "source_ref": self.source_ref,
            "source": list(self.source),
        }

    @classmethod
    def from_dict(cls, data: dict) -> IndexEntry:
        return cls(
            id=data["id"],
            type=data["type"],
            subject=data["subject"],
            fact=data["fact"],
            born_on=data["born_on"],
            updated_on=data["updated_on"],
            source_ref=data["source_ref"],
            source=list(data.get("source", [])),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexEntry):
            return NotImplemented
        return self.to_dict() == other.to_dict()


# ---------------------------------------------------------------------------
# IndexStore: atomic load/save, monotonic id, watermark
# ---------------------------------------------------------------------------


class IndexStore:
    """Curated memory index stored as ``memory-index.json`` (flat JSON).

    ``directory`` is the memory-scope dir — the same one holding the journal
    (``memory-journal.jsonl``) and ``MEMORY.md``. The index file is
    ``<directory>/memory-index.json``.

    Crash safety: watermark + entries live in ONE atomic write
    (``write_json_atomic`` → staging + ``os.replace``), so an observer sees
    either the prior complete state or the new complete state — never a
    half-applied watermark with unwritten entries (or vice-versa). A re-run over
    an overlapping tail is idempotent (see ``reconcile``).
    """

    def __init__(self, directory: str | os.PathLike) -> None:
        self._dir = os.fspath(directory)
        self._path = os.path.join(self._dir, _INDEX_FILENAME)
        self._next_id = 1

    @property
    def path(self) -> str:
        return self._path

    @property
    def filename(self) -> str:
        return _INDEX_FILENAME

    # --- load / save ------------------------------------------------------

    def load(self) -> list[IndexEntry]:
        """Load current entries. Empty (not raising) if absent/corrupt."""
        raw = read_json(self._path)
        if not isinstance(raw, dict):
            return []
        entries = raw.get("entries", [])
        if not isinstance(entries, list):
            return []
        return [IndexEntry.from_dict(e) for e in entries if isinstance(e, dict)]

    def load_watermark(self) -> int:
        """The read cursor: journal line number already processed into the index."""
        raw = read_json(self._path)
        if not isinstance(raw, dict):
            return 0
        wm = raw.get("watermark", 0)
        return wm if isinstance(wm, int) else 0

    def load_next_id(self) -> int:
        """The persisted monotonic id cursor (survives prunes)."""
        raw = read_json(self._path)
        if not isinstance(raw, dict):
            return 1
        nid = raw.get("next_id")
        if isinstance(nid, int) and nid >= 1:
            return nid
        # v3 (or earlier) wrapper: derive from the largest visible entry id —
        # best effort; prunes in a v3 file may reuse an id, which v4 fixes.
        return _max_entry_num(raw.get("entries")) + 1

    def _advance_id_from(self, entries: list[IndexEntry]) -> None:
        """Seed the monotonic id allocator: persisted cursor if set, else derive.

        Derived-from-entries is ONLY a v3 fallback — the persisted ``next_id``
        is authoritative for v4+ so pruned ids are never reused (contract).
        """
        self._next_id = max(self.load_next_id(), _max_entry_num(entries) + 1)

    def _save(self, entries: list[IndexEntry], watermark: int, next_id: int | None = None) -> None:
        # Unless an explicit cursor is given, persist the FURTHER of the in-run
        # allocator (advanced by allocs this pass) and the persisted cursor (from
        # a prior pass). A fresh store re-saving without allocating must not
        # regress the persisted next_id.
        if next_id is None:
            next_id = max(self.load_next_id(), self._next_id)
        write_json_atomic(
            self._path,
            {
                "version": _INDEX_VERSION,
                "watermark": watermark,
                "next_id": next_id,
                "entries": [e.to_dict() for e in entries],
            },
        )

    def alloc_id(self) -> str:
        """Allocate the next monotonic ``mem-N`` id (never reused)."""
        n = self._next_id
        self._next_id += 1
        return f"mem-{n}"

    # --- persistence conveniences used by the reconcile orchestrator -------

    def read_all(self) -> list[IndexEntry]:
        """Entries in index order (same as ``load``)."""
        return self.load()

    def count(self) -> int:
        return len(self.load())


def _entry_id_num(entry_id: str | None) -> int:
    """Extract the numeric part of ``mem-N``; 0 if malformed/absent."""
    if isinstance(entry_id, str) and entry_id.startswith("mem-") and entry_id[4:].isdigit():
        return int(entry_id[4:])
    return 0


def _max_entry_num(entries) -> int:
    """Largest ``mem-N`` among ``entries`` (raw dicts or IndexEntrys); 0 if none."""
    greatest = 0
    for e in entries:
        eid = e.get("id") if isinstance(e, dict) else getattr(e, "id", None)
        num = _entry_id_num(eid)
        if num > greatest:
            greatest = num
    return greatest


# ---------------------------------------------------------------------------
# Classification model + deterministic apply helpers (the testable core)
# ---------------------------------------------------------------------------
#
# The organizer's *brain* is the LLM (deciding "is this the same subject as that
# entry and does it supersede it?" is semantic). But the *rules* for applying a
# decision are mechanical and deterministic — that is what this section holds,
# so the maintenance contracts are unit-testable WITHOUT a live LLM (mirroring
# how ``test_memory_journal.py`` pins the journal contract).


@dataclass
class NewDecision:
    """Create a NEW index entry from journal record ``line_no``."""

    line_no: int
    type: str
    subject: str
    fact: str
    born_on: str
    source_ref: str


@dataclass
class ConfirmDecision:
    """A journal record confirms an existing entry (same subject, same value).

    Grows the entry's ``source`` lineage with ``line_no``. Does NOT bump
    ``updated_on`` — the value didn't change, only its evidence.
    """

    entry_id: str
    line_no: int


@dataclass
class SupersedeDecision:
    """A journal record supersedes an existing entry (subject changed value).

    Bumps the target's ``updated_on`` (recency axis) and grows ``source``;
    ``fact``/``source_ref`` reflect the new current value. ``prune`` lists other
    entry ids that this new value renders obsolete — each is REMOVED from the
    index entirely (no tombstone; the journal remains the only growing thing).
    ``born_on`` is preserved (the original date of the fact lineage).
    """

    entry_id: str
    line_no: int
    fact: str
    source_ref: str
    updated_on: str
    prune: list[str] = field(default_factory=list)


# MERGE is a special case of SUPERSEDE for v1 (YAGNI): combining two entries is
# "prune the loser(s), keep/update the winner" — expressed as a SupersedeDecision
# with ``prune=[loser ids]``. No separate multi-way merge machinery.


def apply_new(store: IndexStore, entries: list[IndexEntry], d: NewDecision) -> None:
    """Append a new entry from a NEW decision (in place on ``entries``)."""
    entry = IndexEntry(
        id=store.alloc_id(),
        type=d.type,
        subject=d.subject,
        fact=d.fact,
        born_on=_normalize_born_on(d.born_on),
        updated_on=_normalize_born_on(d.born_on),
        source_ref=d.source_ref,
        source=[d.line_no],
    )
    entries.append(entry)


def apply_confirm(entries: list[IndexEntry], d: ConfirmDecision) -> None:
    """Grow an entry's ``source`` lineage; do NOT bump ``updated_on``."""
    target = _find_entry(entries, d.entry_id)
    if target is None:
        raise ValueError(f"confirm target {d.entry_id!r} not in index")
    target.source = _extend_source(target.source, d.line_no)


def apply_supersede(
    entries: list[IndexEntry],
    d: SupersedeDecision,
) -> None:
    """Update the winning entry and prune losers (in place on ``entries``).

    The winner keeps its ``id`` and ``born_on``; ``fact``/``source_ref`` move to
    the new current value, ``updated_on`` bumps to the new value's date (never
    backward), and ``source`` grows with the superseding line. Loser ids in
    ``d.prune`` are REMOVED entirely from the index.
    """
    target = _find_entry(entries, d.entry_id)
    if target is None:
        raise ValueError(f"supersede target {d.entry_id!r} not in index")
    target.fact = d.fact
    target.source_ref = d.source_ref
    # updated_on never moves backward (field contract).
    target.updated_on = _max_date(target.updated_on, _normalize_born_on(d.updated_on))
    target.source = _extend_source(target.source, d.line_no)
    for loser_id in d.prune:
        _prune_entry(entries, loser_id)


def _find_entry(entries: list[IndexEntry], entry_id: str) -> IndexEntry | None:
    for e in entries:
        if e.id == entry_id:
            return e
    return None


def _prune_entry(entries: list[IndexEntry], entry_id: str) -> None:
    """Remove ``entry_id`` from the index entirely (no tombstone)."""
    for i, e in enumerate(entries):
        if e.id == entry_id:
            del entries[i]
            return


def _extend_source(source: list[int], line_no: int) -> list[int]:
    """Append a journal line number to an entry's source lineage, deduped."""
    return [*source, line_no] if line_no not in source else list(source)


def _max_date(a: str, b: str) -> str:
    """Return the chronologically later of two ISO dates (lexical == chronological)."""
    return a if a >= b else b


# ---------------------------------------------------------------------------
# Reconcile orchestrator: journal tail -> index mutations + watermark advance
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# In-memory staging: the organizer's working set for ONE run
# ---------------------------------------------------------------------------


class Stage:
    """The in-memory working set the organizer mutates across one run.

    The organizer accumulates its NEW / CONFIRM / SUPERSEDE decisions against a
    ``Stage`` (loaded once from the on-disk index) and persists with ONE atomic
    write at run end (:meth:`finalize`). Nothing touches disk between tool calls,
    so a whole run is either committed or not — the same all-or-nothing shape as
    the journal's block append, but for the curated index.

    ``entries`` is the current (possibly mutated) entry list; ``watermark`` the
    current read cursor. ``finalize(end_line)`` resolves the new watermark
    (never backwards) and writes ``{version, watermark, next_id, entries}``
    atomically.
    """

    def __init__(self, store: IndexStore) -> None:
        self._store = store
        self.entries = store.load()
        self.watermark = store.load_watermark()
        store._advance_id_from(self.entries)

    def apply(self, decision: object) -> None:
        """Apply one decision to the staged entry list (no disk I/O)."""
        if isinstance(decision, NewDecision):
            apply_new(self._store, self.entries, decision)
        elif isinstance(decision, ConfirmDecision):
            apply_confirm(self.entries, decision)
        elif isinstance(decision, SupersedeDecision):
            apply_supersede(self.entries, decision)
        else:
            raise TypeError(f"unknown decision type: {type(decision).__name__}")

    def apply_all(self, decisions: list[object]) -> None:
        for d in decisions:
            self.apply(d)

    def finalize(self, end_line: int | None = None) -> int:
        """Atomic single write of the staged index + resolved watermark.

        Returns the new watermark. ``end_line`` is the exclusive end of the
        processed journal slice; the watermark never moves backward (idempotent
        re-runs). ``next_id`` persists the monotonic cursor so pruned ids are
        never reused (see ``_save``).
        """
        new_watermark = new_watermark_for(end_line, self.watermark)
        self._store._save(self.entries, new_watermark)
        self.watermark = new_watermark
        return new_watermark

    def render(self) -> str:
        return render_index_text(self.entries)


def reconcile(
    store: IndexStore,
    journal_records: list[tuple[int, MemoryRecord]],
    decisions: list[object],
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    watermark: int | None = None,
) -> int:
    """Apply ``decisions`` over a journal slice and persist, returning the new watermark.

    Deterministic single-shot convenience over :class:`Stage` (one atomic write):
    loads the index, applies all ``decisions`` mechanically, advances the
    watermark to ``end_line`` (never backward), and saves once. Re-running over
    an overlapping tail is idempotent (a decision for a line already in an
    entry's ``source`` is a no-op because ``_extend_source`` dedupes).

    ``journal_records`` is reserved for the organizer's line->record lookup
    (classify step); not required for applying decisions.

    This is the surface the deterministic unit tests use. The live organizer
    agent instead holds a :class:`Stage` across its tool calls and calls
    ``finalize`` once at run end.
    """
    del journal_records  # reserved
    stage = Stage(store)
    if watermark is not None:
        stage.watermark = watermark
    stage.apply_all(decisions)
    return stage.finalize(end_line)


def new_watermark_for(end_line: int | None, current: int) -> int:
    """New watermark = max(current, end_line) — never moves backward.

    ``end_line`` is the exclusive end of the processed journal slice. If it is
    behind/equal to ``current`` (a re-run over an overlapping tail), the cursor
    stays put (idempotence).
    """
    if end_line is None:
        return current
    return end_line if end_line > current else current


# ---------------------------------------------------------------------------
# Organizer agent: the sibling leaf agent that classifies the journal tail
# ---------------------------------------------------------------------------


def organize_model() -> str:
    """Resolve the organizer model from Settings, defaulting to the owner's.

    ``Settings.organize_model`` (``ORGANIZE_MODEL`` env / .env) wins when set;
    otherwise we fall back to the consolidation owner's model
    (``_consolidate.consolidate_model``), per the decision that the organizer
    shares the owner's model for now while staying a separately-tunable knob.
    """
    if settings.organize_model:
        return settings.organize_model.strip()
    from ._consolidate import consolidate_model

    return consolidate_model()


def _organize_model_settings() -> ModelSettings:
    """Build the organizer's `model_settings`, pinning OpenRouter routing.

    Mirrors ``_consolidate._consolidation_model_settings`` (deliberately
    duplicated, not imported — each leaf agent owns its routing pin): sampling
    determinism (``top_p: 1.0`` + optional fixed seed) plus the OpenRouter
    ``provider`` object via ``extra_body`` when ``organize_provider_order`` is
    set.
    """
    out: dict = {"temperature": 0}
    out["top_p"] = 1.0
    if settings.organize_seed is not None:
        out["seed"] = settings.organize_seed
    pinned_quantizations = [q for q in settings.organize_provider_quantizations if q]
    if settings.organize_provider_order:
        provider: dict = {"order": list(settings.organize_provider_order)}
        if pinned_quantizations:
            provider["quantizations"] = pinned_quantizations
        extra_body: dict = dict(out.get("extra_body") or {})
        extra_body["provider"] = provider
        out["extra_body"] = extra_body
    return cast(ModelSettings, out)


ORGANIZER_GUIDANCE = """\
You are the memory organizer for this user's durable memory. You maintain a
curated INDEX (memory-index.json) of durable facts, reconciling the new tail of
an append-only journal into it.

The consolidation owner already wrote immutable facts to the journal. Your job is
to fold the NEW journal records — lines the index has not yet seen — into the
index, classifying each one:

- NEW — the journal record expresses a fact not yet in the index. Create a new
  index entry. Pick `subject` by REUSING an existing subject string when this
  fact concerns that topic, otherwise mint a concise new one. Grouping is
  exact-string match.
- CONFIRM — the record restates an existing entry's fact (same subject, same
  value, no change). Do NOT create a new entry: confirm the existing one (grows
  its lineage, keeps its date).
- SUPERSEDE — the record states a new value for an existing entry's subject
  (e.g. "user moved to X" where the index said "user lives in Y"). Update the
  winning entry to the new value, and PRUNE any other entry the new value makes
  obsolete (removed entirely). A merge of two entries is a special case: keep
  the winner, prune the losers.

ADDRESSING: when a tool asks for `entry_id`, pass the entry's `mem-N` id EXACTLY
as shown in the index text (e.g. `mem-1`), never the subject string. The index
text lists each entry as `mem-1 [type/subject] ...` — copy the `mem-N` token.

TYPE vs SUBJECT (read carefully — do not confuse them):
- `type` is ALWAYS exactly one bare member of: identity | preference | fact |
  event | task | current. It is NEVER a compound and NEVER carries the topic.
- `subject` is the coarse entity/topic the fact is about (e.g. "paf",
  "Landon", "tooling approach"). The tool call passes `type` and `subject` as
  TWO SEPARATE fields.
- The `[type/subject]` you see in the index text (e.g. `[fact/paf]`) is just a
  display abbreviation of two fields — it is NOT a valid `type` value. When
  calling `index_new` for a fact about paf, pass `type="fact"` and
  `subject="paf"` — never `type="fact/paf"`.

Contracts (do not violate):
- Never edit the journal. You only write the index.
- `born_on` of an entry never changes (original date). `updated_on` bumps ONLY
  when the value changes; confirming does not bump it.
- `id` is never reused; you never hand out ids — the index store allocates them.
- A prune REMOVES the loser entry entirely (no tombstone). Only the journal grows.
- Facts must be plain, self-contained statements. Never invent facts absent from
  the journal records.

After classifying the whole tail, call `finalize_index` ONCE with `end_line` =
the last journal line you processed, to commit the index atomically. Do not
finalize more than once.

Reply with one short line: how many entries you added, confirmed, or superseded.
"""

ORGANIZE_PROMPT = """\
Reconcile these journal records into the index.

New journal records:
{journal_tail}

Current index entries:
{index_text}
"""


def render_index_text(entries: list[IndexEntry]) -> str:
    """Render the current index as compact text for the organizer's context."""
    if not entries:
        return "(empty index)"
    lines = []
    for e in entries:
        lines.append(
            f"{e.id} [{e.type}/{e.subject}] {e.fact} "
            f"(born {e.born_on}, updated {e.updated_on}, source {e.source})"
        )
    return "\n".join(lines)


# Near-match threshold for the `index_new` dedup guard. A candidate entry is a
# "re-paraphrase" and must be CONFIRM/SUPERSEDE'd (not freshly `index_new`'d)
# when its token overlap is above this. Overlap is MAX of Jaccard and
# containment, so both the "verbatim repeat" shape (high Jaccard) and the
# "short fact fully contained in a richer one" shape (mem-2 vs mem-31-style,
# high containment, low Jaccard) are caught.
_NEAR_MATCH_THRESHOLD = 0.60
# Tokens too generic to discriminate facts (articles, auxiliaries, "has/a/the"…).
_STOP = frozenset(
    "a an the and or but of to in for on with is are was were be been "
    "has have had do does did it its this that these those about into from "
    "using uses use used user via by as at not no yes i you we they he she".split()
)


def _fact_tokens(fact: str) -> set[str]:
    """Lowercased alnum tokens, minus stopwords; empty/whitespace-drops when short."""
    raw = re.findall(r"[a-z0-9]+", fact.lower())
    tokens = {t for t in raw if t not in _STOP and len(t) > 2}
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(a: set[str], b: set[str]) -> float:
    """Fraction of the smaller token set present in the larger.

    Catches "short fact is a paraphrase/restatement of a richer one" even when
    the richer entry has many extra tokens that crush the Jaccard score (e.g.
    ``mem-2``'s full identity vs ``mem-31``'s one-line repeat: Jaccard 0.25 but
    containment 0.80).
    """
    if not a or not b:
        return 0.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short & long) / len(short)


def _overlap(a: set[str], b: set[str]) -> float:
    """Max of Jaccard and containment — the full dedup similarity signal."""
    return max(_jaccard(a, b), _containment(a, b))


def _near_match(entries: list[IndexEntry], fact: str, *, type: str | None = None,
                subject: str | None = None) -> IndexEntry | None:
    """Return the existing entry whose fact is a near-dup of ``fact``, else None.

    Scans the whole entry list — ``Stage.entries`` is the on-disk index plus
    anything staged this run — so it catches not just in-run re-adds (the old
    blind spot) but also a NEW fact that re-states an existing disk entry. A
    match means the model should have called ``index_confirm``/``index_supersede``
    against that entry instead of a fresh ``index_new``.

    When ``type``/``subject`` are given, only entries of the SAME type and
    subject are candidates (removes cross-topic false positives like a fact
    about \"paf\" vs one about \"Hope Technologies\" that happen to share a few
    generic tokens).
    """
    tokens = _fact_tokens(fact)
    if not tokens:
        return None
    best: IndexEntry | None = None
    best_score = _NEAR_MATCH_THRESHOLD
    for e in entries:
        if type is not None and e.type != type:
            continue
        if subject is not None and e.subject != subject:
            continue
        score = _overlap(tokens, _fact_tokens(e.fact))
        if score > best_score:
            best, best_score = e, score
    return best


def _find_consolidation_candidates(
    entries: list[IndexEntry],
) -> list[list[IndexEntry]]:
    """Return connected near-duplicate clusters in ``entries`` (empty if none).

    A cluster is a group of entries of the SAME ``type`` and ``subject`` whose
    facts pairwise exceed the near-match overlap — i.e. they say the same thing
    and only one should survive. Used by the organizer's second (consolidation)
    pass to know which legacy/accumulated entries to fold together via
    ``supersede`` + ``prune``.

    Grouping is transitive (union-find): if A~B and B~C, all three form one
    cluster even if A and C are individually weaker matches, so a chain of
    paraphrases doesn't leave stragglers.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_key: dict[tuple[str, str], list[IndexEntry]] = {}
    for e in entries:
        by_key.setdefault((e.type, e.subject), []).append(e)

    for _key, group in by_key.items():
        toks = {e.id: _fact_tokens(e.fact) for e in group}
        ids = [e.id for e in group]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if _overlap(toks[ids[i]], toks[ids[j]]) > _NEAR_MATCH_THRESHOLD:
                    union(ids[i], ids[j])

    roots: dict[str, list[IndexEntry]] = {}
    for e in entries:
        roots.setdefault(find(e.id), []).append(e)
    # Keep only clusters with more than one member (a singleton isn't a dup).
    return [cl for cl in roots.values() if len(cl) > 1]



def _build_index_toolset():
    """Build the organizer's writable index toolset (in-memory staged).

    Mirrors ``JournalAppendToolset`` in shape: a ``FunctionToolset`` exposing
    the index-maintenance tools under an ``id='memory-index'``. Unlike the
    journal's per-call append, the organizer stages its decisions in memory
    (a :class:`Stage` created on first tool call from the current index) and
    persists with ONE atomic write via :func:`finalize_index` at run end —
    never between tool calls. Tools: ``index_new`` / ``index_confirm`` /
    ``index_supersede`` (accumulate) + ``finalize_index`` (commit).
    """

    from pydantic_ai.toolsets import FunctionToolset

    class _IndexToolset(FunctionToolset):
        def __init__(self, store: IndexStore) -> None:
            super().__init__(id="memory-index")
            self._store = store
            self._stage: Stage | None = None
            self._last_diff: str | None = None  # human summary of last applied action
            self.add_function(self.add_new, name="index_new")
            self.add_function(self.confirm, name="index_confirm")
            self.add_function(self.supersede, name="index_supersede")
            self.add_function(self.finalize_index, name="finalize_index")

        def _get_stage(self) -> Stage:
            if self._stage is None:
                self._stage = Stage(self._store)
            return self._stage

        def _staged_view(self) -> str:
            """A compact running view of the staged (post-edit) entries.

            Shows each staged entry as ``mem-N [type/subject] <fact-truncated>``
            so the model can see — on every call — exactly what it has already
            staged this run and avoid re-adding the same fact.
            """
            entries = self._get_stage().entries
            if not entries:
                return "(no staged entries)"
            lines = []
            for e in entries:
                fact = e.fact if len(e.fact) <= 90 else e.fact[:87] + "..."
                lines.append(f"{e.id} [{e.type}/{e.subject}] {fact}  (source {e.source})")
            return "\n".join(lines)

        def _result(self) -> MemoryWriteResult:
            staged = self._get_stage()
            return {
                "memory": "index",
                "file": _INDEX_FILENAME,
                "version": str(len(staged.entries)),
                "replayed": False,
                "status": "updated",
                "diff": (
                    f"{self._last_diff or 'no change'}; "
                    f"staged entries={len(staged.entries)}:\n{self._staged_view()}"
                ),
            }

        async def add_new(self, ctx, line_no: int, type: str, subject: str,
                          fact: str, born_on: str, source_ref: str):
            if type not in MEMORY_TYPES:
                raise ModelRetry(
                    f"invalid memory type {type!r}; expected one of "
                    f"{sorted(MEMORY_TYPES)}. `type` is ONLY the bare type "
                    f"member — never a `type/subject` compound. The topic goes "
                    f"in `subject`, not `type` (e.g. type='fact', subject='paf', "
                    f"not type='fact/paf')."
                )
            match = _near_match(
                self._get_stage().entries, fact, type=type, subject=subject
            )
            if match is not None:
                raise ModelRetry(
                    f"near-duplicate of existing entry {match.id}: "
                    f"{match.fact!r}. This fact already exists in the index — "
                    f"do NOT `index_new` it. Call `index_confirm(entry_id={match.id}, "
                    f"line_no=... )` if it just restates {match.id}, or "
                    f"`index_supersede(entry_id={match.id}, ...)` if it carries a "
                    f"new value/detail to fold into {match.id}. `index_new` is only "
                    f"for a genuinely new fact not already represented."
                )
            self._get_stage().apply(NewDecision(
                line_no=line_no, type=type, subject=subject, fact=fact,
                born_on=born_on, source_ref=source_ref,
            ))
            # Report the id the new entry got (last in the staged list) so the
            # model can reference it in later confirm/supersede calls.
            new_id = self._get_stage().entries[-1].id
            self._last_diff = f"index_new -> added {new_id} [{type}/{subject}] (line {line_no})"
            return self._result()

        def _valid_ids(self) -> list[str]:
            return [e.id for e in self._get_stage().entries]

        async def confirm(self, ctx, entry_id: str, line_no: int):
            try:
                self._get_stage().apply(ConfirmDecision(entry_id=entry_id, line_no=line_no))
            except ValueError:
                raise ModelRetry(
                    f"invalid entry_id {entry_id!r}; use one of the index ids "
                    f"{self._valid_ids()} (from the index text), never a subject"
                ) from None
            self._last_diff = f"index_confirm -> {entry_id} now sourced from line {line_no}"
            return self._result()

        async def supersede(self, ctx, entry_id: str, line_no: int, fact: str,
                            source_ref: str, updated_on: str,
                            prune: list[str] | None = None):
            try:
                self._get_stage().apply(SupersedeDecision(
                    entry_id=entry_id, line_no=line_no, fact=fact,
                    source_ref=source_ref, updated_on=updated_on,
                    prune=list(prune or []),
                ))
            except ValueError:
                raise ModelRetry(
                    f"invalid entry_id {entry_id!r}; use one of the index ids "
                    f"{self._valid_ids()} (from the index text), never a subject"
                ) from None
            pruned = f", pruned {prune}" if prune else ""
            self._last_diff = (
                f"index_supersede -> {entry_id} updated (line {line_no}){pruned}"
            )
            return self._result()

        async def finalize_index(self, ctx, end_line: int | None = None) -> MemoryWriteResult:
            """Persist the staged index atomically and advance the watermark.

            Call ONCE at the end of the run, with ``end_line`` = the last journal
            line processed (the new watermark). The watermark never moves
            backward. If a previous commit advanced past ``end_line``, nothing
            regresses (idempotent re-run).
            """
            new_wm = self._get_stage().finalize(end_line)
            return {
                "memory": "index",
                "file": _INDEX_FILENAME,
                "version": str(len(self._get_stage().entries)),
                "replayed": False,
                "status": "updated",
                "diff": f"committed watermark={new_wm}",
            }

    return _IndexToolset


def _build_memory_organizer(store: IndexStore, *, agent_name: str = "memory-organizer"):
    """Construct the writable memory-organizer agent.

    Returns ``(organizer, toolset)``. The organizer is a separate leaf agent
    (sibling of the consolidation owner) with its own model knob
    (``ORGANIZE_MODEL``, defaulting to the owner's model) and the fixed
    ``ORGANIZER_GUIDANCE`` operating manual. The assistant never carries this
    toolset — the organizer is the sole index writer.
    """
    toolset_cls = _build_index_toolset()
    toolset = toolset_cls(store)
    organizer = Agent(
        organize_model(),
        name=agent_name,
        toolsets=[toolset],
        model_settings=_organize_model_settings(),
        retries=3,
    )
    return organizer, toolset


def organize_batch_size() -> int:
    """Run-end auto-trigger threshold: reconcile only when the journal tail is
    longer than this (``LME_ORGANIZE_BATCH`` env, default 25). No daemon."""
    raw = os.environ.get("LME_ORGANIZE_BATCH")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("LME_ORGANIZE_BATCH=%r not an int; using %d", raw, _DEFAULT_ORGANIZE_BATCH)
    return _DEFAULT_ORGANIZE_BATCH


def render_journal_tail(
    journal_records: list[tuple[int, MemoryRecord]],
    watermark: int,
) -> str:
    """Render the journal records AFTER ``watermark`` as labeled lines for the prompt."""
    lines = []
    for lineno, rec in journal_records:
        if lineno <= watermark:
            continue
        lines.append(
            f"line {lineno} [{rec.type}] {rec.fact} "
            f"(born {rec.born_on}, ref {rec.source_ref})"
        )
    if not lines:
        return "(no new records)"
    return "\n".join(lines)


async def run_organizer(
    journal: JournalStore,
    *,
    index_store: IndexStore,
    agent_name: str = "memory-organizer",
) -> str:
    """Run the memory organizer over the new journal tail, committing the index.

    Reads all journal records, builds/renders the new tail ``(watermark, N]``
    plus the current index, runs the organizer leaf agent (which calls the
    staged index tools and ``finalize_index`` once), and returns the agent's
    final output line. The index is committed atomically by ``finalize_index``,
    advancing the watermark to the last journal line.

    Live-LLM: costs tokens. The caller is responsible for sourcing .env /
    ``ORGANIZE_MODEL`` (or the consol-model default).
    """
    records = journal.read_all()
    if not records:
        return "no changes"

    entries = index_store.load()
    watermark = index_store.load_watermark()
    tail = render_journal_tail(records, watermark)
    if tail == "(no new records)":
        return "no changes"

    prompt = ORGANIZE_PROMPT.format(
        journal_tail=tail,
        index_text=render_index_text(entries),
    )
    organizer, _toolset = _build_memory_organizer(index_store, agent_name=agent_name)
    result = await organizer.run(prompt)
    return str(result.output) if result.output is not None else "no changes"


def _pick_winner(cluster: list[IndexEntry]) -> IndexEntry:
    """Suggest the canonical survivor for a dup cluster.

    Prefer the entry with the best-attested lineage (most source lines — it is
    the entry the journal has reaffirmed most), then the richest fact (most
    significant tokens — captures the most detail to preserve), then the
    earliest-born as a stable tiebreak. This is only a hint: the organizer may
    pick a different member when that is semantically more accurate/complete.
    """
    return max(
        cluster,
        key=lambda e: (len(e.source), len(_fact_tokens(e.fact)), e.born_on),
    )


def _render_consolidation_candidates(clusters: list[list[IndexEntry]]) -> str:
    """Render near-duplicate clusters as a numbered list for the consolidation pass."""
    sections = []
    for i, cl in enumerate(clusters, 1):
        winner = _pick_winner(cl)
        members = "\n".join(
            f"  * {e.id}: {e.fact!r} (born {e.born_on}, updated {e.updated_on}, "
            f"source {e.source})"
            for e in sorted(cl, key=lambda e: e.id)
        )
        losers = ", ".join(sorted(e.id for e in cl if e is not winner))
        sections.append(
            f"Cluster {i} — these {len(cl)} entries say the same thing "
            f"(type={winner.type!r}, subject={winner.subject!r}):\n{members}\n"
            f"Suggested survivor: {winner.id} (richest fact, earliest born). "
            f"Prune the others ({losers}) unless they carry distinct detail that "
            f"must be folded into {winner.id} first."
        )
    return "\n\n".join(sections)


CONSOLIDATION_PROMPT = """\
Second pass — consolidate EXISTING duplicate entries in the index.

The entries below are near-duplicate clusters detected automatically: within
each cluster the facts restate the same thing (same type, same subject). For
EVERY cluster, keep exactly ONE entry and remove the rest, following the
SUPERSEDE contract:

- Pick the best survivor (the suggested one is usually right, but you may pick
  another member if it is clearly more accurate or complete).
- If a loser carries a distinct, valuable detail the winner lacks, fold that
  detail into the winner via `index_supersede(entry_id=WINNER, line_no=<last
  source line>, fact=<merged fact>)` — the merged fact must include the winning
  text PLUS any non-redundant details from the losers.
- If the losers add nothing, a single supersede that simply restores/keeps the
  winner's fact and prunes them is enough (call `index_supersede` once and pass
  the loser ids in `prune`).
- Never invent facts; only merge detail that is actually present.

Then call `finalize_index` ONCE to commit. Reply with one short line: how many
clusters you consolidated and how many entries you pruned.

Clusters to consolidate:
{clusters}

Current index (before this pass):
{index_text}
"""


async def run_consolidation(
    index_store: IndexStore,
    *,
    agent_name: str = "memory-organizer",
) -> str:
    """Second pass: consolidate near-duplicate entries already in the index.

    Loads the current index, detects near-dup clusters (same type + subject,
    high token overlap), and has the organizer merge each cluster into one
    canonical entry via ``supersede`` + ``prune``. Runs even when there is no
    new journal tail — this is what keeps the curated index from accumulating
    duplicate facts over time, every time it organizes. Returns a short summary,
    or ``"no changes"`` (zero tokens) when the index is already clean.

    Live-LLM (only when clusters exist), costs tokens. The caller is
    responsible for sourcing .env / ``ORGANIZE_MODEL`` (or the consol-model
    default) just like :func:`run_organizer`.
    """
    entries = index_store.load()
    clusters = _find_consolidation_candidates(entries)
    if not clusters:
        return "no changes"

    prompt = CONSOLIDATION_PROMPT.format(
        clusters=_render_consolidation_candidates(clusters),
        index_text=render_index_text(entries),
    )
    organizer, _toolset = _build_memory_organizer(index_store, agent_name=agent_name)
    result = await organizer.run(prompt)
    return str(result.output) if result.output is not None else "no changes"
