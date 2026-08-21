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
from dataclasses import dataclass, field
from typing import cast

from pydantic_ai import Agent, ModelSettings

from paf.config import settings

from ._atomic import read_json, write_json_atomic
from ._journal import MEMORY_TYPES, MemoryRecord, _normalize_born_on

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


def reconcile(
    store: IndexStore,
    journal_records: list[tuple[int, MemoryRecord]],
    decisions: list[object],
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    watermark: int | None = None,
) -> int:
    """Apply ``decisions`` over a journal slice and advance the watermark.

    ``decisions`` is the organizer's classification output for the slice; this
    function applies them mechanically (the deterministic core) so callers can
    drive an entire reconcile without an LLM by supplying pre-computed
    decisions — and re-running over an overlapping tail is idempotent (a
    decision for a line already in ``source`` is a no-op because ``_extend_source``
    dedupes).

    Returns the new watermark. The caller persists via ``store._save``; this
    function works over an in-memory entry list so the whole reconcile can be
    one atomic write.

    ``journal_records`` is only needed for provenance/bootstrapping (line->record
    mapping) but is not required for applying decisions; kept for the running
    signature the organizer will use.
    """
    del journal_records  # reserved: line->record lookup for the organizer's classify step
    entries = store.load()
    store._advance_id_from(entries)

    for d in decisions:
        if isinstance(d, NewDecision):
            apply_new(store, entries, d)
        elif isinstance(d, ConfirmDecision):
            apply_confirm(entries, d)
        elif isinstance(d, SupersedeDecision):
            apply_supersede(entries, d)
        else:
            raise TypeError(f"unknown decision type: {type(d).__name__}")

    new_watermark = _advance_watermark(store, watermark=watermark, new_end=end_line)
    store._save(entries, new_watermark)
    return new_watermark


def new_watermark_for(end_line: int | None, current: int) -> int:
    """New watermark = max(current, end_line) — never moves backward.

    ``end_line`` is the exclusive end of the processed journal slice. If it is
    behind/equal to ``current`` (a re-run over an overlapping tail), the cursor
    stays put (idempotence).
    """
    if end_line is None:
        return current
    return end_line if end_line > current else current


def _advance_watermark(store: IndexStore, *, watermark: int | None, new_end: int | None) -> int:
    current = watermark if watermark is not None else store.load_watermark()
    return new_watermark_for(new_end, current)


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

Contracts (do not violate):
- Never edit the journal. You only write the index.
- `born_on` of an entry never changes (original date). `updated_on` bumps ONLY
  when the value changes; confirming does not bump it.
- `id` is never reused; you never hand out ids — the index store allocates them.
- A prune REMOVES the loser entry entirely (no tombstone). Only the journal grows.
- Facts must be plain, self-contained statements. Never invent facts absent from
  the journal records.

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


def _build_index_toolset():
    """Build the organizers' writable index toolset.

    Mirrors ``JournalAppendToolset`` in shape: a ``FunctionToolset`` exposing
    the index-maintenance tools under an ``id='memory-index'``. Tools emit the
    three decision types; all apply mechanically via the deterministic core.
    """
    from pydantic_ai.toolsets import FunctionToolset

    from ._toolset import MemoryWriteResult

    class _IndexToolset(FunctionToolset):
        def __init__(self, store: IndexStore) -> None:
            super().__init__(id="memory-index")
            self._store = store
            self.add_function(self.add_new, name="index_new")
            self.add_function(self.confirm, name="index_confirm")
            self.add_function(self.supersede, name="index_supersede")

        def _stage(self, ctx, decisions: list[object]) -> MemoryWriteResult:
            # Apply against the CURRENT on-disk index, atomically, advancing the
            # watermark to the slice end supplied by the caller.
            new_wm = reconcile(
                self._store, [], decisions,
            )
            return {
                "memory": "index",
                "file": _INDEX_FILENAME,
                "version": str(self._store.count()),
                "replayed": False,
                "status": "updated",
                "diff": f"watermark={new_wm}",
            }

        async def add_new(self, ctx, line_no: int, type: str, subject: str,
                          fact: str, born_on: str, source_ref: str):
            if type not in MEMORY_TYPES:
                raise ValueError(f"invalid memory type {type!r}")
            return self._stage(ctx, [NewDecision(line_no=line_no, type=type,
                                                 subject=subject, fact=fact,
                                                 born_on=born_on, source_ref=source_ref)])

        async def confirm(self, ctx, entry_id: str, line_no: int):
            return self._stage(ctx, [ConfirmDecision(entry_id=entry_id, line_no=line_no)])

        async def supersede(self, ctx, entry_id: str, line_no: int, fact: str,
                            source_ref: str, updated_on: str,
                            prune: list[str] | None = None):
            return self._stage(
                ctx,
                [SupersedeDecision(entry_id=entry_id, line_no=line_no, fact=fact,
                                   source_ref=source_ref, updated_on=updated_on,
                                   prune=list(prune or []))],
            )

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
