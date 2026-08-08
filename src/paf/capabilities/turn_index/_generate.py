"""Turn-index generation: merge collector events into a per-conversation ledger + write refs.

The "step 1" of the turn-index build order (see `_capability` docstring): the writer that
produces the one-line-per-turn index. It is structured so the LLM summarizer can move to the
background later without a redesign -- for now the summarizer is a **stub** (hash/event-derived
placeholder, no model call) that runs synchronously, so the pipeline can be tested and the real
LLM dropped in at a single seam.

Two data surfaces (the "collect a lot / show a little" split):

1. **Collector -- `events.jsonl` per conversation.** The LARGE surface. A consolidated,
   deduplicated mirror of the per-run `StepEvent` shards `StepPersistence` already writes
   (`<persistence>/<run_id>/events.jsonl`). It is derived, not a second writer: we read the
   harness's own ledger, not a new inline tap. Never enters the model context.

   Layout::

       <turn_index_dir>/<conversation_id>/events.jsonl   # one trimmed StepEvent line per event

2. **Summary + ref -- `index.jsonl` (the `TurnIndexStore`).** The SMALL surface. One line per
   turn: ``{conversation_id, run_id, message_index, ts, summary, hash}``. The `summary` is the
   one-liner; the `(run_id, message_index)` key is the ref that `read_turn('{run_id}:{message_index}')`
   dereferences back to the full durable turn. This is what enters context.

A **`cursor.json`** per conversation records how far the collector mirror has been consumed, so
re-running the summarizer processes only new events (idempotent, no opaque DB).

Future-seam commentary:
- `summarize_turn` is a plain function: the real LLM summarizer replaces the stub body (or a
  `summarizer: Callable` is passed in). Same signature either way.
- This module runs sync-blocking today. To move it to the background, call `generate_index`
  from `after_node_run` / a notification hook or a worker task -- the function is already
  pure over (collector -> cursor) and needs no change.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic_ai_harness.step_persistence import EventKind, FileStepStore, StepEvent

from ._store import TurnIndexStore


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Collector: consolidated per-conversation events.jsonl
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventRecord:
    """A trimmed `StepEvent` line mirrored into the per-conversation collector."""

    run_id: str
    step_index: int
    conversation_id: str | None
    parent_run_id: str | None
    agent_name: str | None
    kind: EventKind
    timestamp: str
    tool_call_id: str | None
    tool_name: str | None
    error: str | None

    @classmethod
    def from_step_event(cls, e: StepEvent) -> "EventRecord":
        return cls(
            run_id=e.run_id,
            step_index=e.step_index,
            conversation_id=e.conversation_id,
            parent_run_id=e.parent_run_id,
            agent_name=e.agent_name,
            kind=e.kind,
            timestamp=e.timestamp.isoformat(),
            tool_call_id=e.tool_call_id,
            tool_name=e.tool_name,
            error=e.error,
        )

    def key(self) -> tuple[str, int, str]:
        """Dedup identity: (run_id, step_index, kind)."""
        return (self.run_id, self.step_index, self.kind)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "step_index": self.step_index,
            "conversation_id": self.conversation_id,
            "parent_run_id": self.parent_run_id,
            "agent_name": self.agent_name,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "error": self.error,
        }


def event_line(record: EventRecord) -> str:
    return json.dumps(record.to_dict(), ensure_ascii=False)


class CollectorStore:
    """Flat-file ledger of consolidated events, one per conversation."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _segments(self, conversation_id: str | None) -> str:
        seg = conversation_id or "__default__"
        return seg.replace("/", "_").replace("\\", "_")

    def _path(self, conversation_id: str | None) -> Path:
        return self.directory / self._segments(conversation_id) / "events.jsonl"

    def append(self, record: EventRecord) -> None:
        """Append a record, skipping it if an identical key already exists (idempotent).

        The collector is a derived mirror of the harness's per-run shards; re-running the
        merge must not duplicate events already mirrored. Checking the on-disk keys each
        call is O(n) but the collector is small (one line per event) and the alternative
        (a per-run marker) adds state the flat-file-first design avoids.
        """
        if any(existing.key() == record.key() for existing in self.read(record.conversation_id)):
            return
        path = self._path(record.conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = event_line(record)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def read(self, conversation_id: str | None = None) -> list[EventRecord]:
        """Read the collector ledger, optionally scoped to one conversation.

        When ``conversation_id`` is ``None``, return every conversation's events
        (used for a global sweep). Otherwise filter to just that conversation's own
        ``events.jsonl`` -- the per-conversation ledger is authoritative and the
        file set is exactly ``<dir>/<conversation_id>/events.jsonl``.
        """
        if conversation_id is not None:
            files = [self._path(conversation_id)]
        else:
            files = [
                self.directory / sub / "events.jsonl"
                for sub in sorted(p.name for p in self.directory.iterdir() if p.is_dir())
            ]
        lines: list[EventRecord] = []
        for f in files:
            if not f.exists():
                continue
            with f.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    lines.append(EventRecord(**json.loads(raw)))
        lines.sort(key=lambda e: (e.conversation_id or "", e.run_id, e.step_index))
        return lines


# ---------------------------------------------------------------------------
# Cursor: flat-file watermark per conversation (what has been summarized)
# ---------------------------------------------------------------------------


@dataclass
class Cursor:
    """Watermark of which runs have been summarized, per conversation.

    `run_id` / `step_index` / `message_index` record the LAST run advanced past
    (kept for introspection/back-compat), but the authoritative idempotency check
    is `covered_runs` -- the set of run_ids whose `index.jsonl` entry already
    exists. A single watermark is insufficient: runs in a shared conversation
    iterate out of write order (a newer run can sort before an older one), so a
    lone `run_id` only gates the most-recent run and older runs get re-summarized
    every hook (O(n^2) duplicate index lines). Tracking the covered set makes each
    `after_run` a true no-op for already-indexed turns. Flat-file JSON; the ledger
    (`events.jsonl`) and `index.jsonl` remain authoritative -- the cursor is a
    derived performance/forward-watermark, and losing it only re-runs idempotent
    appends.
    """

    run_id: str | None = None
    step_index: int = 0
    message_index: int = 0
    covered_runs: set[str] | None = None

    def __post_init__(self) -> None:
        # Normalize to a set so `_covered` is always cheap even for old cursors.
        if self.covered_runs is None:
            self.covered_runs = set()
        else:
            self.covered_runs = set(self.covered_runs)

    @property
    def covered(self) -> set[str]:
        # NB: `x or set()` is WRONG here -- an empty set is falsy, so it would return a
        # throwaway set and `covered.add()` would silently mutate nothing. Check for None.
        return self.covered_runs if self.covered_runs is not None else set()

    def is_covered(self, run_id: str) -> bool:
        """True when `run_id` already has an `index.jsonl` entry in this conversation."""
        return run_id in self.covered

    def mark_covered(self, run_id: str) -> None:
        """Record that `run_id`'s turn has been summarized into this conversation's index."""
        self.covered.add(run_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "step_index": self.step_index,
            "message_index": self.message_index,
            "covered_runs": sorted(self.covered),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Cursor":
        rid = data.get("run_id")
        step = data.get("step_index", 0)
        msg = data.get("message_index", 0)
        covered = data.get("covered_runs")
        return cls(
            run_id=str(rid) if isinstance(rid, str) else None,
            step_index=int(step) if isinstance(step, int) else 0,
            message_index=int(msg) if isinstance(msg, int) else 0,
            covered_runs=set(covered) if isinstance(covered, list) else set(),
        )

    def advance(self, run_id: str, step_index: int, message_index: int) -> None:
        # Runs are appended in order, so a single watermark is sufficient.
        # (Now supplemented by `covered_runs`, which is authoritative.)
        self.run_id = run_id
        self.step_index = step_index
        self.message_index = max(self.message_index, message_index)
        self.mark_covered(run_id)


class CursorStore:
    """Read/write the per-conversation cursor.json."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, conversation_id: str | None) -> Path:
        seg = (conversation_id or "__default__").replace("/", "_").replace("\\", "_")
        return self.directory / seg / "cursor.json"

    def load(self, conversation_id: str | None) -> Cursor:
        path = self._path(conversation_id)
        if not path.exists():
            return Cursor()
        return Cursor.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, conversation_id: str | None, cursor: Cursor) -> None:
        path = self._path(conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cursor.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Turn summarizer (STUB — no LLM call; real model drops into this seam)
# ---------------------------------------------------------------------------


def summarize_turn(
    events: list[EventRecord],
    *,
    run_id: str,
    message_index: int,
    conversation_id: str | None,
) -> str:
    """Stub one-liner for a turn. Deterministic, no model call.

    Replace this body (or its `Callable` injection) with the real LLM summarizer; the
    surrounding pipeline (`generate_index`) is unchanged. The stub builds a stable placeholder
    from the events so the index is usable and testable without a model.
    """
    if not events:
        # A turn with no mirrored events: title it from the run.
        return f"turn {message_index} in run {run_id} (no events indexed)"
    kinds = [e.kind for e in events]
    tools = sorted({e.tool_name for e in events if e.tool_name})
    failed = [e for e in events if e.kind.endswith("failed")]
    # Deterministic content hash -> stable placeholder differing by content.
    digest = hashlib.sha256("|".join(kinds).encode()).hexdigest()[:8]
    parts = [f"turn {message_index} in run {run_id}"]
    if tools:
        parts.append("tools: " + ", ".join(tools))
    if failed:
        parts.append(f"{len(failed)} failure(s)")
    return f"{'; '.join(parts)} [{digest}]"


# ---------------------------------------------------------------------------
# Orchestration: merge collector -> ledger, summarize -> index.jsonl, advance cursor
# ---------------------------------------------------------------------------


async def generated_turns(
    store: FileStepStore,
    collector: CollectorStore,
    cursor_store: CursorStore,
    summary_index: TurnIndexStore,
    *,
    conversation_id: str | None = None,
) -> list[dict[str, object]]:
    """Merge new collector events into the per-conversation ledger, then summarize new turns.

    Sync-blocking and idempotent: re-running processes only events past the cursor and appends
    only missing `index.jsonl` entries (dedup on content hash is left to the store/consumers).
    Returns the appended summary records (the newly-indexed turns).

    The `FileStepStore` API is async, so this orchestrator is async too; call it from an existing
    event loop (pydantic-ai hooks already run in one). To move summarization to the background,
    schedule this function on a worker task -- it already needs no change.
    """
    written: list[dict[str, object]] = []
    # 1. Enumerate runs, mirror their events into the consolidated ledger (dedup by key).
    seen: set[tuple[str, int, str]] = set()
    runs = await store.list_runs(conversation_id=conversation_id)
    for record in runs:
        conv = record.conversation_id if conversation_id is None else conversation_id
        for ev in await store.list_events(run_id=record.run_id):
            erec = EventRecord.from_step_event(ev)
            if not conversation_id or ev.conversation_id == conversation_id:
                if erec.key() not in seen:
                    seen.add(erec.key())
                    collector.append(erec)

    # 2. Load cursor, group ledger events by run, summarize each new run past the cursor.
    for conv in _all_conversations(collector):
        cursor = cursor_store.load(conv)
        evs = collector.read(conv)
        # Group by run. A turn = one summary per run for the stub; the real boundary
        # detector (user-turn anchors via `_is_user_turn`) replaces this when the LLM lands.
        by_run: dict[str, list[EventRecord]] = {}
        for e in evs:
            by_run.setdefault(e.run_id, []).append(e)
        for run_id, group in sorted(by_run.items()):
            # Cursor gate: skip runs already summarized (idempotency). `covered_runs`
            # is the authoritative check -- a single watermark can't gate runs that
            # iterate out of write order. The `step_index` high-water is kept as a
            # fallback for pre-`covered_runs` cursors (back-compat).
            if cursor.is_covered(run_id):
                continue
            if (
                cursor.run_id == run_id
                and group
                and cursor.step_index >= max(e.step_index for e in group)
            ):
                continue
            mi = len([e for e in group if e.kind == "tool_call_completed"])
            summary = summarize_turn(group, run_id=run_id, message_index=mi, conversation_id=conv)
            entry = summary_index.append(
                conversation_id=conv,
                run_id=run_id,
                message_index=mi,
                summary=summary,
            )
            cursor.advance(run_id, max(e.step_index for e in group), mi)
            written.append(
                {
                    "conversation_id": entry.conversation_id,
                    "run_id": entry.run_id,
                    "message_index": entry.message_index,
                    "ts": entry.ts,
                    "summary": entry.summary,
                    "hash": entry.hash,
                }
            )
        cursor_store.save(conv, cursor)
    return written


def _all_conversations(collector: CollectorStore) -> list[str | None]:
    return [p.name for p in collector.directory.iterdir() if p.is_dir()]