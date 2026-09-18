"""paf-owned plan store protocol (copied verbatim from pydantic-ai-harness, MIT).

PlanStore protocol plus InMemoryPlanStore and SqlitePlanStore. paf adds a
flat-file FilePlanStore (Task 3) so plans are inspectable/diffable JSON under
data/plans/<conversation_id>/plan.json.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import anyio.to_thread

from paf.capabilities._storage import FlatFileStorage, StorageService
from paf.capabilities.planning._events import PlanEvent, PlanEventEmitter, PlanEventType
from paf.capabilities.planning._types import PlanItem, TaskStatus

_VALID_TABLE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}")


@runtime_checkable
class PlanStore(Protocol):
    """Async CRUD interface over an ordered list of plan steps.

    Bring your own backend by implementing these six methods; the toolset and
    capability depend only on this protocol.
    """

    async def get_items(self) -> list[PlanItem]:
        """Return every step in insertion order."""
        ...  # pragma: no cover

    async def set_items(self, items: list[PlanItem]) -> None:
        """Replace the whole list with `items`.

        This is a bulk replacement and does not emit `PlanEvent`s -- so the
        `write_plan` tool, which calls it, is event-silent. Applications that
        render off events should also read the plan after a run, or steer the
        model toward the granular tools (`add_task`, `update_task_status`, ...),
        which do emit.
        """
        ...  # pragma: no cover

    async def get_item(self, item_id: str) -> PlanItem | None:
        """Return the step with `item_id`, or `None`."""
        ...  # pragma: no cover

    async def add_item(self, item: PlanItem) -> PlanItem:
        """Append `item` and return it.

        Ids are unique within a store. An id that is already present raises `ValueError`
        rather than being appended: every reader here resolves an id to the first match, so a
        duplicate would shadow the original and make updates land on one of them at random.
        The SQL stores enforce this with a primary key; the others check.
        """
        ...  # pragma: no cover

    async def update_item(
        self,
        item_id: str,
        *,
        content: str | None = None,
        status: TaskStatus | None = None,
        active_form: str | None = None,
        parent_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> PlanItem | None:
        """Apply the non-`None` fields to `item_id`; return the updated step or `None`."""
        ...  # pragma: no cover

    async def remove_item(self, item_id: str) -> bool:
        """Delete `item_id`; return whether it existed."""
        ...  # pragma: no cover


def validate_table_name(table: str) -> None:
    """Reject a table name that is not a safe bare SQL identifier."""
    if not _VALID_TABLE_RE.fullmatch(table):
        raise ValueError(f"invalid table name: {table!r}")


def apply_updates(
    item: PlanItem,
    *,
    content: str | None,
    status: TaskStatus | None,
    active_form: str | None,
    parent_id: str | None,
    depends_on: list[str] | None,
) -> None:
    """Mutate `item` in place with each non-`None` field."""
    if content is not None:
        item.content = content
    if status is not None:
        item.status = status
    if active_form is not None:
        item.active_form = active_form
    if parent_id is not None:
        item.parent_id = parent_id
    if depends_on is not None:
        item.depends_on = list(depends_on)


def _snapshot(item: PlanItem) -> PlanItem:
    """Return an item detached from the store and other public values."""
    return item.model_copy(deep=True)


async def emit_created(emitter: PlanEventEmitter | None, item: PlanItem) -> None:
    """Emit a `created` event when an emitter is attached."""
    if emitter is not None:
        await emitter.emit_plan(
            PlanEvent(event_type=PlanEventType.created, item=_snapshot(item))
        )


async def emit_deleted(emitter: PlanEventEmitter | None, item: PlanItem) -> None:
    """Emit a `deleted` event when an emitter is attached."""
    if emitter is not None:
        await emitter.emit_plan(
            PlanEvent(event_type=PlanEventType.deleted, item=_snapshot(item))
        )


async def emit_mutation(
    emitter: PlanEventEmitter | None, updated: PlanItem, previous: PlanItem | None
) -> None:
    """Emit `updated`, then `status_changed`/`completed` if the status moved."""
    if emitter is None or previous is None:
        return
    await emitter.emit_plan(
        PlanEvent(
            event_type=PlanEventType.updated,
            item=_snapshot(updated),
            previous_state=_snapshot(previous),
        )
    )
    if updated.status != previous.status:
        await emitter.emit_plan(
            PlanEvent(
                event_type=PlanEventType.status_changed,
                item=_snapshot(updated),
                previous_state=_snapshot(previous),
            )
        )
        if updated.status is TaskStatus.completed:
            await emitter.emit_plan(
                PlanEvent(
                    event_type=PlanEventType.completed,
                    item=_snapshot(updated),
                    previous_state=_snapshot(previous),
                )
            )


async def emit_set_items_diff(
    emitter: PlanEventEmitter | None,
    previous: list[PlanItem],
    new_items: list[PlanItem],
) -> None:
    """Emit per-item events for a bulk replace (`set_items`).

    Diffs the previous list against `new_items` by id so a listener (e.g. a UI
    sidebar reacting off ``PlanEvent``s) can reconcile a ``write_plan`` bulk
    replace instead of only the granular tools: ``deleted`` for ids that
    vanished, ``created`` for ids that are new, and ``updated`` (+ status
    events) for ids whose content changed. No-op when no emitter is attached.
    """
    if emitter is None:
        return
    prev_by_id = {item.id: item for item in previous if item.id}
    new_by_id = {item.id: item for item in new_items if item.id}
    for pid, item in prev_by_id.items():
        if pid not in new_by_id:
            await emit_deleted(emitter, item)
    for nid, item in new_by_id.items():
        old = prev_by_id.get(nid)
        if old is None:
            await emit_created(emitter, item)
        elif item != old:
            await emit_mutation(emitter, item, old)


class InMemoryPlanStore:
    """In-process plan storage. The default backend; state is lost on exit."""

    def __init__(self, *, event_emitter: PlanEventEmitter | None = None) -> None:
        self._items: list[PlanItem] = []
        self._emitter = event_emitter

    async def get_items(self) -> list[PlanItem]:
        """Return every step in insertion order."""
        return [_snapshot(item) for item in self._items]

    async def set_items(self, items: list[PlanItem]) -> None:
        """Replace the whole list with `items`."""
        self._items = [_snapshot(item) for item in items]

    async def get_item(self, item_id: str) -> PlanItem | None:
        """Return the step with `item_id`, or `None`."""
        item = next((item for item in self._items if item.id == item_id), None)
        return None if item is None else _snapshot(item)

    async def add_item(self, item: PlanItem) -> PlanItem:
        """Append `item` and return it."""
        if any(existing.id == item.id for existing in self._items):
            raise ValueError(f"A step with id {item.id!r} is already in this plan.")
        stored = _snapshot(item)
        self._items.append(stored)
        await emit_created(self._emitter, stored)
        return _snapshot(stored)

    async def update_item(
        self,
        item_id: str,
        *,
        content: str | None = None,
        status: TaskStatus | None = None,
        active_form: str | None = None,
        parent_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> PlanItem | None:
        """Apply the non-`None` fields to `item_id`; return the updated step or `None`."""
        item = next((item for item in self._items if item.id == item_id), None)
        if item is None:
            return None
        previous = _snapshot(item) if self._emitter is not None else None
        apply_updates(
            item,
            content=content,
            status=status,
            active_form=active_form,
            parent_id=parent_id,
            depends_on=depends_on,
        )
        await emit_mutation(self._emitter, item, previous)
        return _snapshot(item)

    async def remove_item(self, item_id: str) -> bool:
        """Delete `item_id`; return whether it existed."""
        for index, item in enumerate(self._items):
            if item.id == item_id:
                removed = self._items.pop(index)
                await emit_deleted(self._emitter, removed)
                return True
        return False


class SqlitePlanStore:
    """SQLite-backed plan storage, scoped to a `session` for multi-tenancy.

    `:memory:` is not supported because this store opens and closes a connection
    for each operation. Use `InMemoryPlanStore` for ephemeral plans, or a
    file-backed SQLite database for persistence.

    Each blocking `sqlite3` call runs on a worker thread under a lock, so any
    single operation is safe. The read-modify-write ops (`update_item`,
    `remove_item`) span two locked calls, though, so concurrent writers on the
    same session can still race (last-write-wins). Drive one session from one
    task, or use `set_items` for atomic whole-plan writes.
    """

    def __init__(
        self,
        database: str = ".agent-plan.db",
        *,
        session: str = "default",
        table: str = "plan_items",
        event_emitter: PlanEventEmitter | None = None,
    ) -> None:
        validate_table_name(table)
        if database == ":memory:":
            raise ValueError(
                "SqlitePlanStore does not support ':memory:'; use InMemoryPlanStore or a file-backed database."
            )
        self._database = database
        self._session = session
        self._table = table
        self._emitter = event_emitter
        self._lock = threading.Lock()
        self._ready = False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database)
        if not self._ready:
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                "session TEXT NOT NULL, seq INTEGER NOT NULL, id TEXT NOT NULL, "
                "content TEXT NOT NULL, status TEXT NOT NULL, active_form TEXT NOT NULL, "
                "parent_id TEXT, depends_on TEXT NOT NULL, PRIMARY KEY (session, id))"
            )
            connection.commit()
            self._ready = True
        return connection

    def _row_to_item(self, row: tuple[object, ...]) -> PlanItem:
        return PlanItem(
            id=str(row[0]),
            content=str(row[1]),
            status=TaskStatus(str(row[2])),
            active_form=str(row[3]),
            parent_id=None if row[4] is None else str(row[4]),
            depends_on=list(json.loads(str(row[5]))),
        )

    def _insert_params(self, seq: int, item: PlanItem) -> tuple[object, ...]:
        return (
            self._session,
            seq,
            item.id,
            item.content,
            item.status.value,
            item.active_form,
            item.parent_id,
            json.dumps(item.depends_on),
        )

    def _get_items_sync(self) -> list[PlanItem]:
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    f"SELECT id, content, status, active_form, parent_id, depends_on "
                    f"FROM {self._table} WHERE session = ? ORDER BY seq",
                    (self._session,),
                ).fetchall()
            finally:
                connection.close()
        return [self._row_to_item(row) for row in rows]

    def _set_items_sync(self, items: list[PlanItem]) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    f"DELETE FROM {self._table} WHERE session = ?", (self._session,)
                )
                connection.executemany(
                    f"INSERT INTO {self._table} "
                    "(session, seq, id, content, status, active_form, parent_id, depends_on) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [self._insert_params(seq, item) for seq, item in enumerate(items)],
                )
                connection.commit()
            finally:
                connection.close()

    def _add_item_sync(self, item: PlanItem) -> None:
        with self._lock:
            connection = self._connect()
            try:
                next_seq = connection.execute(
                    f"SELECT COALESCE(MAX(seq) + 1, 0) FROM {self._table} WHERE session = ?",
                    (self._session,),
                ).fetchone()[0]
                connection.execute(
                    f"INSERT INTO {self._table} "
                    "(session, seq, id, content, status, active_form, parent_id, depends_on) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    self._insert_params(int(next_seq), item),
                )
                connection.commit()
            finally:
                connection.close()

    def _get_item_sync(self, item_id: str) -> PlanItem | None:
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    f"SELECT id, content, status, active_form, parent_id, depends_on "
                    f"FROM {self._table} WHERE session = ? AND id = ?",
                    (self._session, item_id),
                ).fetchone()
            finally:
                connection.close()
        return None if row is None else self._row_to_item(row)

    def _write_item_sync(self, item: PlanItem) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    f"UPDATE {self._table} SET content = ?, status = ?, active_form = ?, "
                    "parent_id = ?, depends_on = ? WHERE session = ? AND id = ?",
                    (
                        item.content,
                        item.status.value,
                        item.active_form,
                        item.parent_id,
                        json.dumps(item.depends_on),
                        self._session,
                        item.id,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

    def _remove_item_sync(self, item_id: str) -> bool:
        with self._lock:
            connection = self._connect()
            try:
                cursor = connection.execute(
                    f"DELETE FROM {self._table} WHERE session = ? AND id = ?",
                    (self._session, item_id),
                )
                connection.commit()
                return cursor.rowcount > 0
            finally:
                connection.close()

    async def get_items(self) -> list[PlanItem]:
        """Return every step for this session in insertion order."""
        return await anyio.to_thread.run_sync(self._get_items_sync)

    async def set_items(self, items: list[PlanItem]) -> None:
        """Replace the whole list for this session with `items`."""
        await anyio.to_thread.run_sync(self._set_items_sync, list(items))

    async def get_item(self, item_id: str) -> PlanItem | None:
        """Return the step with `item_id`, or `None`."""
        return await anyio.to_thread.run_sync(self._get_item_sync, item_id)

    async def add_item(self, item: PlanItem) -> PlanItem:
        """Append `item` and return it."""
        await anyio.to_thread.run_sync(self._add_item_sync, item)
        await emit_created(self._emitter, item)
        return item

    async def update_item(
        self,
        item_id: str,
        *,
        content: str | None = None,
        status: TaskStatus | None = None,
        active_form: str | None = None,
        parent_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> PlanItem | None:
        """Apply the non-`None` fields to `item_id`; return the updated step or `None`."""
        item = await self.get_item(item_id)
        if item is None:
            return None
        previous = item.model_copy() if self._emitter is not None else None
        apply_updates(
            item,
            content=content,
            status=status,
            active_form=active_form,
            parent_id=parent_id,
            depends_on=depends_on,
        )
        await anyio.to_thread.run_sync(self._write_item_sync, item)
        await emit_mutation(self._emitter, item, previous)
        return item

    async def remove_item(self, item_id: str) -> bool:
        """Delete `item_id`; return whether it existed."""
        removed = await self.get_item(item_id) if self._emitter is not None else None
        existed = await anyio.to_thread.run_sync(self._remove_item_sync, item_id)
        if existed and removed is not None:
            await emit_deleted(self._emitter, removed)
        return existed


class FilePlanStore:
    """Flat-file plan storage: one pretty-printed JSON per conversation.

    A thin typed wrapper over the shared ``StorageService`` seam
    (:class:`FlatFileStorage` backend): plans land at ``<root>/<conversation_id>/plan.json``
    (``kind="plan"`` + ``key=conversation_id``), so they are plain, inspectable,
    diffable artifacts (paf's flat-file-first principle) rather than an opaque
    database, and the backend stays swappable (SQLite / Postgres later) without
    changing this store or the toolset.

    Implements the same ``PlanStore`` protocol and ``PlanEventEmitter`` support
    as ``InMemoryPlanStore``/``SqlitePlanStore``. Each blocking ``FlatFileStorage``
    read/write runs on a worker thread under ``anyio.to_thread``.
    """

    #: Artifact ``kind`` under which the whole plan is stored (-> ``plan.json``).
    KIND = "plan"

    def __init__(
        self,
        root: str | os.PathLike[str] = ".agent-plans",
        *,
        conversation_id: str = "default",
        event_emitter: PlanEventEmitter | None = None,
        storage: StorageService | None = None,
    ) -> None:
        self._conversation_id = conversation_id
        # Default backend is the flat-file StorageService; an alternate backend can be
        # dropped in (tests / future SQLite) via `storage`.
        self._storage = storage if storage is not None else FlatFileStorage(Path(root))
        self._emitter = event_emitter

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FilePlanStore(conversation_id={self._conversation_id!r})"

    # --- serialization helpers (run on a worker thread) --------------------

    def _load_sync(self) -> list[PlanItem]:
        payload = self._storage.read(self.KIND, self._conversation_id)
        if payload is None:
            return []
        rows = (
            payload.get("items", [])
            if isinstance(payload, dict)
            else cast(list[Any], payload)
        )
        return [PlanItem.model_validate(row) for row in rows]

    def _save_sync(self, items: list[PlanItem]) -> None:
        self._storage.write(
            self.KIND,
            self._conversation_id,
            {"items": [item.model_dump() for item in items]},
        )

    def _get_items_sync(self) -> list[PlanItem]:
        return [_snapshot(item) for item in self._load_sync()]

    def _set_items_sync(self, items: list[PlanItem]) -> None:
        self._save_sync([_snapshot(item) for item in items])

    def _get_item_sync(self, item_id: str) -> PlanItem | None:
        item = next((i for i in self._load_sync() if i.id == item_id), None)
        return None if item is None else _snapshot(item)

    def _add_item_sync(self, item: PlanItem) -> PlanItem:
        items = self._load_sync()
        if any(existing.id == item.id for existing in items):
            raise ValueError(f"A step with id {item.id!r} is already in this plan.")
        stored = _snapshot(item)
        items.append(stored)
        self._save_sync(items)
        return stored

    def _write_item_sync(self, item: PlanItem) -> None:
        items = self._load_sync()
        for index, existing in enumerate(items):
            if existing.id == item.id:
                items[index] = _snapshot(item)
                self._save_sync(items)
                return
        # not present -> treat as append (mirrors graceful update semantics)
        items.append(_snapshot(item))
        self._save_sync(items)

    def _remove_item_sync(self, item_id: str) -> bool:
        items = self._load_sync()
        for index, existing in enumerate(items):
            if existing.id == item_id:
                del items[index]
                self._save_sync(items)
                return True
        return False

    # --- async PlanStore protocol ------------------------------------------

    async def get_items(self) -> list[PlanItem]:
        """Return every step in insertion order."""
        return await anyio.to_thread.run_sync(self._get_items_sync)

    async def set_items(self, items: list[PlanItem]) -> None:
        """Replace the whole list with `items`.

        Emits per-item diff events (`created`/`deleted`/`updated`) through the
        attached emitter (Task 4), so an event-driven sidebar sees a `write_plan`
        bulk replace and not just granular edits. No-op when no emitter is set.
        """
        previous = await self.get_items() if self._emitter is not None else []
        await anyio.to_thread.run_sync(self._set_items_sync, list(items))
        await emit_set_items_diff(self._emitter, previous, list(items))

    async def get_item(self, item_id: str) -> PlanItem | None:
        """Return the step with `item_id`, or `None`."""
        return await anyio.to_thread.run_sync(self._get_item_sync, item_id)

    async def add_item(self, item: PlanItem) -> PlanItem:
        """Append `item` and return it."""
        stored = await anyio.to_thread.run_sync(self._add_item_sync, item)
        await emit_created(self._emitter, stored)
        return _snapshot(stored)

    async def update_item(
        self,
        item_id: str,
        *,
        content: str | None = None,
        status: TaskStatus | None = None,
        active_form: str | None = None,
        parent_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> PlanItem | None:
        """Apply the non-`None` fields to `item_id`; return the updated step or `None`."""
        item = await self.get_item(item_id)
        if item is None:
            return None
        previous = item.model_copy() if self._emitter is not None else None
        apply_updates(
            item,
            content=content,
            status=status,
            active_form=active_form,
            parent_id=parent_id,
            depends_on=depends_on,
        )
        await anyio.to_thread.run_sync(self._write_item_sync, item)
        await emit_mutation(self._emitter, item, previous)
        return _snapshot(item)

    async def remove_item(self, item_id: str) -> bool:
        """Delete `item_id`; return whether it existed."""
        removed = await self.get_item(item_id) if self._emitter is not None else None
        existed = await anyio.to_thread.run_sync(self._remove_item_sync, item_id)
        if existed and removed is not None:
            await emit_deleted(self._emitter, removed)
        return existed
