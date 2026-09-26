"""Background subagent tasks: launch, per-conversation cap, cancel, notify.

A ``background=True`` delegation returns its task id immediately; the child runs
on the process-level :data:`paf.background._background_runner` (so it outlives
the parent turn and the session) through the same ``run_subagent`` path as a
sync call. Tasks run in PARALLEL, capped per conversation by a semaphore — NOT
through a ``BackgroundQueueService``, whose one-worker-per-session design would
serialize them.

When the task settles, the injected ``on_complete(task_id, cid, ok, text)``
hook (the web transport's notify) is fired best-effort with the ready-made
``[system]`` message: the whole output if short, else a boundary-cut head plus a
``check_task`` pointer. A cancelled task does not notify — the model asked for it.

Children are in-process asyncio tasks, so a server restart kills them; the
store marks such rows ``interrupted`` on next open (see ``_store``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from paf.background import _background_runner

from ._notify import notify_text
from ._roster import Leaf
from ._runner import emit, run_subagent
from ._store import SubagentTaskStore

log = logging.getLogger(__name__)

OnComplete = Callable[[str, str | None, bool, str], Awaitable[None]]
"""``(task_id, conversation_id, ok, notify_text)`` — same shape as shell's hook."""

DEFAULT_MAX_CONCURRENT = 5


class BackgroundTasks:
    """Registry of live background subagent tasks (one per process)."""

    def __init__(self) -> None:
        self._live: dict[str, asyncio.Task[None]] = {}
        # Keyed by (event loop, conversation) so a semaphore is never shared
        # across loops (each `asyncio.run` in tests is a new loop).
        self._sems: dict[tuple[int, str], asyncio.Semaphore] = {}

    def _sem(self, cid: str | None, max_concurrent: int) -> asyncio.Semaphore:
        key = (id(asyncio.get_running_loop()), cid or "global")
        sem = self._sems.get(key)
        if sem is None:
            sem = self._sems[key] = asyncio.Semaphore(max(1, max_concurrent))
        return sem

    def launch(
        self,
        rec: dict[str, Any],
        leaf: Leaf,
        store: SubagentTaskStore,
        *,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        on_complete: OnComplete | None = None,
    ) -> asyncio.Task[None]:
        """Start ``rec`` in the background; returns the tracked task."""
        task_id = str(rec["task_id"])
        sem = self._sem(store.conversation_id, max_concurrent)
        task = _background_runner.spawn(
            self._guarded(rec, leaf, store, sem, on_complete)
        )
        self._live[task_id] = task
        task.add_done_callback(lambda _t: self._live.pop(task_id, None))
        return task

    async def _guarded(
        self,
        rec: dict[str, Any],
        leaf: Leaf,
        store: SubagentTaskStore,
        sem: asyncio.Semaphore,
        on_complete: OnComplete | None,
    ) -> None:
        task_id = str(rec["task_id"])
        try:
            async with sem:
                final, output = await run_subagent(rec, leaf, store)
        except asyncio.CancelledError:
            # Cancelled while still waiting for a slot: run_subagent never ran,
            # so the record is still `queued` — settle it here.
            current = store.get(task_id)
            if current is not None and current.get("status") == "queued":
                await emit(
                    store, store.update(task_id, status="cancelled", detail="cancelled")
                )
            raise
        if on_complete is None:
            return
        ok = final.get("status") == "done" and output is not None
        body = output if ok and output is not None else str(final.get("detail") or "")
        try:
            await on_complete(
                task_id,
                store.conversation_id,
                ok,
                notify_text(leaf.name, task_id, ok, body),
            )
        except Exception:  # notify is best-effort; the result is already saved
            log.exception("subagent notify failed for %s", task_id)

    def cancel(self, task_id: str) -> bool:
        """Cancel a live task; False when it is not running in this process."""
        task = self._live.get(task_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    def pending(self) -> list[asyncio.Task[None]]:
        """Live tasks (tests await these to settle background work)."""
        return [t for t in self._live.values() if not t.done()]


background_tasks = BackgroundTasks()
"""Process-wide registry: cancel works across per-run toolset instances."""
