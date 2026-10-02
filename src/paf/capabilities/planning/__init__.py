"""paf-owned planning capability (originally copied from pydantic-ai-harness, MIT).

Model-owned task planning: a structured, self-updating plan the agent can
create and edit (``write_plan`` / ``read_plan`` / ``add_task`` /
``update_task_status`` / ``update_task_statuses`` / ``remove_task``), surfaced
back to the model each request as a cache-safe tail reminder. The plan lives
in a ``PlanStore``, resolved per-run so the web sidebar can react to changes.

- ``_toolset`` — the ``PlanningToolset`` (the function tools).
- ``_capability`` — the ``Planning`` capability (wraps the toolset).
- ``_store`` — the ``PlanStore`` protocol + ``InMemoryPlanStore``/``SqlitePlanStore``.
- ``_events`` — ``PlanEventEmitter`` + ``PlanEvent`` for sidebars/external trackers.
- ``_types`` — ``PlanItem``, ``TaskStatus``, ``PlanStatusUpdate``.

paf owns the surface and wires it through the capability-toolset seam (registry
+ profile + builder) instead of using the harness directly. ``enable_subtasks``
adds subtasks and dependencies (``add_subtask`` / ``set_dependency`` /
``get_available_tasks``); a parent completes on its own once its last subtask is
done. The Postgres/Redis stores are deliberately NOT ported (YAGNI); a paf
flat-file ``FilePlanStore`` is provided separately.
"""

from __future__ import annotations

from ._capability import Planning
from ._events import EventCallback, PlanEvent, PlanEventEmitter, PlanEventType
from ._store import FilePlanStore, InMemoryPlanStore, PlanStore, SqlitePlanStore
from ._toolset import PlanningToolset, render_plan
from ._types import PlanItem, PlanStatusUpdate, TaskStatus

__all__ = [
    "EventCallback",
    "FilePlanStore",
    "InMemoryPlanStore",
    "PlanEvent",
    "PlanEventEmitter",
    "PlanEventType",
    "PlanItem",
    "PlanStatusUpdate",
    "PlanStore",
    "Planning",
    "PlanningToolset",
    "SqlitePlanStore",
    "TaskStatus",
    "render_plan",
]
