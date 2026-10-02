"""paf-owned planning toolset (originally copied from pydantic-ai-harness, MIT).

PlanningToolset: write_plan, read_plan, add_task, update_task_status,
update_task_statuses, remove_task. Subtask tools (add_subtask, set_dependency,
get_available_tasks) are gated behind enable_subtasks. A parent's status follows
its subtasks: it cannot be completed while one is open, and it completes on its
own once the last one is resolved.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset

from paf.capabilities.planning._types import PlanItem, PlanStatusUpdate, TaskStatus

if TYPE_CHECKING:
    from paf.capabilities.planning._capability import Planning
    from paf.capabilities.planning._store import PlanStore

WRITE_PLAN_DESCRIPTION = """\
Create or replace the entire plan. Pass the whole ordered list every time -- \
including steps that are unchanged, completed, or cancelled -- so there are no \
indices to track. Keep exactly one step `in_progress`. Call this first for \
multi-step work, then again as you start and finish steps. The reply echoes \
each step's bracketed [id] -- copy those ids verbatim when you later call \
update_task_status or update_task_statuses; never invent ids.\
"""

READ_PLAN_DESCRIPTION = """\
Read the current plan: each step's id, content, and status, plus a progress \
summary. Use it before granular edits (the ids come from here) and to check \
what is left.\
"""

ADD_TASK_DESCRIPTION = """\
Append one new `pending` step without replacing the plan. Prefer this over \
write_plan when you only need to add a single step.\
"""

UPDATE_TASK_STATUS_DESCRIPTION = """\
Update one step's status by id. Set `in_progress` when you START a step and \
`completed` when it is fully done -- never mark work complete while tests fail \
or the implementation is partial. If the id is wrong, the tool returns an error \
that echoes the current plan -- use the real bracketed [id]s from it. \
"""

UPDATE_TASK_STATUSES_DESCRIPTION = """\
Update several steps' statuses in one call -- ideal for handing off from a \
finished step to the next one. The whole batch is validated first: if any entry \
is invalid nothing is applied and the error echoes the current plan (use the \
real bracketed [id]s from it). Entries apply in \
order, so when a batch both completes a prerequisite and starts its dependent, \
list the prerequisite's completion first.\
"""

REMOVE_TASK_DESCRIPTION = """\
Permanently delete a step by id -- use it for steps that are no longer relevant \
or were created in error. To mark work done, use update_task_status instead.\
"""

ADD_SUBTASK_DESCRIPTION = """\
Add a `pending` subtask under an existing step, creating a parent/child link. \
Break a complex step into a handful of independently completable subtasks and \
work those, not the parent: the parent cannot be completed while a subtask is \
open, and it is completed automatically once its last subtask is done. Adding a \
subtask to a completed parent reopens it.\
"""

SET_DEPENDENCY_DESCRIPTION = """\
Record that one step must wait for another to complete. The dependent step is \
automatically marked `blocked` until its prerequisite is resolved (`completed` \
or `cancelled`). Self-dependencies, cycles, and duplicates are rejected.\
"""

GET_AVAILABLE_TASKS_DESCRIPTION = """\
List the steps that can be worked on now -- those that are not completed, not \
blocked, and have no incomplete dependencies. A step with open subtasks is not \
listed; its subtasks are. Use it to choose the next step when dependencies or \
subtasks are involved.\
"""

WRITE_PLAN_SUBTASKS_NOTE = """\
 To nest a step, set its `parent_id` to the parent's id. Within one call you may \
give a new parent a short id of your own (e.g. "p1") for its subtasks to \
reference. A completed step cannot have open subtasks, and a parent whose \
subtasks are all done is completed for you. Rewriting the plan without a step's \
`parent_id` or `depends_on` drops that link.\
"""

STATUS_SUBTASKS_NOTE = """\
 A step with subtasks cannot be completed while any of them is open; it is \
completed automatically when its last subtask is done.\
"""

CORE_TOOL_NAMES = (
    "write_plan",
    "read_plan",
    "add_task",
    "update_task_status",
    "update_task_statuses",
    "remove_task",
)
"""Tools registered in every mode. `tools=['write_plan']` narrows the surface to
just the whole-plan tool."""

SUBTASK_TOOL_NAMES = ("add_subtask", "set_dependency", "get_available_tasks")
"""Tools registered only when `enable_subtasks` is set."""

_ALL_DONE_NOTE = "All steps are completed. Do NOT call read_plan again -- respond to the user with a summary instead."
_MULTI_IN_PROGRESS_NOTE = "\n\nNote: keep only one step in_progress at a time."

_ICONS = {
    TaskStatus.pending: "[ ]",
    TaskStatus.in_progress: "[~]",
    TaskStatus.completed: "[x]",
    TaskStatus.cancelled: "[-]",
    TaskStatus.blocked: "[!]",
}


def available_tool_names(*, subtasks: bool) -> set[str]:
    """Return every tool name the toolset registers in the given mode."""
    return {*CORE_TOOL_NAMES, *(SUBTASK_TOOL_NAMES if subtasks else ())}


def reject_unknown_tool_names(
    names: Iterable[str] | None, offered: set[str], *, field: str
) -> None:
    """Raise if `names` mentions a tool this mode does not register.

    A typo here is otherwise inert -- an unknown `descriptions` key is ignored, and an unknown
    `tools` entry silently narrows nothing -- so it surfaces at construction instead of as a
    tool the model never sees. Subtask tool names are only offered with `enable_subtasks`.
    """
    if names is None:
        return
    unknown = sorted(set(names) - offered)
    if unknown:
        raise ValueError(
            f"Unknown planning {field}: {', '.join(unknown)}. Available: {', '.join(sorted(offered))}."
        )


def status_icon(status: TaskStatus) -> str:
    """Return the one-glyph status marker for a step."""
    return _ICONS[status]


def render_plan(items: list[PlanItem], *, tree: bool = False) -> str:
    """Render the compact checklist with a one-line progress summary.

    The flat form is the classic checklist. `tree` (subtasks mode) nests each
    subtask under its parent with outline numbers (1, 1.1, ...) and names the
    unresolved steps a step is waiting on, so the per-turn reminder shows the
    plan's structure rather than subtasks trailing at the end of the list.
    """
    if not items:
        return "No plan yet."
    if tree:
        by_id = {item.id: item for item in items}
        lines = []
        for item, number, depth in walk_tree(items):
            line = f"{'  ' * depth}{number}. {status_icon(item.status)} {item.content}"
            waiting = [
                by_id[dep_id].content
                for dep_id in item.depends_on
                if dep_id in by_id and not is_terminal(by_id[dep_id].status)
            ]
            if waiting:
                line += f" (waits on: {', '.join(waiting)})"
            lines.append(line)
    else:
        lines = [
            f"{index}. {status_icon(item.status)} {item.content}"
            for index, item in enumerate(items, 1)
        ]
    completed = sum(1 for item in items if item.status is TaskStatus.completed)
    lines.append(f"({completed}/{len(items)} completed)")
    return "\n".join(lines)


def walk_tree(items: list[PlanItem]) -> Iterator[tuple[PlanItem, str, int]]:
    """Yield `(item, outline_number, depth)` depth-first, each subtask under its parent.

    Siblings keep plan order, so a subtask appended to the end of the store still
    renders under its parent. A step whose parent is not in the plan is shown at
    the top level rather than dropped. Parent cycles are rejected on write
    (`validate_hierarchy`), so every step is reachable from a top-level one.
    """
    known = {item.id for item in items}
    children: dict[str | None, list[PlanItem]] = {}
    for item in items:
        parent_id = item.parent_id if item.parent_id in known else None
        children.setdefault(parent_id, []).append(item)
    seen: set[str] = set()

    def walk(
        nodes: list[PlanItem], prefix: str, depth: int
    ) -> Iterator[tuple[PlanItem, str, int]]:
        index = 0
        for item in nodes:
            if item.id in seen:
                continue
            seen.add(item.id)
            index += 1
            number = f"{prefix}{index}"
            yield item, number, depth
            yield from walk(children.get(item.id, []), f"{number}.", depth + 1)

    yield from walk(children.get(None, []), "", 0)


def find_item(items: list[PlanItem], item_id: str) -> PlanItem | None:
    """Return the step with `item_id` from a snapshot list, or `None`."""
    return next((item for item in items if item.id == item_id), None)


def has_cycle(items: list[PlanItem], item_id: str, depends_on_id: str) -> bool:
    """Return whether making `item_id` depend on `depends_on_id` would form a cycle."""
    by_id = {item.id: item for item in items}
    visited: set[str] = set()

    def visit(current: str) -> bool:
        if current == item_id:
            return True
        if current in visited:
            return False
        visited.add(current)
        node = by_id.get(current)
        if node is not None:
            return any(visit(dep) for dep in node.depends_on)
        return False

    return visit(depends_on_id)


def is_terminal(status: TaskStatus) -> bool:
    """Return whether `status` is terminal -- a step that will not progress further."""
    return status in (TaskStatus.completed, TaskStatus.cancelled)


def is_blocked(items: list[PlanItem], item: PlanItem) -> bool:
    """Return whether `item` has a dependency that is not yet resolved.

    A terminal prerequisite (completed or cancelled) no longer blocks -- a
    cancelled step will never complete, so it must not hold up its dependents.
    """
    by_id = {node.id: node for node in items}
    for dep_id in item.depends_on:
        dep = by_id.get(dep_id)
        if dep is not None and not is_terminal(dep.status):
            return True
    return False


def subtasks_of(items: list[PlanItem], item: PlanItem) -> list[PlanItem]:
    """Return the direct subtasks of `item`, in plan order."""
    return [node for node in items if node.parent_id == item.id]


def open_subtasks(items: list[PlanItem], item: PlanItem) -> list[PlanItem]:
    """Return the direct subtasks of `item` that are not yet resolved."""
    return [node for node in subtasks_of(items, item) if not is_terminal(node.status)]


def subtasks_done(items: list[PlanItem], item: PlanItem) -> bool:
    """Return whether `item` has subtasks, none still open, and at least one completed.

    This is the state in which a parent is completed for the model. Subtasks that
    were all cancelled do not finish a parent -- nothing was done -- so that case
    is left to the model.
    """
    subtasks = subtasks_of(items, item)
    return (
        bool(subtasks)
        and all(is_terminal(node.status) for node in subtasks)
        and any(node.status is TaskStatus.completed for node in subtasks)
    )


def dependency_block_transitions(
    items: list[PlanItem],
) -> Iterator[tuple[PlanItem, TaskStatus]]:
    """Yield `(item, new_status)` for each step whose dependency-driven block must change.

    A step with an unresolved prerequisite becomes `blocked`; a `blocked` step
    whose prerequisites are all resolved returns to `pending`. Steps without
    dependencies, and terminal steps, are left as they are. Every transition is
    computed against the same input snapshot, so callers can apply them in any
    order.
    """
    for item in items:
        if not item.depends_on:
            continue
        blocked = is_blocked(items, item)
        if blocked and item.status in (TaskStatus.pending, TaskStatus.in_progress):
            yield item, TaskStatus.blocked
        elif not blocked and item.status is TaskStatus.blocked:
            yield item, TaskStatus.pending


def parent_status_transitions(
    items: list[PlanItem],
) -> Iterator[tuple[PlanItem, TaskStatus]]:
    """Yield `(parent, new_status)` for each parent whose status its subtasks now decide.

    A parent is completed once its last subtask is resolved (see `subtasks_done`),
    and a completed parent goes back to `pending` when it has an open subtask again
    (one was reopened or added). A parent still waiting on a dependency is left
    until that resolves, and `blocked` is left to `dependency_block_transitions`,
    so the two never move the same step in one snapshot.
    """
    for item in items:
        if item.status is TaskStatus.completed:
            if open_subtasks(items, item):
                yield item, TaskStatus.pending
        elif (
            item.status in (TaskStatus.pending, TaskStatus.in_progress)
            and not is_blocked(items, item)
            and subtasks_done(items, item)
        ):
            yield item, TaskStatus.completed


def reconcile_plan(items: list[PlanItem]) -> list[tuple[PlanItem, TaskStatus]]:
    """Apply every structure-forced status change to `items` in place, until none remain.

    Changes cascade -- a parent completing can complete its own parent, or free a
    step that depends on it -- so this repeats to a fixed point. Returns the parent
    changes, for the tool reply to report.
    """
    parent_changes: list[tuple[PlanItem, TaskStatus]] = []
    for _ in range(2 * len(items) + 1):
        dependency_changes = list(dependency_block_transitions(items))
        parent_moves = list(parent_status_transitions(items))
        if not dependency_changes and not parent_moves:
            break
        for item, status in [*dependency_changes, *parent_moves]:
            item.status = status
        parent_changes.extend(parent_moves)
    return parent_changes


def describe_parent_changes(changes: list[tuple[PlanItem, TaskStatus]]) -> str:
    """Render the reply lines that tell the model which parents moved on their own."""
    lines = [
        f"\nParent '{item.content}' was completed automatically: all its subtasks are done."
        if status is TaskStatus.completed
        else f"\nParent '{item.content}' was reopened: it has an open subtask."
        for item, status in changes
    ]
    return "".join(lines)


def validate_hierarchy(items: list[PlanItem]) -> str | None:
    """Return an error if the plan has duplicate ids, broken/cyclic parents, or bad dependencies, else `None`."""
    ids = [item.id for item in items]
    duplicates = sorted({id_ for id_ in ids if ids.count(id_) > 1})
    if duplicates:
        return f"Duplicate step ids: {', '.join(duplicates)}. Every step needs a unique id."
    known = set(ids)
    by_id = {item.id: item for item in items}
    for item in items:
        if item.parent_id is not None and item.parent_id not in known:
            return f"Step '{item.id}' has parent_id '{item.parent_id}', which is not in the plan."
        for dep_id in item.depends_on:
            if dep_id not in known:
                return (
                    f"Step '{item.id}' depends on '{dep_id}', which is not in the plan."
                )
    for item in items:
        seen: set[str] = set()
        parent_id = item.parent_id
        while parent_id is not None:
            if parent_id in seen:
                return f"Step '{item.id}' is part of a parent cycle."
            seen.add(parent_id)
            parent = by_id.get(parent_id)
            parent_id = parent.parent_id if parent is not None else None
    color: dict[str, int] = {}

    def has_dependency_cycle(node_id: str) -> bool:
        color[node_id] = 1
        for dep_id in by_id[node_id].depends_on:
            state = color.get(dep_id, 0)
            if state == 1 or (state == 0 and has_dependency_cycle(dep_id)):
                return True
        color[node_id] = 2
        return False

    for item in items:
        if color.get(item.id, 0) == 0 and has_dependency_cycle(item.id):
            return "Plan has a dependency cycle."
    return None


def render_flat(items: list[PlanItem], *, subtasks: bool) -> str:
    """Render the detailed numbered list with ids.

    With `subtasks`, each subtask is indented under its parent with an outline
    number (1, 1.1, ...) and dependencies are listed by id.
    """
    lines = ["Current plan:"]
    if not subtasks:
        for index, item in enumerate(items, 1):
            lines.append(
                f"{index}. {status_icon(item.status)} [{item.id}] {item.content}"
            )
        return "\n".join(lines)
    for item, number, depth in walk_tree(items):
        indent = "  " * depth
        lines.append(
            f"{indent}{number}. {status_icon(item.status)} [{item.id}] {item.content}"
        )
        if item.depends_on:
            lines.append(f"{indent}   (depends on: {', '.join(item.depends_on)})")
    return "\n".join(lines)


def _counts(items: list[PlanItem]) -> dict[TaskStatus, int]:
    counts = {status: 0 for status in TaskStatus}
    for item in items:
        counts[item.status] += 1
    return counts


def status_counts_line(items: list[PlanItem]) -> str:
    """Render the `X completed, Y in progress, Z pending` status tally."""
    counts = _counts(items)
    parts = [f"{counts[TaskStatus.completed]} completed"]
    if counts[TaskStatus.blocked] > 0:
        parts.append(f"{counts[TaskStatus.blocked]} blocked")
    parts.append(f"{counts[TaskStatus.in_progress]} in progress")
    parts.append(f"{counts[TaskStatus.pending]} pending")
    if counts[TaskStatus.cancelled] > 0:
        parts.append(f"{counts[TaskStatus.cancelled]} cancelled")
    return ", ".join(parts)


def render_summary(items: list[PlanItem]) -> str:
    """Render the trailing `Summary:` block, with the all-done note when finished."""
    counts = _counts(items)
    summary = f"Summary: {status_counts_line(items)}"
    active = (
        counts[TaskStatus.pending]
        + counts[TaskStatus.in_progress]
        + counts[TaskStatus.blocked]
    )
    if active == 0 and counts[TaskStatus.completed] > 0:
        summary += f"\n\n{_ALL_DONE_NOTE}"
    return summary


class PlanningToolset(FunctionToolset[AgentDepsT]):
    """Plan tools registered against a `Planning` capability's resolved store.

    `write_plan` (whole-plan replacement) is always present, alongside granular
    `read_plan`, `add_task`, `update_task_status`, `update_task_statuses`, and
    `remove_task`. When `enable_subtasks` is set, `add_subtask`, `set_dependency`,
    and `get_available_tasks` are added and the `blocked` status becomes valid.
    """

    def __init__(self, capability: Planning[AgentDepsT]) -> None:
        super().__init__(id="planning")
        self._capability = capability
        self._subtasks = capability.enable_subtasks
        descriptions = capability.descriptions or {}
        offered = available_tool_names(subtasks=self._subtasks)
        reject_unknown_tool_names(capability.tools, offered, field="tools")
        reject_unknown_tool_names(descriptions, offered, field="descriptions")
        selected = offered if capability.tools is None else set(capability.tools)

        def wanted(name: str) -> bool:
            return name in selected

        # Subtasks change what writing a plan and moving a status mean, so those
        # built-in descriptions say so; an override is used verbatim.
        subtasks_note = STATUS_SUBTASKS_NOTE if self._subtasks else ""
        if wanted("write_plan"):
            self.add_function(
                self.write_plan,
                name="write_plan",
                description=descriptions.get(
                    "write_plan",
                    WRITE_PLAN_DESCRIPTION
                    + (WRITE_PLAN_SUBTASKS_NOTE if self._subtasks else ""),
                ),
            )
        if wanted("read_plan"):
            self.add_function(
                self.read_plan,
                name="read_plan",
                description=descriptions.get("read_plan", READ_PLAN_DESCRIPTION),
            )
        if wanted("add_task"):
            self.add_function(
                self.add_task,
                name="add_task",
                description=descriptions.get("add_task", ADD_TASK_DESCRIPTION),
            )
        if wanted("update_task_status"):
            self.add_function(
                self.update_task_status,
                name="update_task_status",
                description=descriptions.get(
                    "update_task_status", UPDATE_TASK_STATUS_DESCRIPTION + subtasks_note
                ),
            )
        if wanted("update_task_statuses"):
            self.add_function(
                self.update_task_statuses,
                name="update_task_statuses",
                description=descriptions.get(
                    "update_task_statuses",
                    UPDATE_TASK_STATUSES_DESCRIPTION + subtasks_note,
                ),
            )
        if wanted("remove_task"):
            self.add_function(
                self.remove_task,
                name="remove_task",
                description=descriptions.get("remove_task", REMOVE_TASK_DESCRIPTION),
            )
        if wanted("add_subtask"):
            self.add_function(
                self.add_subtask,
                name="add_subtask",
                description=descriptions.get("add_subtask", ADD_SUBTASK_DESCRIPTION),
            )
        if wanted("set_dependency"):
            self.add_function(
                self.set_dependency,
                name="set_dependency",
                description=descriptions.get(
                    "set_dependency", SET_DEPENDENCY_DESCRIPTION
                ),
            )
        if wanted("get_available_tasks"):
            self.add_function(
                self.get_available_tasks,
                name="get_available_tasks",
                description=descriptions.get(
                    "get_available_tasks", GET_AVAILABLE_TASKS_DESCRIPTION
                ),
            )

    def _resolve(self, ctx: RunContext[AgentDepsT]) -> PlanStore:
        return self._capability.resolve_store(ctx)

    def _valid_status(self, status: TaskStatus) -> bool:
        return self._subtasks or status is not TaskStatus.blocked

    async def write_plan(
        self, ctx: RunContext[AgentDepsT], items: list[PlanItem]
    ) -> str:
        """Create or replace the whole plan.

        Args:
            ctx: Framework-provided run context.
            items: The complete ordered list of plan steps.
        """
        new_items = [item.model_copy(deep=True) for item in items]
        ids = [item.id for item in new_items]
        duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
        if duplicates:
            return f"Plan not updated: Duplicate step ids: {', '.join(duplicates)}. Every step needs a unique id."
        for item in new_items:
            if not self._valid_status(item.status):
                return f"Plan not updated: status '{item.status.value}' is only valid with subtasks enabled."
            if not self._subtasks and (item.parent_id is not None or item.depends_on):
                # Storing them would be a write the plan does not reflect: nothing reconciles
                # dependencies without the subtask tools, and no view renders the hierarchy.
                return (
                    f"Plan not updated: step '{item.content}' sets parent_id/depends_on, which are "
                    "only valid with subtasks enabled."
                )
            if item.status is TaskStatus.blocked and not item.depends_on:
                # Nothing would ever free it: reconciliation only touches steps that have
                # dependencies, and `get_available_tasks` excludes `blocked`.
                return f"Plan not updated: step '{item.content}' is blocked but depends on nothing."
        if self._subtasks:
            error = validate_hierarchy(new_items)
            if error is not None:
                return f"Plan not updated: {error}"
            for item in new_items:
                if item.status is TaskStatus.completed and open_subtasks(
                    new_items, item
                ):
                    return (
                        f"Plan not updated: step '{item.content}' is completed but has open "
                        "subtasks. Leave the parent open -- it completes automatically when "
                        "its last subtask is done."
                    )
        # Reconcile dependency blocks and parent rollups in memory before storing, so
        # `write_plan` commits one bulk `set_items` instead of a write per step.
        parent_changes = reconcile_plan(new_items) if self._subtasks else []
        await self._resolve(ctx).set_items(new_items)
        in_progress = sum(
            1 for item in new_items if item.status is TaskStatus.in_progress
        )
        note = "" if in_progress <= 1 else _MULTI_IN_PROGRESS_NOTE
        # Echo the assigned step ids (id-echo): the model passes items without ids on
        # write, and the store materializes uuids it would otherwise never learn. Rendering
        # them here (matching read_plan's `[id]` format) means a follow-up
        # update_task_status / update_task_statuses can reference the real ids instead of
        # guessing "1", "2", ... The bracketed ids ARE what the update tools take.
        return (
            f"Plan updated: {len(new_items)} step(s).\n\n"
            f"{render_flat(new_items, subtasks=self._subtasks)}"
            f"\n\nUse the bracketed [id] values above with update_task_status or "
            f"update_task_statuses — do NOT guess ids.{note}"
            f"{describe_parent_changes(parent_changes)}"
        )

    async def read_plan(self, ctx: RunContext[AgentDepsT]) -> str:
        """Read the current plan."""
        items = await self._resolve(ctx).get_items()
        if not items:
            return "No plan yet. Use write_plan to create one."
        return (
            f"{render_flat(items, subtasks=self._subtasks)}\n\n{render_summary(items)}"
        )

    async def add_task(
        self, ctx: RunContext[AgentDepsT], content: str, active_form: str = ""
    ) -> str:
        """Add one new pending step.

        Args:
            ctx: Framework-provided run context.
            content: The step description in imperative form.
            active_form: Optional present-continuous label, e.g. "Fix bug" -> "Fixing bug".
        """
        item = await self._resolve(ctx).add_item(
            PlanItem(content=content, active_form=active_form)
        )
        return f"Added step '{content}' with id: {item.id}"

    async def _reconcile(self, store: PlanStore) -> list[tuple[PlanItem, TaskStatus]]:
        """Write every structure-forced status change through the store, until none remain.

        A step with dependencies is `blocked` while any prerequisite is incomplete
        and returns to `pending` once they all complete; a parent completes when its
        last subtask is done and reopens when it has an open subtask again (see
        `parent_status_transitions`). Each change is written through the store, so
        the granular tools emit `status_changed` events for it. Returns the parent
        changes for the reply.
        """
        items = await store.get_items()
        parent_changes: list[tuple[PlanItem, TaskStatus]] = []
        for _ in range(2 * len(items) + 1):
            dependency_changes = list(dependency_block_transitions(items))
            parent_moves = list(parent_status_transitions(items))
            if not dependency_changes and not parent_moves:
                break
            for item, new_status in [*dependency_changes, *parent_moves]:
                await store.update_item(item.id, status=new_status)
            parent_changes.extend(parent_moves)
            items = await store.get_items()
        return parent_changes

    @staticmethod
    def _refuse_inert_block(
        items: list[PlanItem], item: PlanItem, status: TaskStatus
    ) -> str | None:
        """Refuse a `blocked` that the dependency reconciliation would undo in the same call.

        A step whose prerequisites are all resolved is not blocked by anything, so
        `dependency_block_transitions` returns it to `pending` immediately. Applying the write
        and then silently reversing it, while the reply says it worked, is the worst of the
        three available answers. A step with no dependencies at all is a different case and
        stays supported: nothing reconciles it, so a manual block there persists.
        """
        if (
            status is not TaskStatus.blocked
            or not item.depends_on
            or is_blocked(items, item)
        ):
            return None
        return f"Cannot block '{item.content}': every step it depends on is already resolved."

    @staticmethod
    def _refuse_parent_status(
        items: list[PlanItem], item: PlanItem, status: TaskStatus
    ) -> str | None:
        """Refuse a parent status that its subtasks contradict.

        A parent's completion follows its subtasks. Completing it while one is open
        would claim work that is not done, and reopening one whose subtasks are all
        done would be undone in the same call by the rollup -- so both are refused
        with the move that does work instead.
        """
        if status is TaskStatus.completed:
            remaining = open_subtasks(items, item)
            if not remaining:
                return None
            names = ", ".join(f"'{node.content}'" for node in remaining)
            return (
                f"Cannot complete '{item.content}': {len(remaining)} subtask(s) still open "
                f"({names}). Complete or cancel them -- the parent then completes automatically."
            )
        if (
            not is_terminal(status)
            and item.status is TaskStatus.completed
            and subtasks_done(items, item)
        ):
            return (
                f"Cannot reopen '{item.content}': all its subtasks are done, so it would be "
                "completed again. Reopen one of its subtasks or add one with add_subtask -- "
                "the parent reopens with it."
            )
        return None

    @staticmethod
    async def _stored_status(
        store: PlanStore, task_id: str, requested: TaskStatus
    ) -> str:
        """The status the store now holds, which is not always the one that was asked for.

        Reconciliation may have blocked the step on an unmet dependency. Reporting the request
        instead would tell the model one thing and show it another in the next turn's plan.
        """
        item = await store.get_item(task_id)
        return (requested if item is None else item.status).value

    async def update_task_status(
        self, ctx: RunContext[AgentDepsT], task_id: str, status: TaskStatus
    ) -> str:
        """Update one step's status by id.

        Args:
            ctx: Framework-provided run context.
            task_id: Id of the step to update.
            status: New status.
        """
        store = self._resolve(ctx)
        if not self._valid_status(status):
            return (
                "Invalid status 'blocked': subtasks are not enabled on this capability."
            )
        items = await store.get_items()
        item = find_item(items, task_id)
        if item is None:
            raise ModelRetry(
                f"Step with id '{task_id}' not found. Current plan:\n"
                f"{render_flat(items, subtasks=self._subtasks)}\n\n"
                f"Use the bracketed [id] values above with update_task_status — do NOT guess ids."
            )
        if (
            self._subtasks
            and status is TaskStatus.in_progress
            and is_blocked(items, item)
        ):
            return f"Cannot start '{item.content}': it has incomplete dependencies."
        if refusal := self._refuse_inert_block(items, item, status):
            return refusal
        if refusal := self._refuse_parent_status(items, item, status):
            return refusal
        await store.update_item(task_id, status=status)
        parent_changes = await self._reconcile(store)
        return (
            f"Updated step '{item.content}' status to "
            f"'{await self._stored_status(store, task_id, status)}'."
            f"{describe_parent_changes(parent_changes)}"
        )

    async def update_task_statuses(
        self, ctx: RunContext[AgentDepsT], updates: list[PlanStatusUpdate]
    ) -> str:
        """Update several steps' statuses in one call.

        The batch is validated as a unit -- if any entry is invalid, nothing is
        applied. Valid updates are then written to the store sequentially (the
        store, not this method, decides how each write is committed).

        Args:
            ctx: Framework-provided run context.
            updates: The `{task_id, status}` entries to apply.
        """
        store = self._resolve(ctx)
        if not updates:
            return "No updates provided."
        # Validate against a projection so earlier entries are visible to later ones
        # (e.g. completing a prerequisite and starting its dependent in one call).
        projected = [item.model_copy(deep=True) for item in await store.get_items()]
        errors: list[str] = []
        resolved: list[tuple[PlanItem, TaskStatus]] = []
        for update in updates:
            if not self._valid_status(update.status):
                errors.append(
                    f"Invalid status 'blocked' for '{update.task_id}': subtasks are not enabled."
                )
                continue
            item = find_item(projected, update.task_id)
            if item is None:
                errors.append(f"Step with id '{update.task_id}' not found.")
                continue
            if (
                self._subtasks
                and update.status is TaskStatus.in_progress
                and is_blocked(projected, item)
            ):
                errors.append(
                    f"Cannot start '{item.content}': it has incomplete dependencies."
                )
                continue
            if refusal := self._refuse_inert_block(projected, item, update.status):
                errors.append(refusal)
                continue
            if refusal := self._refuse_parent_status(projected, item, update.status):
                errors.append(refusal)
                continue
            item.status = update.status
            resolved.append((item, update.status))
        if errors:
            raise ModelRetry(
                "No changes applied. Errors:\n"
                + "\n".join(f"- {error}" for error in errors)
                + "\n\nCurrent plan:\n"
                + f"{render_flat(projected, subtasks=self._subtasks)}\n\n"
                + "Use the bracketed [id] values above with update_task_status or "
                "update_task_statuses — do NOT guess ids."
            )
        for item, status in resolved:
            await store.update_item(item.id, status=status)
        parent_changes = await self._reconcile(store)
        lines = [
            f"- [{item.id}] {item.content} -> {await self._stored_status(store, item.id, status)}"
            for item, status in resolved
        ]
        return (
            f"Updated {len(resolved)} step(s):\n"
            + "\n".join(lines)
            + describe_parent_changes(parent_changes)
        )

    async def remove_task(self, ctx: RunContext[AgentDepsT], task_id: str) -> str:
        """Remove a step by id.

        Args:
            ctx: Framework-provided run context.
            task_id: Id of the step to remove.
        """
        store = self._resolve(ctx)
        item = await store.get_item(task_id)
        if item is None:
            return f"Step with id '{task_id}' not found."
        await store.remove_item(task_id)
        subtasks_removed = 0
        parent_changes: list[tuple[PlanItem, TaskStatus]] = []
        if self._subtasks:
            subtasks_removed = await self._cleanup_after_removal(store, task_id)
            # Removing a parent's last open subtask leaves it finished.
            parent_changes = await self._reconcile(store)
        extra = f" and {subtasks_removed} subtask(s)" if subtasks_removed else ""
        return (
            f"Removed step '{item.content}' (id: {task_id}){extra}."
            f"{describe_parent_changes(parent_changes)}"
        )

    async def _cleanup_after_removal(self, store: PlanStore, removed_id: str) -> int:
        """Cascade-remove the deleted step's descendants and drop dangling `depends_on` refs.

        Returns the number of descendant subtasks removed. Prevents orphaned
        subtasks and stale dependencies (which `is_blocked` would treat as met).
        """
        items = await store.get_items()
        children: dict[str, list[str]] = {}
        for item in items:
            if item.parent_id is not None:
                children.setdefault(item.parent_id, []).append(item.id)
        descendants: list[str] = []
        stack = list(children.get(removed_id, []))
        while stack:
            current = stack.pop()
            descendants.append(current)
            stack.extend(children.get(current, []))
        for descendant_id in descendants:
            await store.remove_item(descendant_id)
        gone = {removed_id, *descendants}
        stripped: list[str] = []
        for item in await store.get_items():
            if any(dep in gone for dep in item.depends_on):
                await store.update_item(
                    item.id,
                    depends_on=[dep for dep in item.depends_on if dep not in gone],
                )
                stripped.append(item.id)
        # Only unblock steps whose dependencies we just changed -- removing a
        # prerequisite can free a dependent, but must not touch unrelated blocks.
        remaining = await store.get_items()
        for item in remaining:
            if (
                item.id in stripped
                and item.status is TaskStatus.blocked
                and not is_blocked(remaining, item)
            ):
                await store.update_item(item.id, status=TaskStatus.pending)
        return len(descendants)

    async def add_subtask(
        self,
        ctx: RunContext[AgentDepsT],
        parent_id: str,
        content: str,
        active_form: str = "",
    ) -> str:
        """Add a subtask under an existing step.

        Args:
            ctx: Framework-provided run context.
            parent_id: Id of the parent step.
            content: The subtask description in imperative form.
            active_form: Optional present-continuous label.
        """
        store = self._resolve(ctx)
        parent = await store.get_item(parent_id)
        if parent is None:
            return f"Parent step with id '{parent_id}' not found."
        if parent.status is TaskStatus.cancelled:
            return f"Cannot add a subtask to '{parent.content}': it is cancelled."
        item = await store.add_item(
            PlanItem(content=content, active_form=active_form, parent_id=parent_id)
        )
        # A new open subtask reopens a completed parent.
        parent_changes = await self._reconcile(store)
        return (
            f"Added subtask '{content}' with id: {item.id} (parent: {parent_id})"
            f"{describe_parent_changes(parent_changes)}"
        )

    async def set_dependency(
        self, ctx: RunContext[AgentDepsT], task_id: str, depends_on_id: str
    ) -> str:
        """Make one step depend on another.

        Args:
            ctx: Framework-provided run context.
            task_id: The step that depends on another (it may become blocked).
            depends_on_id: The prerequisite that must complete first.
        """
        store = self._resolve(ctx)
        items = await store.get_items()
        item = find_item(items, task_id)
        if item is None:
            return f"Step with id '{task_id}' not found."
        dependency = find_item(items, depends_on_id)
        if dependency is None:
            return f"Dependency step with id '{depends_on_id}' not found."
        if task_id == depends_on_id:
            return "A step cannot depend on itself."
        if has_cycle(items, task_id, depends_on_id):
            return "Cannot add dependency: it would create a cycle."
        if depends_on_id in item.depends_on:
            return "Dependency already exists."
        new_depends_on = [*item.depends_on, depends_on_id]
        if (
            not is_terminal(dependency.status)
            and not is_terminal(item.status)
            and item.status is not TaskStatus.blocked
        ):
            await store.update_item(
                task_id, depends_on=new_depends_on, status=TaskStatus.blocked
            )
            return f"Added dependency: '{item.content}' now depends on '{dependency.content}'. Step automatically blocked."
        await store.update_item(task_id, depends_on=new_depends_on)
        # A step already `blocked` keeps its status above, which leaves it blocked on a
        # prerequisite that may be terminal -- and so excluded from `get_available_tasks`
        # with nothing left to free it. Every other write reconciles; so does this one.
        parent_changes = await self._reconcile(store)
        return (
            f"Added dependency: '{item.content}' now depends on '{dependency.content}'."
            f"{describe_parent_changes(parent_changes)}"
        )

    async def get_available_tasks(self, ctx: RunContext[AgentDepsT]) -> str:
        """List steps that can be worked on now (no incomplete dependencies)."""
        items = await self._resolve(ctx).get_items()
        available = [
            item
            for item in items
            if item.status
            not in (TaskStatus.completed, TaskStatus.blocked, TaskStatus.cancelled)
            and not is_blocked(items, item)
            and not open_subtasks(items, item)
        ]
        if not available:
            return "No available steps. All steps are either completed, cancelled, or blocked."
        lines = ["Available steps (no blocking dependencies):"]
        for index, item in enumerate(available, 1):
            lines.append(
                f"{index}. {status_icon(item.status)} [{item.id}] {item.content}"
            )
        return "\n".join(lines)
