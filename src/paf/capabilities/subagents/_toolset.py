"""Model-facing subagent tools: ``delegate_task`` and ``check_task``.

``delegate_task(agent_name, task)`` runs a named leaf to completion and returns
its full output. Every delegation is recorded in the conversation's task store
and its output saved to a ``.md`` file (see ``_store``), so ``check_task`` can
page a long result back later. Failures come back as a soft observation (a
normal tool result), never an exception that would abort the parent run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.toolsets import FunctionToolset

from ._roster import Leaf
from ._runner import run_subagent
from ._store import SubagentTaskStore

CHECK_MAX_CHARS = 8000
"""Default page size for ``check_task`` output slices."""

_MAX_TRACKED_RUNS = 64
"""Per-run call counters kept in memory (oldest dropped first)."""


def _soft_failure(agent_name: str, detail: str) -> str:
    return (
        f"Subagent {agent_name!r} failed: {detail}. Treat this as a recoverable "
        f"observation and decide from existing evidence."
    )


class SubagentToolset(FunctionToolset[Any]):
    """Delegation tools over a fixed roster of leaf agents."""

    def __init__(
        self,
        *,
        leaves: dict[str, Leaf],
        processes_root: str | Path,
        agent: str,
    ) -> None:
        super().__init__()
        self._leaves = leaves
        self._processes_root = Path(processes_root)
        self._agent = agent
        # run_id -> {leaf name -> delegations this run}; enforces max_calls_per_run.
        self._call_counts: dict[str, dict[str, int]] = {}
        self.add_function(self.delegate_task, name="delegate_task")
        self.add_function(self.check_task, name="check_task")

    # -- helpers ---------------------------------------------------------------

    def _store(self, ctx: RunContext[Any]) -> SubagentTaskStore:
        return SubagentTaskStore(
            self._processes_root,
            self._agent,
            conversation_id=getattr(ctx, "conversation_id", None),
        )

    def _leaf(self, agent_name: str) -> Leaf:
        leaf = self._leaves.get(agent_name)
        if leaf is None:
            raise ModelRetry(
                f"Unknown subagent {agent_name!r}. Available: "
                f"{', '.join(sorted(self._leaves)) or '(none)'}."
            )
        return leaf

    def _over_budget(self, run_id: str, leaf: Leaf) -> bool:
        """Count this delegation; True when it exceeds the leaf's per-run cap."""
        if leaf.policy.max_calls_per_run is None:
            return False
        counts = self._call_counts.setdefault(run_id, {})
        while len(self._call_counts) > _MAX_TRACKED_RUNS:
            self._call_counts.pop(next(iter(self._call_counts)))
        counts[leaf.name] = counts.get(leaf.name, 0) + 1
        return counts[leaf.name] > leaf.policy.max_calls_per_run

    # -- tools -----------------------------------------------------------------

    async def delegate_task(
        self, ctx: RunContext[Any], agent_name: str, task: str
    ) -> str:
        """Delegate a self-contained task to a named subagent and return its output.

        The subagent runs in its own fresh context and does NOT see this
        conversation, so ``task`` must contain everything it needs — prefer
        passing file paths over pasting content. The full output is also saved
        and can be re-read later with ``check_task``.

        Args:
            agent_name: Which subagent to run (see the available subagents list).
            task: The complete, self-contained instruction for the subagent.
        """
        leaf = self._leaf(agent_name)
        if leaf.policy.background == "required":
            raise ModelRetry(
                f"Subagent {agent_name!r} only runs in the background; "
                f"call delegate_task with background=true."
            )
        if self._over_budget(ctx.run_id or "", leaf):
            return (
                f"Subagent {agent_name!r} reached its call budget "
                f"({leaf.policy.max_calls_per_run} per run). Treat this as a "
                f"recoverable observation and decide from existing evidence."
            )
        store = self._store(ctx)
        rec = store.create(
            agent_name=leaf.name, task=task, mode="sync", parent_run_id=ctx.run_id
        )
        final, output = await run_subagent(rec, leaf, store)
        if output is None:
            return _soft_failure(
                agent_name, str(final.get("detail") or "unknown error")
            )
        return output

    async def check_task(
        self,
        ctx: RunContext[Any],
        task_id: str,
        offset: int = 0,
        max_chars: int = CHECK_MAX_CHARS,
    ) -> str:
        """Get a subagent task's status and (when finished) a page of its output.

        Args:
            task_id: The id of the task (from delegate_task or a completion message).
            offset: Character offset into the output to start from.
            max_chars: Maximum characters of output to return.
        """
        store = self._store(ctx)
        rec = store.get(task_id)
        if rec is None:
            return f"[Error: unknown task id {task_id!r} in this conversation]"
        status = rec.get("status")
        header = f"Task {task_id} ({rec.get('agent')}): {status}"
        if status != "done":
            detail = rec.get("detail")
            return f"{header}" + (f" — {detail}" if detail else "")
        output = store.read_output(task_id)
        if output is None:
            return f"{header} — output file missing ({rec.get('output_path')})"
        total = len(output)
        start = max(0, offset)
        end = min(total, start + max(1, max_chars))
        page = output[start:end]
        more = (
            f"\n\n[... {total - end} more chars; call check_task('{task_id}', "
            f"offset={end})]"
            if end < total
            else ""
        )
        return f"{header} — output chars {start}-{end} of {total}:\n\n{page}{more}"
