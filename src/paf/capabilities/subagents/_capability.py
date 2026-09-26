"""``Subagents`` — paf-owned delegation capability (replaces the harness builtin).

Holds a fixed roster of leaf agents (built from ``specs/sub_agents/`` by
``_build_subagents``) and exposes the delegation tools plus a static,
cache-stable listing of the available subagents in the instructions.

Like the harness builtin it replaces, it is NOT spec-serializable (it holds live
``Agent`` instances), so it is only ever built in Python and supplied through
the ``capabilities=`` seam, gated by the profile allowlist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai.capabilities import AbstractCapability

from ._roster import Leaf
from ._toolset import SubagentToolset


@dataclass
class Subagents(AbstractCapability[Any]):
    """Delegate self-contained tasks to named child agents."""

    leaves: list[Leaf] = field(default_factory=list)
    """The delegatable children, keyed at runtime by ``Leaf.name``."""

    processes_root: str | Path = "data/processes"
    """Root of the durable task store (``data/processes`` in the app)."""

    agent: str = "assistant"
    """paf agent name — scopes the task store and the process-feed emitter."""

    def __post_init__(self) -> None:
        names = [leaf.name for leaf in self.leaves]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate subagent names: {sorted(dupes)}")

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable: holds live ``Agent`` instances."""
        return None

    def get_instructions(self) -> str | None:
        """Static listing of the available subagents and how to use them."""
        if not self.leaves:
            return None
        listing = "\n".join(
            f"- {leaf.name}: {leaf.description}"
            if leaf.description
            else f"- {leaf.name}"
            for leaf in self.leaves
        )
        return (
            "You can delegate self-contained tasks to these subagents with the "
            "`delegate_task` tool. Each runs in its own fresh context and does not "
            "see this conversation, so pass everything it needs — prefer file paths "
            "over pasted content. Every result is saved; `check_task` re-reads a "
            f"task's output.\n\nAvailable subagents:\n{listing}"
        )

    def get_toolset(self) -> SubagentToolset | None:
        """The delegation tools, or ``None`` when no subagents are configured."""
        if not self.leaves:
            return None
        return SubagentToolset(
            leaves={leaf.name: leaf for leaf in self.leaves},
            processes_root=self.processes_root,
            agent=self.agent,
        )
