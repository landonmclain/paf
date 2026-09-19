"""paf-owned Skills capability — one capability for all skill functionality.

Unifies two layers:
  1. Load-on-demand — `list_skills`/`load_skill`/`list_skill_files`/
     `read_skill_file` (the load mechanism idea from pydantic-ai-harness's
     `Skills` / pydantic-ai deferred-capability catalog), so the prompt stays
     thin and only the skills the model actually needs get pulled.
  2. The hint pointer engine — `point_skills` (the skill-note-hint pointer), so
     the agent is **pre-armed** with which skill a message needs. This is an
     **internal primitive** for the auto per-turn hint layer, NOT an
     agent-facing tool — a volitional search the model must remember to call
     never fires reliably.

Indexes a skill-library directory at construction; re-reads SKILL.md bodies on
demand. paf-native (AbstractCapability wrapping a FunctionToolset), in the same
shape as FileSystem/Shell/Git/Web.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from ._toolset import SkillToolset


@dataclass
class Skills(AbstractCapability[AgentDepsT]):
    """List, load, and read resources over a skill-library tree.

    `libraries` is one or more directories containing skills, each a directory
    with a `SKILL.md` (optionally nested under a category), plus skill-scoped
    resources (`references/`/`scripts/`/`templates/`/`assets/`). Skills only —
    plain notes are a vault-capability concern. `top_k` is the default number of
    `point_skills` pointers the hint layer emits per call.
    """

    libraries: tuple[str | Path, ...] = ("skills",)
    """Directories to scan for SKILL.md skills."""

    top_k: int = 3
    """Default number of `point_skills` hints the hint layer returns per call."""

    def __post_init__(self) -> None:
        if isinstance(self.libraries, (str, Path)):
            self.libraries = (self.libraries,)
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {self.top_k!r}")

    def get_toolset(self) -> SkillToolset:
        """Build and return the skills toolset."""
        return SkillToolset(
            libraries=[str(p) for p in self.libraries],
            top_k=self.top_k,
        )
