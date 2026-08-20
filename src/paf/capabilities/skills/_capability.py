"""paf-owned Skills capability — one capability for all skill functionality.

Unifies two layers (Landon 2026-08-19):
  1. Load-on-demand — `list_skills`/`load_skill` (the load mechanism idea from
     the harness `Skills` / pydantic-ai deferred-capability catalog), so the
     prompt stays thin and only the skills the model actually needs get pulled.
  2. The deterministic trigger — `point_to_skill` re-homing the skill-note-hint
     pointer engine, so the model is *pre-armed* with which skill a message
     needs rather than relying on self-noticing a floating menu.

Indexes a skill-library directory at construction; re-reads SKILL.md bodies on
demand. Kept paf-native (AbstractCapability wrapping a FunctionToolset) in the
same shape as FileSystem/Shell/Git/Web, not a deferred `Capability` bundle — it
SHOULD be tools, per Landon.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from ._toolset import SkillToolset


@dataclass
class Skills(AbstractCapability[AgentDepsT]):
    """List, load, and point-to-skills over a library directory.

    `libraries` is one or more directories whose immediate `SKILL.md` files
    (skills) and plain `.md` files (notes) form the hint corpus. `top_k` is the
    default number of `point_to_skill` pointers per call.
    """

    libraries: tuple[str | Path, ...] = ("skills",)
    """Directories to scan for SKILL.md skills and .md notes."""

    top_k: int = 3
    """Default number of `point_to_skill` pointers to return per call."""

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
