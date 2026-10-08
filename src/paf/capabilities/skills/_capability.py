"""paf-owned Skills capability — one capability for all skill functionality.

Load-on-demand — `list_skills`/`load_skill`/`list_skill_files`/
`read_skill_file` (the load mechanism idea from pydantic-ai-harness's `Skills` /
pydantic-ai deferred-capability catalog), so the prompt stays thin and only the
skills the model actually needs get pulled. Plus `create_skill`/`edit_skill`.

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
    plain notes are a vault-capability concern.
    """

    libraries: tuple[str | Path, ...] = ("skills",)
    """Directories to scan for SKILL.md skills."""

    def __post_init__(self) -> None:
        if isinstance(self.libraries, (str, Path)):
            self.libraries = (self.libraries,)

    def get_toolset(self) -> SkillToolset:
        """Build and return the skills toolset."""
        return SkillToolset(libraries=[str(p) for p in self.libraries])
