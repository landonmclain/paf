"""Git capability that provides safe repository inspection and checkpoints.

Rooted at a workspace directory so all git commands and path validation are
confined there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability

from ._toolset import DEFAULT_DIFF_CHUNK_LINES, GitToolset


@dataclass
class Git(AbstractCapability):
    """Safe git inspection + checkpoint tools for an agent.

    Exposes five tools (`git_status`, `git_changed_files`, `git_diff`,
    `git_checkpoint`, `git_restore_paths`) that operate on the repository
    rooted at `root`. All paths are validated to stay inside `root`; the two
    write tools refuse empty path lists (no broad `git add -A`).
    """

    root: str | Path = "."
    """Workspace root all git commands operate within relative to."""

    timeout: float = field(default=60.0)
    """Maximum seconds for any single git subprocess."""

    diff_default_lines: int = field(default=DEFAULT_DIFF_CHUNK_LINES)
    """Default max lines for git_diff before paging is needed."""

    def get_toolset(self) -> GitToolset:
        """Build and return the git toolset."""
        return GitToolset(
            root=Path(self.root),
            timeout=self.timeout,
        )
