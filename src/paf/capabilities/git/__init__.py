"""Git capability -- safe repository inspection and checkpointing for agents.

paf-owned adaptation of `coding_agent/app/tools/git_tools.py`, surfaced through
the standard capability seam (`AbstractCapability` + `FunctionToolset` + registry
+ profile allowlist + builder). Exposes `git_status`, `git_changed_files`,
`git_diff` (chunked/paged), `git_checkpoint`, and `git_restore_paths`.
"""

from ._capability import Git
from ._toolset import GitToolset

__all__ = [
    'Git',
    'GitToolset',
]
