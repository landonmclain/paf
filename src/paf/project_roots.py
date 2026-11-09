"""Resolve API-selected project directories within host-configured policy."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paf.conversations import (
    ConversationMetaStore,
    ProjectStore,
    default_project_store,
    project_for_conversation,
)


class ProjectRootError(RuntimeError):
    """A project root or its policy is unusable; the message is for the user."""


@dataclass(frozen=True)
class ProjectRootPolicy:
    within: tuple[Path, ...]
    """Expanded, resolved host directories that may contain project roots."""


def check_root(root: str, policy: ProjectRootPolicy) -> Path:
    """Resolve symlinks and require an existing directory inside ``within``."""
    try:
        if not isinstance(root, str) or not root.strip():
            raise ValueError("root must be a non-blank string")
        path = Path(root).expanduser()
        if not path.is_absolute():
            raise ValueError("root must be absolute")
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise ValueError("root must be a directory")
        if not any(path.is_relative_to(base) for base in policy.within):
            raise ValueError("root is outside workspace.project_roots.within")
        return path
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProjectRootError(f"Cannot use project root {root!r}: {exc}") from exc


def project_root(
    slug: str, policy: ProjectRootPolicy | None, projects: ProjectStore | None = None
) -> Path | None:
    """Shared assignment/turn check; archived projects still resolve."""
    if policy is None:
        return None
    store = projects if projects is not None else default_project_store()
    entry = store.get(slug)
    try:
        if entry is None:
            raise ProjectRootError("project record is missing")
        if "root" not in entry:
            return None
        return check_root(entry["root"], policy)
    except ProjectRootError as exc:
        raise ProjectRootError(
            f"Project {slug!r}: {exc}. Fix its root with /projects root {slug} <path>, "
            "repair the project record/policy, or clear the assignment with /project none."
        ) from exc


def conversation_root(
    agent: str,
    conversation_id: str | None,
    *,
    profile: str | None,
    policy: ProjectRootPolicy | None,
    projects: ProjectStore | None = None,
    meta: ConversationMetaStore | None = None,
) -> Path | None:
    """Checked root for this conversation, or None to use the spec's roots."""
    if policy is None:
        return None
    slug = project_for_conversation(agent, conversation_id, profile=profile, store=meta)
    return project_root(slug, policy, projects) if slug else None


_ROOT_KEYS = {
    "filesystem_config": "root_dir",
    "shell_config": "cwd",
    "python_config": "cwd",
    "git_config": "root",
    "codebase_map_config": "scan_root",
}


def with_project_root(
    configs: dict[str, Any], root: Path, *, keep_existing: bool = False
) -> dict[str, Any]:
    """Copy configs with string root overrides; preserve every non-root option."""
    result = dict(configs)
    for block, key in _ROOT_KEYS.items():
        config = dict(configs.get(block) or {})
        if not keep_existing or not config.get(key):
            config[key] = str(root)
        result[block] = config
    return result


def is_git_work_tree(root: Path) -> bool:
    """Recognize repos, subfolders, and worktrees without inspecting ``.git``."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except (OSError, subprocess.TimeoutExpired):
        return False
