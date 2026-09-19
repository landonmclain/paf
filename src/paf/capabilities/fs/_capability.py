"""paf-owned filesystem capability (originally copied from pydantic-ai-harness, MIT).

Provides sandboxed file system access scoped to a root directory. Copy of the
harness `FileSystem` capability so paf owns the surface; imported and wired
through paf's capability-toolset seam (registry + profile + builder) rather than
used directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from ._toolset import FileSystemToolset

_DEFAULT_PROTECTED: list[str] = [
    ".git/*",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "**/secrets*",
]


@dataclass
class FileSystem(AbstractCapability[AgentDepsT]):
    """File system access scoped to a root directory.

    All paths are resolved relative to `root_dir`. Traversal above the root
    is rejected. Symlinks are resolved before authorization.
    """

    root_dir: str | Path = "."
    """Root directory for all file operations. Defaults to the current directory."""

    allowed_patterns: Sequence[str] = field(default_factory=list[str])
    """If non-empty, only paths matching at least one glob pattern are accessible."""

    denied_patterns: Sequence[str] = field(default_factory=list[str])
    """Paths matching any of these glob patterns are rejected."""

    protected_patterns: Sequence[str] = field(
        default_factory=lambda: list(_DEFAULT_PROTECTED)
    )
    """Paths matching these patterns are read-only (writes are rejected).

    Defaults to protecting `.git/`, `.env`, key files, and secrets.
    Set to an empty list to disable protection.
    """

    max_read_lines: int = 2000
    """Maximum number of lines returned by a single `read_file` call."""

    max_search_results: int = 1000
    """Maximum number of matches returned by `search_files`."""

    max_find_results: int = 1000
    """Maximum number of matches returned by `find_files`."""

    # search_files result-bound guards: a content search for a common string
    # walked the whole ~/ tree, read multi-MB JSON journals/snapshots in full,
    # and returned a 13MB / 6-minute result. Bounds below are the sensible
    # defaults.
    max_search_file_bytes: int = 200 * 1024
    """Skip files larger than this in `search_files` (read via read_file instead)."""

    search_excerpt_chars: int = 200
    """Return at most this many chars of each matching `search_files` line."""

    max_search_total_bytes: int = 256 * 1024
    """Hard cap on the total `search_files` result string."""

    def __post_init__(self) -> None:
        # Runtime validation: dataclass field annotations are advisory, not enforced.
        # A config-driven caller could pass a string that would otherwise propagate.
        values: dict[str, Any] = {
            "max_read_lines": self.max_read_lines,
            "max_search_results": self.max_search_results,
            "max_find_results": self.max_find_results,
            "max_search_file_bytes": self.max_search_file_bytes,
            "search_excerpt_chars": self.search_excerpt_chars,
            "max_search_total_bytes": self.max_search_total_bytes,
        }
        for name, value in values.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    def get_toolset(self) -> FileSystemToolset[AgentDepsT]:
        """Build and return the filesystem toolset."""
        return FileSystemToolset[AgentDepsT](
            root_dir=Path(self.root_dir),
            allowed_patterns=self.allowed_patterns,
            denied_patterns=self.denied_patterns,
            protected_patterns=self.protected_patterns,
            max_read_lines=self.max_read_lines,
            max_search_results=self.max_search_results,
            max_find_results=self.max_find_results,
            max_search_file_bytes=self.max_search_file_bytes,
            search_excerpt_chars=self.search_excerpt_chars,
            max_search_total_bytes=self.max_search_total_bytes,
        )
