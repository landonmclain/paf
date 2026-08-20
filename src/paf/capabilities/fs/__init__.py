"""paf-owned filesystem capability (copied verbatim from pydantic-ai-harness, MIT).

Scoped file access rooted at a directory: read/write/edit with `expected_hash`
optimistic-concurrency (stale-write) guarding, listing, content regex search,
glob find, directory creation, and metadata — all confined to the root with
symlink resolution and allow/deny/protected glob filtering.

- `_toolset` — the `FileSystemToolset` (the function tools).
- `_capability` — the `FileSystem` capability (wraps the toolset).

Ported so paf owns the surface and can wire it through the capability-toolset
seam (registry + profile + builder) instead of using the harness directly.
"""

from __future__ import annotations

from ._capability import FileSystem
from ._toolset import FileSystemToolset

__all__ = ["FileSystem", "FileSystemToolset"]
