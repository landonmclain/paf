"""paf-owned persistent memory -- the durable-facts `Memory` capability.

This package re-homes ``pydantic_ai_harness.memory`` into paf so the whole
recall surface (index, read_turn, wide search, memory) shares one marker-strip
convention. It is a near-copy of the harness `Memory` capability (MIT): the
user-role ``MEMORY.md`` notebook is injected each request, tagged with the
harness memory marker (``pydantic-ai-harness.memory.v1:<scope_hash>``) so any
reader (read_turn, the search corpus) recognizes and strips it. Keeping the
harness marker base means the existing strip sites (``turn_index/_markers``,
``search/_render``) already drop injected memory -- no strip churn.

Decisions vs the harness original:
- Only the ``file`` (``FileStore``: plain-Markdown files + a hidden SQLite
  journal) and in-memory (``InMemoryStore``) backends are kept. SQLite and
  Postgres stores are dropped (not used by paf; the FileStore journal already
  uses sqlite internally).
- ``from_spec`` accepts ``backend='memory' | 'file'`` only.

- ``_store``     -- ``MemoryStore``/``SearchableMemoryStore`` protocols, CAS
                    mutation machinery, ``InMemoryStore``, ``FileStore``.
- ``_toolset``   -- ``MemoryToolset`` (write/read/delete/search_memory).
- ``_capability``-- the ``Memory`` capability (drop-in for the harness).
"""

from __future__ import annotations

from ._capability import Memory
from ._store import (
    FileStore,
    InMemoryStore,
    MemoryConflictError,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemoryOperationConflictError,
    MemorySearchMatch,
    MemorySearchResult,
    MemoryStore,
    SearchableMemoryStore,
    lexical_search,
    validate_store_path,
    validate_store_prefix,
)
from ._toolset import (
    MAIN_FILENAME,
    MemoryDeleteResult,
    MemorySearchMatchResult,
    MemorySearchResponse,
    MemoryToolset,
    MemoryWriteResult,
    list_subfiles,
    normalize_filename,
    render_memory_prompt,
)

__all__ = [
    "FileStore",
    "InMemoryStore",
    "MAIN_FILENAME",
    "Memory",
    "MemoryConflictError",
    "MemoryDeleteResult",
    "MemoryFile",
    "MemoryMutation",
    "MemoryOperation",
    "MemoryOperationConflictError",
    "MemorySearchMatch",
    "MemorySearchMatchResult",
    "MemorySearchResponse",
    "MemorySearchResult",
    "MemoryStore",
    "MemoryToolset",
    "MemoryWriteResult",
    "SearchableMemoryStore",
    "lexical_search",
    "list_subfiles",
    "normalize_filename",
    "render_memory_prompt",
    "validate_store_path",
    "validate_store_prefix",
]
