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
- The store backends kept are ``InMemoryStore`` (process-lifetime), ``FileStore``
  (plain-Markdown files + a hidden SQLite journal), and ``MarkdownStorage``
  (plain `.md` files + a flat ``.memory-state.json`` CAS sidecar, written with
  the foundation's atomic helpers). SQLite and Postgres stores are dropped (not
  used by paf).
- ``from_spec`` accepts ``backend='memory' | 'file' | 'markdown'``. Process
  memory ignores ``directory``; ``file`` and ``markdown`` use it as the store
  root (default ``.agent-memory``).

- ``_store``     -- ``MemoryStore``/``SearchableMemoryStore`` protocols, CAS
                    mutation machinery, ``InMemoryStore``, ``FileStore``.
- ``_markdown``  -- ``MarkdownStorage``: flat-file backend. Bodies real ``.md``,
                    version/CAS in a flat sidecar, atomic writes via the
                    foundation helpers. Deliberately does NOT ride the
                    ``StorageService`` protocol (that seam models the
                    derived/index shape; memory is authoritative/document).
- ``_toolset``   -- ``MemoryToolset`` (write/read/delete/search_memory).
- ``_capability``-- the ``Memory`` capability (drop-in for the harness).
"""

from __future__ import annotations

from ._capability import Memory
from ._markdown import MarkdownStorage
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
    "MarkdownStorage",
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
