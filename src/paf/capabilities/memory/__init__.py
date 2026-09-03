"""paf-owned persistent memory -- the durable-facts `Memory` capability.

The user-role ``MEMORY.md`` notebook is injected each request, tagged with the
harness memory marker (``pydantic-ai-harness.memory.v1:<scope_hash>``) so any
reader (the search corpus) recognizes and strips it. The strip sites
(``capabilities/_markers``, ``search/_render``) already drop injected memory.

The M2 memory model: the **owner** appends immutable facts to the append-only
**journal** (``JournalStore``); the **renderer** (``write_render``) materializes
the root ``MEMORY.md`` deterministically from the journal, which the assistant's
read-only ``JournalViewStore`` injects. There is **no curated index** — the
organizer / index / consolidation-second-pass that maintained ``memory-index.json``
were **unhooked 2026-08-25 and archived** to ``archive/organizer/``; the journal
is the sole source of truth.

The legacy rehomed pydantic-ai builtin surface (the ``FileStore`` SQLite-journal
backend, ``MarkdownStorage``, the 4-tool ``MemoryToolset``, and the
``InMemoryStore`` process-lifetime placeholder) was removed — none of it is used
by the M2 journal model or by benchmarking/prod. A ``Memory`` always carries a
concrete backend (the ``store`` arg is required); the M2 path supplies the
read-only ``JournalViewStore`` of the rendered journal.

- ``_store``     -- ``MemoryStore``/``SearchableMemoryStore`` protocols, shared
                    record types, ``lexical_search``, validation.
- ``_journal``   -- append-only ``JournalStore`` (source of truth).
- ``_render``    -- ``write_render`` / ``render_journal_memory`` (MEMORY.md).
- ``_view``      -- ``JournalViewStore`` (read-only injection seam).
- ``_toolset``   -- ``JournalAppendToolset`` (the owner's only tool) + ``OverrideToolset``.
- ``_overrides`` -- append-only override sidecar: the assistant's narrow
                    ``memory_archive_suggestion`` curation (pending), resolved
                    later by a reviewer's ``memory_review`` at Phase 3. A
                    separate display layer over the faithful index render.
- ``_capability``-- the ``Memory`` capability.
- ``archive/organizer/`` -- ARCHIVED curated index + organizer + consolidation
                            second pass (unhooked; kept importable for revival).
"""

from __future__ import annotations

from ._capability import Memory
from ._journal import MEMORY_TYPES, JournalStore, MemoryRecord
from ._overrides import OverrideStore, fold_overrides, load_overrides
from ._render import read_render, render_journal_memory, write_render
from ._store import (
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
    JournalAppendToolset,
    MemoryWriteResult,
    OverrideResult,
    OverrideToolset,
    ReviewerResult,
    ReviewerToolset,
    list_subfiles,
    normalize_filename,
    render_memory_prompt,
)

__all__ = [
    "JournalAppendToolset",
    "JournalStore",
    "MAIN_FILENAME",
    "MEMORY_TYPES",
    "Memory",
    "MemoryConflictError",
    "MemoryFile",
    "MemoryMutation",
    "MemoryOperation",
    "MemoryOperationConflictError",
    "MemoryRecord",
    "MemorySearchMatch",
    "MemorySearchResult",
    "MemoryStore",
    "MemoryWriteResult",
    "OverrideResult",
    "OverrideStore",
    "OverrideToolset",
    "ReviewerResult",
    "ReviewerToolset",
    "SearchableMemoryStore",
    "fold_overrides",
    "lexical_search",
    "list_subfiles",
    "load_overrides",
    "normalize_filename",
    "read_render",
    "render_journal_memory",
    "render_memory_prompt",
    "validate_store_path",
    "validate_store_prefix",
    "write_render",
]
