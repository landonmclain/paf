"""paf-owned persistent memory -- the durable-facts `Memory` capability.

The user-role ``MEMORY.md`` notebook is injected each request, tagged with the
harness memory marker (``pydantic-ai-harness.memory.v1:<scope_hash>``) so any
reader (the search corpus) recognizes and strips it. The strip sites
(``capabilities/_markers``, ``search/_render``) already drop injected memory.

The M2 three-layer model: the **owner** appends immutable facts to the
append-only **journal** (``JournalStore``); the **organizer** reconciles the
journal tail into the curated **index** (``IndexStore`` / ``IndexEntry``); and
the **index renderer** materializes the root ``MEMORY.md`` (``write_index_render``)
that the assistant's read-only ``JournalViewStore`` injects.

The legacy rehomed pydantic-ai builtin surface (the ``FileStore`` SQLite-journal
backend, ``MarkdownStorage``, the 4-tool ``MemoryToolset``, and the
``InMemoryStore`` process-lifetime placeholder) was removed — none of it is used
by the M2 journal model or by benchmarking/prod. A ``Memory`` always carries a
concrete backend (the ``store`` arg is required); the M2 path supplies the
read-only ``JournalViewStore`` of the rendered journal.

- ``_store``     -- ``MemoryStore``/``SearchableMemoryStore`` protocols, shared
                    record types, ``lexical_search``, validation.
- ``_journal``   -- append-only ``JournalStore`` (source of truth).
- ``_organize``  -- ``run_organizer`` + ``IndexStore`` (curated index).
- ``_render``    -- ``write_render`` / ``write_index_render`` (MEMORY.md).
- ``_view``      -- ``JournalViewStore`` (read-only injection seam).
- ``_toolset``   -- ``JournalAppendToolset`` (the owner's only tool) + render helpers.
- ``_capability``-- the ``Memory`` capability.
"""

from __future__ import annotations

from ._capability import Memory
from ._journal import MEMORY_TYPES, JournalStore, MemoryRecord
from ._organize import (
    ConfirmDecision,
    IndexEntry,
    IndexStore,
    NewDecision,
    Stage,
    SupersedeDecision,
    render_index_text,
    render_journal_tail,
    run_organizer,
)
from ._render import (
    render_index,
    render_journal_memory,
    write_index_render,
    write_render,
)
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
    list_subfiles,
    normalize_filename,
    render_memory_prompt,
)

__all__ = [
    "ConfirmDecision",
    "IndexEntry",
    "IndexStore",
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
    "NewDecision",
    "SearchableMemoryStore",
    "Stage",
    "SupersedeDecision",
    "lexical_search",
    "list_subfiles",
    "normalize_filename",
    "render_index",
    "render_index_text",
    "render_journal_memory",
    "render_journal_tail",
    "render_memory_prompt",
    "run_organizer",
    "validate_store_path",
    "validate_store_prefix",
    "write_index_render",
    "write_render",
]
