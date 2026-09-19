"""paf-owned persistent memory -- the durable-facts ``Memory`` capability.

The user-role ``MEMORY.md`` notebook is injected into context, tagged with the
harness memory marker (``pydantic-ai-harness.memory.v1:<scope_hash>``) so any
reader (the search corpus) recognizes and strips it. The strip sites
(``capabilities/_markers``, ``search/_render``) already drop injected memory.

Three layers, journal -> index -> render:

- **journal** (``_journal``) -- the append-only ``JournalStore``, the source of
  truth. The owner appends immutable facts; nothing is edited or deleted.
- **index** (``_index``) -- a derived, deterministic projection of the journal
  (``memory-index.json``): which journal line is the live copy of each fact and
  which are its deduped predecessors. Regenerable from the journal alone, so it
  can never drift into a parallel source of truth.
- **render** (``_render``) -- ``write_render`` materializes ``MEMORY.md``
  deterministically, tier-sorted, from the index when present (else the raw
  journal). The assistant's read-only ``JournalViewStore`` injects it.

Curation runs through an append-only override sidecar (``_overrides``): the
assistant proposes archives (pending), a reviewer resolves them. The owner
(``_consolidate``) and the reviewer (``_review``) are background sub-agents,
fired by the watermark trigger (``_watermark``) through the queues in
``_background``.

A ``Memory`` always carries a concrete backend (the ``store`` arg is required);
the read path supplies the read-only ``JournalViewStore`` of the rendered
journal.

- ``_store``      -- ``MemoryStore``/``SearchableMemoryStore`` protocols, shared
                    record types, ``lexical_search``, validation.
- ``_journal``    -- append-only ``JournalStore`` (source of truth).
- ``_index``      -- derived ``MemoryIndex`` of surviving facts (dedup winners).
- ``_render``     -- ``write_render`` / ``render_journal_memory`` (MEMORY.md).
- ``_view``       -- ``JournalViewStore`` (read-only injection seam).
- ``_overrides``  -- append-only override sidecar (assistant suggestions +
                    reviewer verdicts), folded into the render.
- ``_toolset``    -- the tool surfaces: ``JournalAppendToolset`` (owner),
                    ``OverrideToolset`` (assistant), ``ReviewerToolset``.
- ``_capability`` -- the ``Memory`` capability.
- ``_consolidate``-- the owner sub-agent (sole writer: appends facts).
- ``_review``     -- the reviewer sub-agent (curates the rendered memory).
- ``_watermark``  -- the persisted-consumption watermark + trigger predicate.
- ``_background`` -- the consolidation + review background queues.
- ``_window``     -- per-agent memory directory resolution.
- ``_atomic``     -- re-export shim to ``_storage._atomic``.
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
    MemoryMergeResult,
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
    "MemoryMergeResult",
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
