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
journal. The public surface is intentionally small: the capability, the journal
store and its record type, the override sidecar, the render helpers, and the
owner's append toolset. Sub-agents, toolsets, and the view reader are reached
through their private modules.
"""

from __future__ import annotations

from ._capability import Memory
from ._journal import JournalStore, MemoryRecord
from ._overrides import OverrideStore
from ._render import render_journal_memory, write_render
from ._toolset import JournalAppendToolset

__all__ = [
    "JournalAppendToolset",
    "JournalStore",
    "Memory",
    "MemoryRecord",
    "OverrideStore",
    "render_journal_memory",
    "write_render",
]
