"""paf-owned full-context search -- the wide `search_conversation_history` tool.

This package re-homes the "last-resort" rung of the recall ladder so the WHOLE
recall surface is owned by paf with one strip convention. The harness
`ConversationSearch` (which we previously borrowed) renders `UserPromptPart`
blocks verbatim and has no marker-aware `TextContent` exclusion, so injected
`<turn-index>` / `<memory>` blocks leaked into its BM25 corpus. Here the corpus
is rendered with the injected-artifact strip (`_render`), so search never
surfaces derived artifacts.

- `_ranking` — dependency-free BM25 core (re-homed from the harness, MIT).
- `_render` — message rendering with the injected-artifact strip.
- `_toolset` — the `search_conversation_history` tool.
- `_capability` — the `ConversationSearch` capability (drop-in for the harness).
"""

from __future__ import annotations

from pydantic_ai_harness.conversation_search._source import (
    HistorySource,
    SnapshotHistorySource,
)

from ._capability import ConversationSearch
from ._toolset import ConversationSearchToolset

__all__ = [
    "HistorySource",
    "SnapshotHistorySource",
    "ConversationSearch",
    "ConversationSearchToolset",
]
