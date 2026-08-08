"""Turn Summary Index -- M2's first feature, as an installable pydantic-ai capability.

A one-line-per-turn index injected into context (recognition beats recall) with a
`read_turn` tool that resolves a ref back into pydantic-ai persistence through the
`HistorySource` seam. Works with any pydantic-ai persistence backend.

See `turn_index._capability` for the composition (source + own append-only index).
"""

from pydantic_ai_harness.conversation_search._source import (
    HistorySource,
    SnapshotHistorySource,
)

from ._capability import TurnIndexCapability
from ._generate import (
    CollectorStore,
    CursorStore,
    EventRecord,
    summarize_turn,
)
from ._store import TurnIndexEntry, TurnIndexStore
from ._toolset import TurnIndexToolset

__all__ = [
    'HistorySource',
    'SnapshotHistorySource',
    'TurnIndexEntry',
    'TurnIndexStore',
    'TurnIndexCapability',
    'TurnIndexToolset',
    'CollectorStore',
    'CursorStore',
    'EventRecord',
    'summarize_turn',
]