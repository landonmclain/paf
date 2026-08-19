"""Turn Summary Index -- M2's first feature, as an installable pydantic-ai capability.

A one-line-per-turn index injected into context (recognition beats recall) with a
`read_turn` tool that resolves a ref back into pydantic-ai persistence through the
`HistorySource` seam. Works with any pydantic-ai persistence backend.

See `turn_index._capability` for the composition (source + own append-only index)
and the write/read separation.
"""

from pydantic_ai_harness.conversation_search._source import (
    HistorySource,
    SnapshotHistorySource,
)

from ._builders import build_turn_index
from ._capability import TurnIndexCapability
from ._generate import (
    Turn,
    summarize_turn,
    summarize_turn_append,
    this_turn_id,
    this_turn_slice,
    user_anchor_indices,
)
from ._store import TurnIndexEntry, TurnIndexStore
from ._summarizer import (
    DEFAULT_BUDGET_CHARS,
    MAX_KEY_TERMS,
    TurnSummarizer,
    TurnSummary,
)
from ._toolset import TurnIndexToolset

__all__ = [
    'HistorySource',
    'SnapshotHistorySource',
    'TurnIndexEntry',
    'TurnIndexStore',
    'TurnIndexCapability',
    'TurnIndexToolset',
    'build_turn_index',
    'DEFAULT_BUDGET_CHARS',
    'MAX_KEY_TERMS',
    'TurnSummarizer',
    'TurnSummary',
    'Turn',
    'summarize_turn',
    'summarize_turn_append',
    'this_turn_id',
    'this_turn_slice',
    'user_anchor_indices',
]
