"""The paf-owned `ConversationSearch` capability: BM25 recall over persisted history.

A drop-in replacement for `pydantic_ai_harness.conversation_search.ConversationSearch`
that paf owns end-to-end. Same public surface (`source`, `scope`, `tool_id`,
`max_matches`, `context_lines`, `bm25_k1`, `bm25_b`, `add_instructions`) and same
`get_toolset` / `get_instructions` behavior, but the search corpus is rendered with
the injected-artifact strip (see `_render`), so marker-tagged `<turn-index>` /
`<memory>` blocks never leak into the BM25 index.

The harness's `ConversationSearch` renders `UserPromptPart` blocks verbatim and
has no marker-aware `TextContent` exclusion. This capability is the record-owning
version: the whole recall surface (index search, read_turn, wide search) shares
one strip convention.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.agent.abstract import AgentInstructions
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai_harness.conversation_search._source import HistorySource

from ._toolset import ConversationSearchToolset, SearchScope

_INSTRUCTIONS = (
    "A `search_conversation_history` tool can retrieve exact details from persisted history: "
    "earlier turns that context compaction has since dropped from the live context, and past "
    "runs persisted in the same store. Reach for it when the current context, or a compaction "
    "summary, lacks a detail you need."
)


@dataclass
class ConversationSearch(AbstractCapability[AgentDepsT]):
    """Search persisted conversation history with a dependency-free BM25 tool.

    Persists nothing itself: it reads whatever history a persistence capability
    already stores, through a `HistorySource`. Pair it with `StepPersistence`
    sharing the same store, and the model can recall what compaction dropped
    from the live context as well as anything from past runs.

    ```python
    store = FileStepStore(directory="data/persist")
    agent = Agent(
        "openai:gpt-5",
        capabilities=[
            StepPersistence(store=store),
            ConversationSearch(SnapshotHistorySource(store), scope="conversation"),
        ],
    )
    ```
    """

    source: HistorySource
    """Where the search corpus comes from. Use `SnapshotHistorySource` over the
    store a `StepPersistence` capability writes to."""

    scope: SearchScope = "all"
    """How much of the store one search may reach.

    `all` searches every run the source enumerates. `conversation` restricts the
    corpus to runs whose `conversation_id` matches the calling run's, which a store
    shared across users or tenants needs. Under `conversation`, a run with no
    `conversation_id` searches nothing and the tool says so, rather than falling
    back to every other unlabelled run.
    """

    tool_id: str = "conversation-search"
    """Toolset id for the `search_conversation_history` tool."""

    max_matches: int = 10
    """Maximum number of matching excerpts the search tool returns. Must be non-negative."""

    context_lines: int = 5
    """Number of surrounding lines shown around each search match. Must be non-negative."""

    bm25_k1: float = 1.5
    """BM25 term-frequency saturation, non-negative. Lucene's `BM25Similarity` uses 1.2."""

    bm25_b: float = 0.75
    """BM25 length-normalization, between 0.0 and 1.0 (Lucene/Elasticsearch default)."""

    add_instructions: bool = False
    """Emit a short instruction telling the model the recall tool exists."""

    memory_directory: str | None = None
    """When set, `read_turn('#N')` resolves memory id N (a journal line) to the
    turn(s) the fact came from. Set by the builder when the agent has Memory."""

    include_read_turn: bool = True
    """Expose the `read_turn` full-turn expansion tool alongside search.

    When `False`, only `search_conversation_history` is registered (a pure-BM25
    arm that must answer from truncated excerpts) and the read_turn-keyed search
    loop guard is bypassed, since `read_turn` no longer exists to clear its streak.
    """

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide the `search_conversation_history` tool over the source."""
        return ConversationSearchToolset(
            self.source,
            tool_id=self.tool_id,
            max_matches=self.max_matches,
            context_lines=self.context_lines,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
            scope=self.scope,
            include_read_turn=self.include_read_turn,
            memory_directory=self.memory_directory,
        )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Tell the model the recall tool exists, unless `add_instructions` is false."""
        if not self.add_instructions:
            return None
        return _INSTRUCTIONS
