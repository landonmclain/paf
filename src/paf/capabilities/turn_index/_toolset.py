"""The `read_turn` / `search_index` toolset for the Turn Summary Index capability.

`read_turn(ref)` is the "narrow, cheap, targeted read" of the escalation ladder:

    index read (in context, no tool call)
      -> read_turn (one round-trip, gated behind *detected* relevance)
        -> conversation_search (expensive wide fallback)

`search_index` is the index's own probe (BM25 over the one-line summaries) —
the index's real job. Full-content reads (`read_turn`) belong to `search` and are
ALWAYS composed alongside, so this toolset registers only `search_index`.

It does NOT store or copy full turns. It owns only the one-line index; it holds
the ref into pydantic-ai persistence. Resolving that ref is `search`'s job.
"""

from __future__ import annotations

from pydantic_ai import RunContext
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.conversation_search._source import HistorySource

from ..search._ranking import Bm25Params, bm25_rank, recency_boost
from ._store import TurnIndexEntry, TurnIndexStore

__all__ = ["SEARCH_INDEX_DESCRIPTION", "TurnIndexToolset"]

SEARCH_INDEX_DESCRIPTION = """\
Search this conversation's one-line turn summaries, ranked by BM25 relevance.

Reach for it when you have NO ref yet and don't know which turn holds a fact:
a cheap, scoped probe between the injected index (rung 2) and the wide
`search_conversation_history` (last resort). It ranks every indexed turn of the
CURRENT conversation across the one-liner `summary` (anchored, weighted highest),
its `key_terms` (entity/vocabulary), and its `outcome` (resolved/blocked/...).

Querying: best as 1-3 concrete keywords -- real names, APIs, topics, error codes --
not full sentences. Multi-word queries score each word independently, so a phrase
with one distinctive term still matches. Rare terms outweigh common ones (IDF).

Result: the top few matches are returned rank-ordered with a relevance `score`
and each turn's `ref` -- call `read_turn(ref)` for the full turn. Do NOT jump
straight to `search_conversation_history` if a ref names your turn.

Ranking note: ties break toward the more recent turn (a light recency nudge), so
between two equally-relevant matches the fresher one ranks first -- but a genuine
relevance edge still wins. When facts have since been superseded (e.g. a move or
a changed value), prefer the more recent match; check `read_turn` if in doubt.

No match: if the index has no keyword coverage for your query, the tool says so
explicitly and points you to `search_conversation_history` (which searches the
full original messages, not just one-liners) -- escalate there when the index
cannot help.
"""


class TurnIndexToolset(FunctionToolset[AgentDepsT]):
    """`search_index` (BM25 over one-liners) — the index's own recognition layer.

    Full-content reads (`read_turn`) are owned by `search` and always present
    alongside, so this toolset does NOT register it (avoiding a toolname collision).
    """

    def __init__(
        self,
        source: HistorySource,
        index_store: TurnIndexStore,
        *,
        tool_id: str = 'turn-index',
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        recency_weight: float = 0.1,
        recency_half_life_days: float = 30.0,
        max_matches: int = 5,
    ) -> None:
        super().__init__(id=tool_id)
        if max_matches < 1:
            raise ValueError(f'max_matches must be >= 1, got {max_matches!r}.')
        if recency_weight < 0:
            raise ValueError(f'recency_weight must be >= 0, got {recency_weight!r}.')
        if recency_half_life_days <= 0:
            raise ValueError(
                f'recency_half_life_days must be > 0, got {recency_half_life_days!r}.'
            )
        self._source = source
        self._index_store = index_store
        self._bm25_params = Bm25Params(k1=bm25_k1, b=bm25_b)
        self._recency_weight = recency_weight
        self._recency_half_life_days = recency_half_life_days
        self._max_matches = max_matches
        # `read_turn` is owned by `search` (ConversationSearchToolset) and is
        # ALWAYS present when this index toolset is composed — do NOT register it
        # here or the combined toolset collides on the name and hard-errors.
        # This toolset is the index's own probe layer only: `search_index`.
        self.add_function(self.search_index, name="search_index", description=SEARCH_INDEX_DESCRIPTION)

    async def search_index(
        self,
        ctx: RunContext[AgentDepsT],
        query: str,
    ) -> str:
        """Rank this conversation's index entries by BM25 relevance to `query`.

        Each indexed turn is scored across its full record -- the one-liner
        `summary` (anchored: the running text is weighted by being present twice,
        so a summary term outranks a bare `key_terms`/`outcome` term), the
        `key_terms` vocabulary, and `outcome`. Returns the top `max_matches`,
        rank-ordered with a relevance score and each turn's `ref`. When the index
        has no keyword coverage for the query, it says so and points to the wide
        `search_conversation_history` fallback.

        Matches are additionally nudged by recency: a newer turn edges out a
        relevance-tied older one (weighted by `recency_weight`, decayed over
        `recency_half_life_days`). This is a conservative tie-breaker -- it never
        overrides a meaningful relevance gap, only breaks near-equal BM25 scores
        in favor of the more recent turn (helps the index-not-losing-updates
        concern raised during the memory/ablation work).
        """
        if ctx.conversation_id is None:
            # Fail closed: an unscoped run would otherwise search EVERY conversation's
            # index in the shared store (same exposure the wide tool guards against).
            return _NO_CONVERSATION_MSG

        entries = self._index_store.read_all(conversation_id=ctx.conversation_id)
        if not entries:
            return f'No turn index entries for this conversation ({query!r}).'

        scored = bm25_rank(
            query,
            [_searchable_text(e) for e in entries],
            self._bm25_params,
        )
        if not scored:
            return (
                f'No turn index matches {query!r} across {len(entries)} indexed turns. '
                "The index has no keyword coverage here -- escalate to "
                '`search_conversation_history` (it searches the full original '
                'messages, not just the one-liners).'
            )

        # Fold recency in as a conservative nudge. Anchor to the newest entry so
        # ranking stays deterministic (no wall-clock): boost ∈ [0, 1], and the
        # default `recency_weight` (0.1) keeps it well below a single BM25 IDF
        # unit, so it only breaks genuine near-ties and never overrides a real
        # relevance gap.
        anchor_ts = max((e.ts for e in entries), default='')
        blended = []
        for idx, score in scored:
            entry = entries[idx]
            recency = (
                self._recency_weight
                * recency_boost(entry.ts, anchor_ts, self._recency_half_life_days)
                if self._recency_weight
                else 0.0
            )
            blended.append((idx, score + recency))
        blended.sort(key=lambda item: item[1], reverse=True)

        lines: list[str] = []
        for idx, score in blended[: self._max_matches]:
            entry = entries[idx]
            sid = f' session={entry.session_id}' if entry.session_id else ''
            lines.append(f'[{entry.ref}] (score {score:.2f}){sid} {entry.summary}')
        return 'Index matches (BM25-ranked):\n' + '\n'.join(lines)

    def _ref(self, entry: TurnIndexEntry) -> str:
        return entry.ref


def _searchable_text(entry: TurnIndexEntry) -> str:
    """Flatten one index entry into the text BM25 ranks against.

    The `summary` is deliberately included TWICE so its terms carry higher
    term-frequency than `key_terms` / `outcome`: this keeps `search_index` a
    *summary-anchored* probe (a turn whose one-liner uses a query term outranks
    one that only matches an incidental keyword or status), while still letting
    keyword / outcome matches surface turns the sentence phrases abstractly.
    """
    parts = [entry.summary, entry.summary]
    if entry.key_terms:
        parts.append(' '.join(entry.key_terms))
    if entry.outcome:
        parts.append(entry.outcome)
    return ' '.join(parts)


_NO_CONVERSATION_MSG = (
    "Search-index needs a conversation id, and this run has none. "
    "Pass `conversation_id=` to `Agent.run(...)` so the tool can tell this "
    "conversation's index from other conversations in the shared store."
)