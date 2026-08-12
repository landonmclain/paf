"""The `read_turn` / `search_index` toolset for the Turn Summary Index capability.

`read_turn(ref)` is the "narrow, cheap, targeted read" of the escalation ladder:

    index read (in context, no tool call)
      -> read_turn (one round-trip, gated behind *detected* relevance)
        -> conversation_search (expensive wide fallback)

It does NOT store or copy full turns. It owns only the one-line index; it holds
the ref into pydantic-ai persistence and, on demand, resolves that ref through
the `HistorySource` seam back to the original messages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelRequest, TextContent, UserPromptPart
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.conversation_search._source import HistorySource

from ..search._ranking import Bm25Params, bm25_rank, recency_boost
from ._markers import injected_artifact_markers
from ._store import TurnIndexEntry, TurnIndexStore

READ_TURN_DESCRIPTION = """\
Read one specific turn from persisted conversation history, by its index ref.

The turn index keeps a one-line summary of every turn in context. When
you recognize from the index (or a `search_index` hit) that a turn is relevant,
call this with that run's (ref) to pull the FULL original turn -- the exact user prompt
and assistant responses / tool calls -- back into context.

Pass ref='{run_id}' to fetch one run's entire durable history, or
ref='{run_id}:{turn_id}' to fetch just the turn at that index (copy the exact
ref shown in the index). `turn_id` is the ordinal of the turn's user prompt
within the run (0, 1, ...), so refs are stable.

This is the targeted read step (rung 4): call it only once you HAVE a ref for the
turn you want -- from the injected index or a `search_index` hit. If you don't yet
have a ref and no index line names the turn, use `search_index` first. When one
turn is not enough and you need a wider picture, escalate to
`search_conversation_history` (last resort).
"""

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


@dataclass(frozen=True)
class _IndexLine:
    """An index entry + its resolved presence in the durable history."""

    entry: TurnIndexEntry
    ref: str
    found: bool
    user_prompt: str = ''


class _TurnFinder:
    """Resolves a (conversation, run, turn_id) ref into the durable turn.

    A user turn is anchored by the `ModelRequest` that carries a `UserPromptPart`
    in `HistorySource.run_history(run_id)` -- exactly the linking key captured in
    the design doc: one-liner -> (run_id, turn_id) -> walk `run_history(run_id)`
    to the turn_id-th user-prompt anchor. `turn_id` is the ORDINAL of the anchor
    among the run's user prompts, so it is STABLE for the life of the run (the
    same messages always resolve to the same ordinal) -- unlike a `message_index`,
    which rots as later messages are appended.
    """

    def __init__(self, source: HistorySource) -> None:
        self._source = source

    async def resolve(
        self,
        ref: str,
        *,
        run_id: str,
        conversation_id: str | None = None,
    ) -> str:
        history = await self._source.run_history(run_id=run_id)
        if not history:
            return f"No persisted history for run '{run_id}' (is it in this store?)."
        if ':' in ref:
            try:
                turn_id = int(ref.rsplit(':', 1)[1])
            except ValueError:
                return f"Malformed ref '{ref}'; expected '{run_id}' or '{run_id}:<turn_id>'."
            return self._render_turn(history, turn_id, run_id)
        return self._render_run(history, run_id)

    def _render_turn(self, history, turn_id: int, run_id: str) -> str:
        # Find the turn_id-th ModelRequest carrying a UserPromptPart (the anchor).
        anchors = [
            i for i, m in enumerate(history) if _is_user_turn(m)
        ]
        if turn_id < 0 or turn_id >= len(anchors):
            return (
                f"No turn at index {turn_id} in run '{run_id}' "
                f"({len(anchors)} user turns total)."
            )
        start = anchors[turn_id]
        # A turn spans from its anchor to the next user-turn anchor (or the end).
        end = anchors[turn_id + 1] if turn_id + 1 < len(anchors) else len(history)
        lines = []
        for i in range(start, end):
            lines.append(_render_message(history[i]))
        return '\n'.join(lines)

    def _render_run(self, history, run_id: str) -> str:
        lines = [f'--- run {run_id} ({len(history)} messages) ---']
        lines.extend(_render_message(m) for m in history)
        return '\n'.join(lines)

    def anchor_count(self, history) -> int:
        return sum(1 for m in history if _is_user_turn(m))


def _is_user_turn(message) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart) for part in message.parts
    )


def _is_injected_text(text) -> bool:
    """Return whether a `TextContent` is an injected artifact (not durable user content).

    Matches the metadata marker set shared with `_capability` (our turn-index)
    and the harness `Memory` capability. The marker is scope-qualified for the
    memory family (`...v1:<scope_hash>`), so we match on the stable base prefix.
    """
    return getattr(text, 'metadata', None) is not None and any(
        text.metadata.startswith(marker) for marker in injected_artifact_markers()
    )


def _render_message(message) -> str:
    from pydantic_ai.messages import (
        ModelResponse,
        TextPart,
        ToolCallPart,
    )

    if isinstance(message, ModelRequest):
        lines = []
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                content = part.content
                if not isinstance(content, str):
                    # Content is a list[str | TextContent]; drop injected artifacts
                    # (turn-index / memory blocks snapshotted into the User msg), then
                    # join the durable text. This is the read-side strip: a recalled
                    # turn shows the user's real prompt, NOT the transient injected index.
                    parts_out: list[str] = []
                    for item in content:
                        if isinstance(item, str):
                            # Plain string segments are durable user text.
                            parts_out.append(item)
                        elif isinstance(item, TextContent) and not _is_injected_text(item):
                            parts_out.append(item.content)
                    content = ' '.join(parts_out)
                    if not content:
                        # A UserPromptPart with only injected content carries no durable
                        # user text worth showing; skip it entirely.
                        continue
                lines.append(f'User: {content}')
            else:
                lines.append(f'[{type(part).__name__}]')
        return ' '.join(lines) if lines else f'<ModelRequest {len(message.parts)} parts>'
    if isinstance(message, ModelResponse):
        lines = []
        for part in message.parts:
            if isinstance(part, TextPart):
                lines.append(f'Assistant: {part.content}')
            elif isinstance(part, ToolCallPart):
                lines.append(f'Tool Call [{part.tool_name}]: {json.dumps(part.args_as_dict(), ensure_ascii=False)}')
        return '\n'.join(lines)
    return f'<{type(message).__name__}>'


class TurnIndexToolset(FunctionToolset[AgentDepsT]):
    """Includes `read_turn` (resolve a ref through pyai persistence) and `search_index` (BM25 over one-liners)."""

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
        self._finder = _TurnFinder(source)
        self._bm25_params = Bm25Params(k1=bm25_k1, b=bm25_b)
        self._recency_weight = recency_weight
        self._recency_half_life_days = recency_half_life_days
        self._max_matches = max_matches
        self.add_function(self.read_turn, name='read_turn', description=READ_TURN_DESCRIPTION)
        self.add_function(self.search_index, name='search_index', description=SEARCH_INDEX_DESCRIPTION)

    async def read_turn(
        self,
        ctx: RunContext[AgentDepsT],
        ref: str,
    ) -> str:
        """Resolve one turn's index ref into its full durable turn via pydantic-ai persistence."""
        run_id, _, _ = ref.partition(':')
        conversation_id = ctx.conversation_id
        return await self._finder.resolve(ref, run_id=run_id, conversation_id=conversation_id)

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
            lines.append(f'[{entry.ref}] (score {score:.2f}) {entry.summary}')
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
