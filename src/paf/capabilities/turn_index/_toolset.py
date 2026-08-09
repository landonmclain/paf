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
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.conversation_search._source import HistorySource

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
Search the one-line turn summaries with simple substring matching.

This is the CHEAP recall step, between the injected turn index (rung 2) and the
wide `search_conversation_history` (last resort). Reach for it when you have NO
ref yet and don't know which turn holds the fact — the answer isn't in context
and no index line names the turn. It matches both the one-liner summary and its
`key_terms`. If a hit gives you a ref, call `read_turn(ref)` for the full turn;
don't jump straight to `search_conversation_history`.
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


def _render_message(message) -> str:
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

    if isinstance(message, ModelRequest):
        lines = []
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                content = part.content
                if not isinstance(content, str):
                    content = ' '.join(
                        c.content if hasattr(c, 'content') and isinstance(c.content, str) else ''
                        for c in content
                    ) if not isinstance(content, str) else content
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
    ) -> None:
        super().__init__(id=tool_id)
        self._source = source
        self._index_store = index_store
        self._finder = _TurnFinder(source)
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
        """Return index entries matching `query` (case-insensitive substring).

        Matches across BOTH the one-liner `summary` AND the structured
        `key_terms` -- so a turn whose sentence phrases things abstractly
        ("the database") is still found by its search vocabulary ("Postgres").
        """
        entries = self._index_store.read_all(conversation_id=ctx.conversation_id)
        q = query.lower()
        hits = [
            e
            for e in entries
            if q in e.summary.lower()
            or any(q in kt.lower() for kt in (e.key_terms or []))
        ]
        if not hits:
            return f'No index entries matching {query!r} ({len(entries)} indexed).'
        lines = [f'[{self._ref(e)}] {e.summary}' for e in hits[:20]]
        return 'Index matches:\n' + '\n'.join(lines)

    def _ref(self, entry: TurnIndexEntry) -> str:
        return entry.ref