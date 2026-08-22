"""Per-turn slicing + summarizer for the turn index.

This is M2's first feature's write path, made per-turn (2026-08-08). The
one-line-per-turn index is produced from a *single completed turn* — not the
whole run — the moment that turn finishes, on a synchronous hook, structured so
the exact same function can move to a background worker later without redesign.

**Trigger / cadence.** `TurnIndexCapability.after_model_request` fires this for
every completed user turn (a turn = user prompt + all tool traffic + the final
assistant response). It is intentionally *not* run-based: we index at the
earliest point the turn's content is complete and stable, and we index *each
turn on its own*, so nothing waits for the run to end and coverage holds for
every turn that later falls out of live context.

**Background-readiness.** `summarize_turn(turn, ...)` and the renderers in
`_extract` are pure over a message slice. To move summarization off the hot
path later, call this from a worker task / enqueue the rendered turn; the
function, the store append, and the ref model are unchanged. For now it runs
synchronously in the hook for testability.

**Ref model — stable turn ids.** The durable ref for a turn is
`(run_id, turn_id)` where `turn_id` is the ORDINAL of the turn's user-prompt
anchor within the run (`0`, `1`, ...). It is stable (same messages → same
ordinals) and directly sliceable: `read_turn('{run_id}:{turn_id}')` walks
`run_history(run_id)` to the turn_id-th user prompt. This replaces the old
`(run_id, message_index)` ref whose `message_index` rotted as the run grew.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    UserPromptPart,
)

from ._extract import build_turn_context
from ._store import TurnIndexStore
from ._summarizer import TurnSummary

_TURN_ID_MISSING = -1


def _is_user_turn(message: ModelMessage) -> bool:
    """A `ModelRequest` carrying a `UserPromptPart` anchors a user turn."""
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart) for part in message.parts
    )


def user_anchor_indices(messages: list[ModelMessage]) -> list[int]:
    """Indices (into `messages`) of every user-turn anchor, in order."""
    return [i for i, m in enumerate(messages) if _is_user_turn(m)]


def this_turn_slice(messages: list[ModelMessage]) -> list[ModelMessage]:
    """The last completed turn: from its user anchor through the final response.

    `messages` is the live history at `after_model_request`, which already
    includes the just-completed `ModelResponse`. The most recent user anchor
    starts "this turn"; everything after it (its tool traffic + final answer)
    is the turn. Returns an empty list when there is no user anchor yet.
    """
    anchors = user_anchor_indices(messages)
    if not anchors:
        return []
    start = anchors[-1]
    return messages[start:]


def this_turn_id(messages: list[ModelMessage]) -> int:
    """Ordinal of the last user anchor among all anchors in `messages`.

    Stable for the life of the run: the same messages always resolve to the
    same ordinal, and `read_turn` re-finds it via the same walk.
    """
    anchors = user_anchor_indices(messages)
    if not anchors:
        return _TURN_ID_MISSING
    return len(anchors) - 1


@dataclass(frozen=True)
class Turn:
    """A single completed user turn ready for summarization."""

    run_id: str
    conversation_id: str | None
    turn_id: int
    messages: list[ModelMessage]

    @property
    def ref(self) -> str:
        return f'{self.run_id}:{self.turn_id}'

    @property
    def has_content(self) -> bool:
        return len(self.messages) > 0


def summarize_turn(turn: Turn, *, index_store: TurnIndexStore) -> str:
    """Produce the one-line summary for a turn from its message slice.

    `turn.messages` is the single-turn slice (user prompt, tool traffic, final
    answer) rendered by `build_turn_context`. This is the single seam where the
    real LLM summarizer drops in. Today it is a deterministic stub (no model
    call) so the pipeline is testable end-to-end without a model or API key;
    the shape (pure, returns one bounded `str`, keyed by `Turn`) is exactly
    what the LLM version returns.

    Idempotency: `index_store.has(...)` gates the append, so re-running a turn
    (retry, resume, hook re-fire) never duplicates an existing index line.
    """
    if not turn.has_content:
        return f'turn {turn.turn_id} in run {turn.run_id} (no message history)'
    context = build_turn_context(turn.messages)
    lines = [ln for ln in context.splitlines() if ln.strip()]
    if not lines:
        return f'turn {turn.turn_id} in run {turn.run_id} (empty)'
    digest = hashlib.sha256(context.encode()).hexdigest()[:8]
    tools = _tools_in_context(lines)
    tail = 'tools: ' + (', '.join(tools) if tools else 'none') + f' [{digest}]'
    return f'{lines[0][:120]} → {tail}'


async def summarize_turn_append(
    *,
    turn: Turn,
    index_store: TurnIndexStore,
    summary: TurnSummary | None = None,
    session_id: str | None = None,
) -> tuple[bool, str]:
    """Summarize `turn` and append to the index iff it is not already indexed.

    `summary` is the WRITE seam. When provided it is a structured `TurnSummary`
    from the real LLM summarizer -- its `sentence` becomes the one-liner and its
    `key_terms` / `outcome` are persisted alongside it for `search_index`. When
    omitted (`None`, default) the deterministic `summarize_turn` stub one-liner
    is used -- the pipeline still works without a model or API key.

    Returns `(appended, one-liner)`. Idempotent per turn: an already-indexed
    turn is a no-op. This is the function the capability hook calls directly;
    moving it to a background worker changes only WHERE it is called, not what
    it does.
    """
    if summary is None:
        one_line = summarize_turn(turn, index_store=index_store)
        key_terms: list[str] | None = None
        outcome: str | None = None
    else:
        one_line = summary.sentence
        key_terms = summary.key_terms or None
        outcome = summary.outcome
    if index_store.has(turn.conversation_id, turn.run_id, turn.turn_id):
        return False, one_line
    index_store.append(
        conversation_id=turn.conversation_id,
        run_id=turn.run_id,
        turn_id=turn.turn_id,
        summary=one_line,
        key_terms=key_terms,
        outcome=outcome,
        session_id=session_id,
    )
    return True, one_line


def _tools_in_context(lines: list[str]) -> list[str]:
    """Collect distinct tool names from `tool call:` / `tool result (` lines."""
    tools: set[str] = set()
    for ln in lines:
        low = ln.lower()
        if low.startswith('tool call: '):
            name = ln.split('tool call: ', 1)[1].split('(', 1)[0].strip()
            if name:
                tools.add(name)
        elif low.startswith('tool result ('):
            inner = ln.split('tool result (', 1)[1]
            name = inner.split(')', 1)[0].strip()
            if name:
                tools.add(name)
    return sorted(tools)
