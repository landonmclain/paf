"""The shared `read_turn` primitive — full-content read of a persisted turn.

Owned by `search` because search is the permanent retrieval layer. A
`run_id:turn_id` ref (or bare `run_id`) is resolved through the `HistorySource`
seam back to the original `ModelMessage` span — whether the ref came from a
recalled one-liner or from raw search (`search_conversation_history`).

The read primitive is deliberately index-free: it depends only on
`HistorySource` and message rendering. `ConversationSearchToolset` (raw search
ladder) registers `read_turn` from here, so the model has ONE full-content read
tool regardless of how it obtained the ref.
"""

from __future__ import annotations

import json

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.conversation_search._source import HistorySource

from paf.capabilities._turns import _is_user_turn as _turns_is_user_turn

from .._markers import injected_artifact_markers

# Char caps for the bounded (non-`full`) read. Mirrors `search/_render.py` so the
# model-facing read and the search excerpt stay on the same budget contract.
_TOOL_RETURN_TRUNCATE = 500
_TOOL_CALL_TRUNCATE = 200
_TEXT_TRUNCATE = 500

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

Tool calls and results are returned truncated by default (tool results ~500 chars,
tool call args ~200 chars, message text ~500 chars -- same budget as search
excerpts) so a read stays bounded. To pull a byte-complete turn -- including the
full untruncated tool results -- pass full=True. Prefer full=True only when the
truncated read shows the content actually matters and you need the complete value.

This is the targeted read step (rung 4): call it only once you HAVE a ref for the
turn you want -- from the injected index or a `search_index` hit. If you don't yet
have a ref and no index line names the turn, use `search_index` first. When one
turn is not enough and you need a wider picture, escalate to
`search_conversation_history` (last resort).
"""


def _is_user_turn(message: ModelMessage) -> bool:
    """A *real* (non-injected) `UserPromptPart` anchors a user turn.

    Delegates to the shared ``_turns._is_user_turn`` (strict: injected
    ``<memory>``/index blocks with non-str content are not user turns).
    """
    return _turns_is_user_turn(message)


def _source_session_id(message: ModelMessage) -> str | None:
    """The source-session id a persisted message carries, if any.

    Read from the generic `metadata['source_session_id']` provenance key (set
    at persist time). Surfacing it on `read_turn` output lets a retrieval be
    scored by LongMemEval ID-membership (did the surfaced evidence's session id
    appear in `answer_session_ids`) -- the same convention
    `search_conversation_history` uses. Returns None when absent so the turn
    renders unchanged for untagged pipelines.
    """
    md = getattr(message, "metadata", None)
    if isinstance(md, dict):
        sid = md.get("source_session_id")
        if isinstance(sid, str) and sid:
            return sid
    return None


def _is_injected_text(text: TextContent) -> bool:
    """Return whether a `TextContent` is an injected artifact (not durable user content).

    Matches the metadata marker set shared with the other injected-artifact
    producers. The marker is scope-qualified for the memory family
    (`...v1:<scope_hash>`), so we match on the stable base prefix.
    """
    return getattr(text, "metadata", None) is not None and any(
        text.metadata.startswith(marker) for marker in injected_artifact_markers()
    )


def _truncate(text: str, cap: int) -> str:
    return text if len(text) <= cap else text[:cap] + "..."


def _render_message(message: ModelMessage, *, full: bool = False) -> str:
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
                        elif isinstance(item, TextContent) and not _is_injected_text(
                            item
                        ):
                            parts_out.append(item.content)
                    content = " ".join(parts_out)
                    if not content:
                        # A UserPromptPart with only injected content carries no durable
                        # user text worth showing; skip it entirely.
                        continue
                if not full:
                    content = _truncate(content, _TEXT_TRUNCATE)
                lines.append(f"User: {content}")
            elif isinstance(part, ToolReturnPart):
                content = str(part.content)
                if not full:
                    content = _truncate(content, _TOOL_RETURN_TRUNCATE)
                lines.append(f"Tool [{part.tool_name}]: {content}")
            else:
                lines.append(f"[{type(part).__name__}]")
        return (
            " ".join(lines) if lines else f"<ModelRequest {len(message.parts)} parts>"
        )
    if isinstance(message, ModelResponse):
        lines = []
        for part in message.parts:
            if isinstance(part, TextPart):
                content = part.content
                if not full:
                    content = _truncate(content, _TEXT_TRUNCATE)
                lines.append(f"Assistant: {content}")
            elif isinstance(part, ToolCallPart):
                args = json.dumps(part.args_as_dict(), ensure_ascii=False)
                if not full:
                    args = _truncate(args, _TOOL_CALL_TRUNCATE)
                lines.append(f"Tool Call [{part.tool_name}]: {args}")
        return "\n".join(lines)
    return f"<{type(message).__name__}>"


class TurnFinder:
    """Resolves a (conversation, run, turn_id) ref into the durable turn.

    A user turn is anchored by the `ModelRequest` that carries a `UserPromptPart`
    in `HistorySource.run_history(run_id)`. `turn_id` is the ORDINAL of the anchor
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
        full: bool = False,
    ) -> str:
        history = await self._source.run_history(run_id=run_id)
        if not history:
            return f"No persisted history for run '{run_id}' (is it in this store?)."
        if ":" in ref:
            try:
                turn_id = int(ref.rsplit(":", 1)[1])
            except ValueError:
                return f"Malformed ref '{ref}'; expected '{run_id}' or '{run_id}:<turn_id>'."
            return self._render_turn(history, turn_id, run_id, full=full)
        return self._render_run(history, run_id, full=full)

    def _render_turn(
        self, history, turn_id: int, run_id: str, *, full: bool = False
    ) -> str:
        # Find the turn_id-th ModelRequest carrying a UserPromptPart (the anchor).
        anchors = [i for i, m in enumerate(history) if _is_user_turn(m)]
        if turn_id < 0 or turn_id >= len(anchors):
            return (
                f"No turn at index {turn_id} in run '{run_id}' "
                f"({len(anchors)} user turns total)."
            )
        start = anchors[turn_id]
        # A turn spans from its anchor to the next user-turn anchor (or the end).
        end = anchors[turn_id + 1] if turn_id + 1 < len(anchors) else len(history)
        lines = []
        if start < len(history):
            sid = _source_session_id(history[start])
            if sid:
                lines.append(f"-- session: {sid} --")
        for i in range(start, end):
            lines.append(_render_message(history[i], full=full))
        return "\n".join(lines)

    def _render_run(self, history, run_id: str, *, full: bool = False) -> str:
        lines = [f"--- run {run_id} ({len(history)} messages) ---"]
        lines.extend(_render_message(m, full=full) for m in history)
        return "\n".join(lines)

    def anchor_count(self, history) -> int:
        return sum(1 for m in history if _is_user_turn(m))


class ReadTurnToolset(FunctionToolset[AgentDepsT]):
    """Exposes the shared `read_turn` full-content read tool.

    Tool-id defaults to `"reader"` so a capability mounting it can register it
    under its own id without colliding. It owns NO
    state of its own -- just a `TurnFinder` over a `HistorySource`.

    `ConversationSearchToolset` exposes this same tool
    so the model has a single full-content read regardless of retrieval ladder.
    """

    def __init__(
        self,
        source: HistorySource,
        *,
        tool_id: str = "reader",
    ) -> None:
        super().__init__(id=tool_id)
        self._finder = TurnFinder(source)
        self.add_function(
            self.read_turn, name="read_turn", description=READ_TURN_DESCRIPTION
        )

    async def read_turn(
        self,
        ctx: RunContext[AgentDepsT],
        ref: str,
        full: bool = False,
    ) -> str:
        """Resolve one ref into its full durable turn via pydantic-ai persistence.

        `full=False` renders tool calls/results and message text truncated at the
        same budget as search excerpts (bounded read). `full=True` renders the
        byte-complete turn, including untruncated tool returns.
        """
        run_id, _, _ = ref.partition(":")
        return await self._finder.resolve(
            ref, run_id=run_id, conversation_id=ctx.conversation_id, full=full
        )
