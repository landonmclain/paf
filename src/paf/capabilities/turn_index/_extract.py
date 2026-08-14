"""Extract a bounded, flat-text rendition of a full run for the LLM summarizer.

The turn summarizer must see the *whole* run -- the user's intent, every tool
call (with arguments), the results, and the final answer -- so its one-liner
captures what actually happened, not just which tools were touched.

Per the design decision (Landon 2026-08-08):

  * send the whole run
  * the ONLY things we truncate are tool-call **arguments** and tool
    **results** (the payloads that can legitimately be huge)
  * never truncate the user prompt or the assistant's final text (intent and
    outcome)

This module is pure and DB-free: it formats an already-recovered
``list[ModelMessage]`` (from ``SnapshotHistorySource.run_history``) into a
single flat string. It MUST NOT lose data permanently -- truncation is only
on what is *shown to the summarizer model*; the full turn stays in the store
and ``read_turn`` returns it whole later.
"""

from __future__ import annotations

from typing import Iterable

from .._markers import injected_artifact_markers

# NOTE: part types are imported lazily inside the narrowers to match the rest
# of this package (see _toolset._render_message) -- pydantic-ai exposes no
# public `Part` union, so isinstance-narrowing on the concrete classes is the
# pattern that keeps the typing sane without a stubs dance.

# Per-part caps (chars). These bound only the two payload kinds that can be
# large; the default for everything else is "whole". Tunable -- start generous.
ARG_CAP = 500      # ToolCallPart.args
RESULT_CAP = 2000  # ToolReturnPart.content

_TRUNC_MARKER = " …[truncated]"


def truncate(text: object, limit: int | None) -> str:
    """Render `text` as a string, cut to `limit` chars (None = whole).

    A trimmed string gets a ``...\\u00a0[truncated]`` suffix so the summarizer
    knows content was elided (and a review can tell a cap from silence).
    """
    s = str(text)
    if limit is None or len(s) <= limit:
        return s
    return s[: max(0, limit - len(_TRUNC_MARKER))] + _TRUNC_MARKER


def _render_user_content(content: object) -> str:
    """UserPromptPart.content is `str | list[Content]`; flatten to a single str, uncapped.

    Drops injected artifacts (`<turn-index>` / harness `<memory>` blocks, which
    carry a metadata marker) so a turn's summarization reflects the user's real
    prompt, not the transient context injection StepPersistence snapshotted.
    """
    if isinstance(content, str):
        return content
    # A list of Content blocks (text/images); render text parts, skip injected
    # artifacts and binary.
    parts: list[str] = []
    for c in content:
        if isinstance(c, str):
            parts.append(c)
        elif hasattr(c, "content") and isinstance(c.content, str):
            metadata = getattr(c, "metadata", None)
            if isinstance(metadata, str) and any(
                metadata.startswith(m) for m in injected_artifact_markers()
            ):
                continue  # injected block -- not durable user content
            parts.append(c.content)
    return " ".join(p for p in parts if p) or str(content)


def _format_part(part, out: list[str]) -> None:
    """Append one message part's flat rendition to `out`."""
    from pydantic_ai.messages import (
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    if isinstance(part, UserPromptPart):
        # Intent -- never truncated.
        out.append(f"user: {_render_user_content(part.content)}")
    elif isinstance(part, ToolCallPart):
        out.append(f"tool call: {part.tool_name}({truncate(part.args, ARG_CAP)})")
    elif isinstance(part, ToolReturnPart):
        out.append(f"tool result ({part.tool_name}): {truncate(part.content, RESULT_CAP)}")
    elif isinstance(part, TextPart):
        # Assistant final / reasoning text -- never truncated.
        out.append(f"assistant: {part.content}")
    else:
        # Unknown part type -- render defensively, keep it, never drop.
        out.append(f"part<{type(part).__name__}>: {truncate(part, RESULT_CAP)}")


def build_turn_context(messages: Iterable[object]) -> str:
    """Render a recovered run's messages into one flat, bounded text block.

    `messages` is the full ``run_history(run_id)`` list. Structure (the
    ``ModelRequest`` / ``ModelResponse`` wrappers and part kinds) is kept so
    the summarizer can follow the turn flow; only verbose payload parts are
    capped. Unknown message/part kinds are never dropped.
    """
    from pydantic_ai.messages import ModelRequest, ModelResponse

    blocks: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                _format_part(part, blocks)
        elif isinstance(message, ModelResponse):
            for part in message.parts:
                _format_part(part, blocks)
        else:
            # Defensive: never silently drop an unhandled message kind.
            blocks.append(truncate(message, RESULT_CAP))
    return "\n".join(blocks)