"""Message rendering for the paf full-context search corpus.

Renders a persisted `ModelMessage` to searchable text with the SAME
injected-artifact strip convention (see
`paf.capabilities._markers`). This is the read-side hygiene that
`pydantic_ai_harness.conversation_search` lacks: its corpus renders marker-tagged
`UserPromptPart` blocks verbatim, so `<memory>` artifacts leak
into the BM25 index and can surface as matches. Owning the search here lets us
apply one consistent strip to the entire recall surface.

Mirrors `pydantic_ai_harness.conversation_search._toolset._format_message`, but:
- strips marker-tagged `TextContent` (via `_markers.injected_artifact_markers()`)
- keeps the same BM25-index rendering contract: `truncate=False` for the ranked
  corpus (terms past the display cutoff stay findable), `truncate=True` for the
  model-facing excerpt.
"""

from __future__ import annotations

import json

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from paf.capabilities._markers import injected_artifact_markers


def is_summary_artifact(message: ModelMessage) -> bool:
    """Return whether a message is a compaction summary artifact (never indexed).

    Byte-for-byte mirror of the harness `SnapshotHistorySource.is_summary_artifact`
    (see `_source.py` for rationale). Compaction summaries are derived transcripts,
    not original prompts, so they stay out of the corpus.
    """
    if not isinstance(message, ModelRequest):
        return False
    return any(isinstance(part, SystemPromptPart) and part.content.startswith(_SUMMARY_PREFIX) for part in message.parts)


# Literal mirror of `pydantic_ai_harness.compaction._summarizing_compaction._SUMMARY_PREFIX`.
_SUMMARY_PREFIX = "Summary of previous conversation:\n\n"


def _is_injected_text(text: TextContent) -> bool:
    """Return whether a `TextContent` is an injected artifact (not durable user content)."""
    metadata = getattr(text, "metadata", None)
    return metadata is not None and any(
        str(metadata).startswith(marker) for marker in injected_artifact_markers()
    )


def _user_prompt_text(part: UserPromptPart) -> str:
    """Render only the durable textual content of a user prompt, dropping injected blocks.

    `UserPromptPart.content` can be `str` or a list of content blocks (text, binary,
    image). Only `str` and `TextContent` carry searchable text. Marker-tagged
    `TextContent` (injected `<turn-index>` / `<memory>` blocks) are dropped.
    """
    if isinstance(part.content, str):
        return part.content
    texts: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent) and not _is_injected_text(item):
            texts.append(item.content)
    return " ".join(texts)


def format_message(message: ModelMessage, *, truncate: bool) -> str:
    """Render one message to text for the search corpus / excerpt.

    `truncate=False` keeps content full-length so the BM25 index matches terms
    past the display cutoff; `truncate=True` produces the shorter excerpt shown
    to the model.
    """
    lines: list[str] = []

    if isinstance(message, ModelRequest):
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                content = _user_prompt_text(part)
                if truncate and len(content) > 500:
                    content = content[:500] + "..."
                lines.append(f"User: {content}")
            elif isinstance(part, SystemPromptPart):
                content = part.content
                if content.startswith(_SUMMARY_PREFIX):
                    lines.append("[Compaction summary]")
                else:
                    if truncate:
                        content = content[:200]
                    lines.append(f"System: {content}")
            elif isinstance(part, ToolReturnPart):
                content = str(part.content)
                if truncate and len(content) > 500:
                    content = content[:500] + "..."
                lines.append(f"Tool [{part.tool_name}]: {content}")
            else:
                # RetryPromptPart / validation errors -- worth recalling, index in full.
                content = str(getattr(part, "content", part))
                if truncate and len(content) > 500:
                    content = content[:500] + "..."
                lines.append(f"Retry: {content}")
    else:
        for part in message.parts:
            if isinstance(part, TextPart):
                content = part.content
                if truncate and len(content) > 500:
                    content = content[:500] + "..."
                lines.append(f"Assistant: {content}")
            elif isinstance(part, ToolCallPart):
                args = json.dumps(part.args_as_dict(), ensure_ascii=False)
                if truncate and len(args) > 200:
                    args = args[:200] + "..."
                lines.append(f"Tool Call [{part.tool_name}]: {args}")

    return "\n".join(lines)


def format_messages(messages: list[ModelMessage], *, truncate: bool) -> list[str]:
    """Render messages to text lines, index-aligned with `messages`."""
    return [format_message(message, truncate=truncate) for message in messages]


def display_lines(messages: list[ModelMessage]) -> list[str]:
    """Render the numbered excerpt lines shown to the model.

    The `[index]` prefix lives only here, not in the indexed rendering: BM25
    would otherwise treat each ordinal as a rare, high-IDF token. A message with
    no renderable text gets a placeholder so excerpts don't show bare numbers.
    """
    lines = format_messages(messages, truncate=True)
    return [f"[{index}] {line or '[no text content]'}" for index, line in enumerate(lines)]
