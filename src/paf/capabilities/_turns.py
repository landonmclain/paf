"""Shared message-turn structure helpers.

Small pure helpers over a pydantic-ai ``list[ModelMessage]`` that several
capabilities need (recall/consolidation) but that do not belong to any single
capability's internals. Lives at the shared ``capabilities`` package level (like
``_markers``) so ``search`` and ``memory`` can both use it without depending on
one another or on the archived ``turn_index`` capability.

These came out of the retired ``turn_index`` experiment: ``user_anchor_indices``
was the genuinely reusable idea (turn-aware pointer addressing), kept here while
the turn-index feature itself was archived.
"""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    TextContent,
    UserPromptPart,
)

from paf.capabilities._markers import injected_artifact_markers


def _user_prompt_text(part: UserPromptPart) -> str | None:
    """Durable user-prompt text, dropping injected artifacts (e.g. ``<memory>``).

    The thread summarizer (and any downstream reader) only wants the *user's
    real words*, not the transient blocks paf injects each request (memory
    notebook, turn-index, etc.). Injected `TextContent` is tagged with a marker
    (see `paf.capabilities._markers`) and dropped; a plain ``str`` content is
    kept verbatim; a list keeps only untagged text blocks.
    """
    if isinstance(part.content, str):
        return part.content
    texts: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            meta = getattr(item, "metadata", None)
            if meta is None or not any(
                str(meta).startswith(marker) for marker in injected_artifact_markers()
            ):
                texts.append(item.content)
    return " ".join(texts) or None


def _is_user_turn(message: ModelMessage) -> bool:
    """A `ModelRequest` carrying a *real* (non-injected) user prompt anchors a turn.

    Uses the shared marker-stripping rule (`_user_prompt_text`): injected
    artifacts (``<memory>``, ``<turn-index>``, etc.) are tagged `TextContent`
    and stripped — if nothing durable remains, the request is not a user turn.
    This is the "strip markers after pulling from persistence" SOP applied at
    every read, so memory consolidation, the thread summarizer, window
    eviction, and search all count the same real turns.
    """
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart) and _user_prompt_text(part) is not None
        for part in (getattr(message, "parts", None) or [])
    )


def user_anchor_indices(messages: list[ModelMessage]) -> list[int]:
    """Indices (into `messages`) of every user-turn anchor, in order."""
    return [i for i, m in enumerate(messages) if _is_user_turn(m)]


def _real_user_prompt_text(message: ModelMessage) -> str | None:
    """The real (non-injected) user-prompt text of `message`, if any.

    Returns the first ``UserPromptPart`` whose marker-stripped text (`_user_prompt_text`)
    is non-empty. Injected marker blocks are stripped, so this is the user's
    actual words — the same content the thread summarizer and memory
    consolidation persist/consume.
    """
    for part in getattr(message, "parts", None) or []:
        if isinstance(part, UserPromptPart):
            text = _user_prompt_text(part)
            if text is not None:
                return text
    return None


def distinct_user_turn_texts(messages: list[ModelMessage]) -> list[str]:
    """Real user-prompt texts with duplicates collapsed (in first-seen order).

    Dedups by the full real prompt text, so the same user turn re-persisted at
    several addresses (e.g. once clean and once with an injected block attached
    to a *different* request) collapses to a single entry. This is the honest
    measure of "distinct user turns" — use it for any read-path turn count
    (the summarizer's minimum-turn gate) where raw ``user_anchor_indices`` is
    inflated by re-persisted copies.
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in messages:
        text = _real_user_prompt_text(m)
        if text is None or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
