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
    UserPromptPart,
)


def _is_user_turn(message: ModelMessage) -> bool:
    """A `ModelRequest` carrying a `UserPromptPart` anchors a user turn."""
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart) for part in message.parts
    )


def user_anchor_indices(messages: list[ModelMessage]) -> list[int]:
    """Indices (into `messages`) of every user-turn anchor, in order."""
    return [i for i, m in enumerate(messages) if _is_user_turn(m)]
