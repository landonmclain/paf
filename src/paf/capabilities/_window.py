"""Turn-bounded sliding-window compaction — never evicts the active turn.

:class:`TurnBoundSlidingWindow` — the assistant's eviction tier. Trim-only
and turn-bounded: it drops the oldest *settled* messages (strictly before the
most recent real user prompt) and does **no memory work**. This is the only
window ``build_compaction`` assembles; there is no ``consolidate`` toggle.

``compact`` slices at the most recent real user prompt, so the window can never
consume or degrade the active turn.

Why ``compact`` and not ``before_model_request``
------------------------------------------------
``TieredCompaction._escalate`` drives tiers directly through their ``compact``
method and explicitly *bypasses* each tier's own trigger (see the harness
class docstring: "Each tier's own trigger is bypassed -- TieredCompaction
drives the tiers directly via their ``compact`` method"). So overriding
``before_model_request`` on a tier would be silently dead when the window sits
inside a ``TieredCompaction`` ladder. Overriding ``compact`` is the seam that
actually fires — ``_escalate`` calls ``tier.compact(messages, ctx)`` exactly
when it intends to trim, so the gate is bound to the real eviction event.
``find_safe_cutoff(...) <= 0`` (nothing about to be evicted) short-circuits to
a no-op, so a window costs nothing on turns where it would not trim.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai_harness.compaction import SlidingWindowCompaction
from pydantic_ai_harness.compaction._shared import find_safe_cutoff

from paf.capabilities._turns import (
    _user_prompt_text,
    _is_user_turn as _turns_is_user_turn,
)
from paf.capabilities._turns import user_anchor_indices

if TYPE_CHECKING:
    from pydantic_ai import RunContext


# Mid-turn eviction gate (Landon 2026-09-02, Option A): the eviction tier must
# never consume the ACTIVE user turn. `compact()` is driven by
# TieredCompaction._escalate before every model request, so it fires mid-turn —
# exactly when the active prompt is vulnerable to find_safe_cutoff's back-walk
# (run 01a05ff8: the live "implement it" prompt was evicted while its response
# survived). The gate bounds every eviction decision to the completed prefix
# strictly BEFORE the most recent real user prompt, so the current turn is
# structurally unreachable by the window tier.
#
# The "<memory>" injected context parts carry a non-str `content` (a list), so
# they are deliberately NOT turn starts — this mirrors runtime2._is_turn_start
# so the gate and the resume_window renderer see the same turns.
def _is_real_user_prompt(message: ModelMessage) -> bool:
    """Whether `message` is a real (non-injected) user-turn anchor.

    Delegates to the shared ``_turns._is_user_turn`` so the "strip markers after
    pulling from persistence" rule (an injected ``<memory>``/index block with
    non-str content is NOT a user turn) lives in exactly one place.
    """
    return _turns_is_user_turn(message)


def _first_text(message: ModelMessage) -> str | None:
    """Curated text payload of an evicted message, or ``None``.

    The thread-summarizer WAL should capture **only** the durable signal per
    message — the user's prompt and the model's *final* answer — never the
    model's mid-step reasoning narration (``ThinkingPart``), interim tool
    chatter, or transient injected blocks:

    - ``ModelRequest`` → the user prompt text, injected blocks stripped.
    - ``ModelResponse`` → the LAST ``TextPart`` (the model's final answer),
      skipping ``ThinkingPart`` / ``ToolCallPart`` entirely.
    - anything else → ``None`` (no durable text worth summarizing).
    """
    if isinstance(message, ModelResponse):
        # Only a TURN-CLOSING response (one that does NOT call a tool) is the
        # model's final answer. Intermediate responses that issue a tool call
        # carry mid-step narration ("Let me..."), not durable content, so they
        # are dropped wholesale. Return the last real TextPart of the closing
        # response; `ThinkingPart` / `ToolCallPart` are never durable text.
        if any(isinstance(part, ToolCallPart) for part in message.parts):
            return None
        last: str | None = None
        for part in message.parts:
            if isinstance(part, TextPart):
                content = getattr(part, "content", None)
                if isinstance(content, str) and content:
                    last = content
        return last
    if isinstance(message, ModelRequest):
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                return _user_prompt_text(part)
        return None
    return None


class TurnBoundSlidingWindow(SlidingWindowCompaction):
    """Turn-bounded, trim-only sliding window — never evicts the active turn.

    The assistant's eviction tier. Same trigger/behaviour as
    ``SlidingWindowCompaction``; additionally it bounds every eviction decision
    to the settled prefix strictly before the most recent real user prompt,
    re-appending the active turn verbatim.

    It performs **no memory work** — durable-fact capture is owned solely by the
    background watermark worker (``MemoryConsolidationQueue`` in
    :mod:`paf.capabilities.memory._background`), never by eviction. There is no
    ``consolidate`` flag; this window does not know memory exists.

    When ``keep_messages`` protects fewer messages than the tool-pair walk's
    healthy minimum, ``find_safe_cutoff`` may still back-walk *within the settled
    prefix* — but it can never reach the active turn, because that turn is sliced
    out and re-appended by this gate.

    Optional ``capture_hook``: a callable invoked after an eviction as
    ``capture_hook(evicted_rows, keep_messages)`` where ``evicted_rows`` are the
    dropped messages as plain ``{"ref", "role", "text"}`` dicts (oldest first)
    and ``keep_messages`` is this window's retain count. This is a generic
    eviction *report* seam — the window stays decoupled from any consumer (it
    emits provenance data, it never imports memory or the thread summarizer).
    When ``None`` (default) the window does no extra work.
    """

    def __init__(
        self,
        *args: object,
        capture_hook: Callable[[list[dict], int], None] | None = None,
        **kwargs: object,
    ) -> None:
        """Pass through to ``SlidingWindowCompaction`` + an optional capture hook.

        A keyword-only ``capture_hook`` is folded in after the base dataclass
        init (the base is ``@dataclass`` and won't accept an extra field unless
        this subclass redeclares an ``__init__``). Existing call sites
        (``build_compaction``) are unchanged.
        """
        super().__init__(*args, **kwargs)
        self.capture_hook = capture_hook

    @staticmethod
    def _evicted_message_data(
        evicted: list[ModelMessage], settled: list[ModelMessage]
    ) -> list[dict]:
        """Refs + role + text for each dropped message, oldest first.

        ``ref`` is ``{run_id}:{turn_id}`` — ``turn_id`` = ordinal of the
        message's greatest user-prompt anchor within the PRE-eviction ``settled``
        list, identical to the walk search/memory use, so ``read_turn`` can
        re-verify the original. ``None`` when the message carries no ``run_id``.
        """
        anchors = user_anchor_indices(settled)
        rows: list[dict] = []
        for idx, m in enumerate(evicted):
            run_id = getattr(m, "run_id", None)
            tid = next(
                (a for a in range(len(anchors) - 1, -1, -1) if anchors[a] <= idx),
                None,
            )
            rows.append(
                {
                    "ref": f"{run_id}:{tid}"
                    if run_id is not None and tid is not None
                    else None,
                    "role": type(m).__name__,
                    "text": _first_text(m),
                }
            )
        return rows

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext,
    ) -> list[ModelMessage]:
        """Trim only the settled prefix, never consuming the active turn.

        **Mid-turn gate (2026-09-02, Option A):** ``compact`` is driven by
        ``TieredCompaction._escalate`` *before every model request*, so it fires
        mid-turn. To forbid eviction of the ACTIVE user turn, every decision is
        bounded to the completed prefix before the most recent real user prompt:
        the base window is invoked on that prefix alone, with the active turn
        re-appended verbatim. A history that is entirely a live turn (or has
        nothing evictable in completed turns) is returned untouched — the window
        tier never touches it.

        ClearToolResults / ClampOversizedMessages are separate tiers and
        continue to run mid-turn.
        """
        # === Mid-turn gate: locate the active turn's start ===
        anchor: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            if _is_real_user_prompt(messages[i]):
                anchor = i
                break
        # Evictable = strictly before the last real user prompt. If the whole
        # history is the active turn (no completed prefix), nothing may drop.
        if anchor is None or anchor == 0:
            return messages
        settled = list(messages[:anchor])  # completed turns only
        active = messages[anchor:]  # current turn — never evicted
        # Recompute the cutoff against the SETTLED prefix only. If nothing is
        # evictable there (short completed history), skip entirely — even the
        # active tail must not be pushed through the base, which would re-derive
        # an un-clamped cutoff over the full list.
        cutoff = find_safe_cutoff(settled, self.keep_messages)
        if cutoff <= 0:
            return messages
        # Report the eviction to an optional capture hook (plain provenance
        # dicts — never evicts the active turn, fires only on a real trim).
        if self.capture_hook is not None:
            self.capture_hook(
                self._evicted_message_data(settled[:cutoff], settled),
                self.keep_messages,
            )
        # Run the base window over the settled prefix ONLY, then reattach the
        # active turn verbatim. The base's find_safe_cutoff / find_token_cutoff
        # can therefore never reach the active prompt.
        trimmed_settled = await super().compact(settled, ctx)
        return [*trimmed_settled, *active]
