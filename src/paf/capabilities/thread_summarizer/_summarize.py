"""Structured, off-path thread summarization over a conversation turn span.

Produces **10-turn narrative blocks**: for each block of ~ten complete turns a
single (structured-output) model pass emits sparse ``{at, what}`` beats plus a
dense ``summary`` sentence (the block's narrative arc). Receipts cite the exact
``{run_id}:{turn_id}`` refs; a validator cross-checks every beat ref against the
span so "cited" is provable, not prompt-trust.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from paf.capabilities._turns import nearest_run_id, user_anchor_indices

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings


class Beat(BaseModel):
    """One sparse, receipt-backed marker in a thread block."""

    at: str = Field(
        description="The exact {run_id}:{turn_id} ref of the turn this marks."
    )
    what: str = Field(
        description="What happened at that turn, in a few words, without speculation."
    )


class BeatsRecord(BaseModel):
    """The structured output of one 10-turn block summarization pass."""

    turns_covered: list[str] = Field(default_factory=list)
    beats: list[Beat] = Field(default_factory=list)
    summary: str = Field(
        default="",
        description=(
            "A single dense sentence capturing this block's narrative arc "
            "(started on X, hit Y, landed on Z). Not a per-beat restatement."
        ),
    )

    @model_validator(mode="after")
    def _refs_grounded(self) -> BeatsRecord:
        """Every beat must cite a real covered ref — receipts are non-negotiable."""
        covered = set(self.turns_covered)
        bad = [b.at for b in self.beats if b.at not in covered]
        if bad:
            raise ValueError(f"beat(s) cite refs not in turns_covered: {bad}")
        return self


SummarizeSpan = Callable[[list, str], Awaitable[BeatsRecord]]


def _ref_for(messages: list, idx: int) -> str | None:
    """``{run_id}:{turn_id}`` ref for ``messages[idx]`` (greatest user anchor <= idx)."""
    msg = messages[idx]
    run_id = getattr(msg, "run_id", None)
    if run_id is None:
        # A message with no run_id (e.g. a tool-return + user-prompt combined
        # request from a steered/stopped turn) belongs to the run that made the
        # preceding tool call — fall back so the turn stays addressable.
        run_id = nearest_run_id(messages, idx)
    if run_id is None:
        return None
    anchors = user_anchor_indices(messages)
    tid = next((a for a in range(len(anchors) - 1, -1, -1) if anchors[a] <= idx), None)
    return f"{run_id}:{tid}" if tid is not None else None


@dataclass(frozen=True)
class SummarizerPrompt:
    """Prompt material passed to the background summarizer agent.

    Input is one ~10-turn CONVERSATION block (a list of persisted ``ModelMessage``),
    rendered via the curated extraction — user prompts + turn-closing final
    answers only, with thinking / mid-step tool narration dropped (see
    ``paf.capabilities._window._first_text``).
    """

    span: list
    conversation_id: str | None

    def render(self) -> str:
        from paf.capabilities._window import _first_text

        lines: list[str] = []
        seen_by_role: dict[str, set[str]] = {}
        for i, msg in enumerate(self.span):
            text = _first_text(msg)
            if not text:
                continue  # thinking / tool narration / retry — not durable
            role = type(msg).__name__
            # content-dedup: an identical prompt/answer re-persisted at a later
            # address (clean copy vs. one with an injected block attached) must
            # render once, not twice — collapse to the first occurrence.
            if text in seen_by_role.setdefault(role, set()):
                continue
            seen_by_role[role].add(text)
            # Prefer the conversation-absolute ref stamped by the worker (a
            # multi-block span must cite real {run_id}:{turn_id}, not slice-relative).
            stamped = getattr(msg, "_thread_abs_ref", None)
            ref = stamped or _ref_for(self.span, i)
            lines.append(f"[{ref or 'unanchored'}] {role}: {text}")
        messages = "\n".join(lines)
        # The operating instruction comes from the `thread_summarizer` spec's
        # `instructions:` block (Spec-Driven-Background-Agents) — static prose
        # only, like the owner's CONSOLIDATION_GUIDANCE. The live conversation
        # span is appended here in code (Conversation:/Messages:), the same
        # split the extractor uses (guidance vs CONSOLIDATION_PROMPT). Falls
        # back to DEFAULT_SUMMARIZER_INSTRUCTIONS when the spec is absent.
        return (
            f"{_summarizer_instructions().rstrip()}\n\n"
            f"Conversation: {self.conversation_id}\n"
            f"Messages:\n{messages}"
        )


DEFAULT_SUMMARIZER_INSTRUCTIONS = (
    "Summarize this block of conversation turns as a short narrative thread. "
    "Do not infer facts beyond these messages. Output: a compact set of "
    "{at, what} beats (at = the exact source ref, what = a few words) and one "
    "dense `summary` sentence capturing the arc — started on X, hit Y, landed on Z. "
    "Every `at` must be one of the refs this block covers."
)


def _summarizer_instructions() -> str:
    """The summarizer's operating instruction — spec `instructions` or
    :data:`DEFAULT_SUMMARIZER_INSTRUCTIONS`.

    Static prose only; the conversation span is appended by ``render()``.
    """
    try:
        from paf.specs import resolve_background_spec

        spec, _raw = resolve_background_spec("thread_summarizer")
        if spec.instructions:
            from typing import cast

            return cast(str, spec.instructions)
    except Exception:  # noqa: BLE001 - best-effort sugar
        pass
    return DEFAULT_SUMMARIZER_INSTRUCTIONS


DEFAULT_SUMMARIZER_MODEL = "openrouter:deepseek/deepseek-v4-flash-0731"
"""Default model when the ``thread_summarizer`` spec is absent — independent of
the memory consolidation system (the summarizer is its own capability, not a
memory agent; it should not inherit the memory owner's sampler/provider config)."""


def _default_summarizer_model_settings() -> ModelSettings:
    """Default model_settings when the spec is absent: deterministic sampler +
    the OpenRouter provider pin, matching what the spec declares. Independent of
    the memory consolidation defaults."""
    from typing import cast

    from pydantic_ai import ModelSettings as _MS
    from pydantic_ai.settings import ModelSettings as _PMS

    return cast(
        _MS,
        _PMS(
            temperature=0,
            top_p=1.0,
            extra_body={
                "provider": {
                    "order": ["deepinfra", "parasail", "coreweave"],
                    "quantizations": ["fp8"],
                }
            },
        ),
    )


def _summarizer_spec_config() -> tuple[str, ModelSettings, int] | None:
    """Resolve the summarizer leaf spec (model, model_settings, retries).

    Returns None when the ``thread_summarizer`` spec is absent/unreadable so
    callers fall back to the summarizer-specific :data:`DEFAULT_SUMMARIZER_MODEL`
    + :func:`_default_summarizer_model_settings` (keeps unit tests and non-spec
    deployments working). When present it is the source of truth
    (Spec-Driven-Background-Agents).
    """
    try:
        from paf.specs import resolve_background_spec
    except Exception:  # noqa: BLE001 - best-effort sugar
        return None
    try:
        spec, raw = resolve_background_spec("thread_summarizer")
    except Exception:  # noqa: BLE001 - absent/unreadable spec
        return None
    from typing import cast

    from pydantic_ai import ModelSettings as _MS

    model = spec.model or DEFAULT_SUMMARIZER_MODEL
    settings = spec.model_settings or _default_summarizer_model_settings()
    retries = raw.get("retries")
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 1:
        retries = 5
    return model, cast(_MS, settings), retries


async def summarize_span(span: list, conversation_id: str | None) -> BeatsRecord:
    """Run the real structured summarizer agent over one conversation block."""
    from pydantic_ai import Agent

    from paf.degeneration_guard import run_guarded

    cfg = _summarizer_spec_config()
    if cfg is not None:
        model, model_settings, retries = cfg
    else:
        model, model_settings, retries = (
            DEFAULT_SUMMARIZER_MODEL,
            _default_summarizer_model_settings(),
            5,
        )
    agent = Agent(
        model,
        name="thread-summarizer",
        output_type=BeatsRecord,
        model_settings=model_settings,
        retries=retries,
    )
    result = await run_guarded(agent, SummarizerPrompt(span, conversation_id).render())
    return result.output
