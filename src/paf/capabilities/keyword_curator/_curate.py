"""The curator model call: sources in, canonical entities + keywords out.

One structured-output pass over a batch of sources, with NO model toolset —
the worker pre-loads the batch (and the already-known entities) into the prompt,
the same shape as the thread summarizer. The model's only freedom is to name
entities and aliases; an output validator checks that every source in the batch
is answered exactly once, so "keyworded" is provable, not trust.

**Drift prevention.** The prompt carries the entities already in the map, and
tells the model to reuse an existing canonical name when a source is about it.
The worker re-derives that list between batches, so a later batch sees what an
earlier one created.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from ._sources import SourceItem
from ._store import normalize_term

if TYPE_CHECKING:
    from pydantic_ai import ModelSettings

MAX_KEYWORDS = 6
MAX_ENTITIES = 5

KEYWORD_CONVENTION = (
    "Keywords are up to 6 concrete nouns, names or vocabulary a future search or "
    "conversation would use — real entity names (tools, APIs, projects, people, "
    "concepts, error codes). NOT generic words: 'Postgres' not 'the database'; "
    "'OAuth' not 'auth'."
)


class Entity(BaseModel):
    """One thing a source is about, with the ways people refer to it."""

    canonical: str = Field(
        description=(
            "The entity's one stable name — short, lowercase, hyphenated if "
            "multi-word. REUSE the exact canonical from the known-entity list "
            "when the source is about that entity."
        )
    )
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Other ways someone would refer to this entity in conversation: "
            "abbreviations, informal names, the phrase they would actually type. "
            "Each alias must be specific enough that seeing it in a message "
            "strongly implies this entity — no generic words."
        ),
    )

    @field_validator("canonical")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not normalize_term(v):
            raise ValueError("canonical must be non-empty")
        return v

    @field_validator("aliases")
    @classmethod
    def _clean_aliases(cls, v: list[str]) -> list[str]:
        return [a.strip() for a in v if normalize_term(a)]


class CurateResult(BaseModel):
    """The curator's answer for one source."""

    source_ref: str = Field(description="The exact [ref] of the source this answers.")
    keywords: list[str] = Field(
        default_factory=list,
        description=f"Up to {MAX_KEYWORDS} concrete searchable terms. {KEYWORD_CONVENTION}",
    )
    entities: list[Entity] = Field(
        default_factory=list,
        description=(
            f"1 to {MAX_ENTITIES} entities this source is mainly about "
            "(the project, tool, or topic it covers) — not every term it mentions."
        ),
    )

    @field_validator("keywords")
    @classmethod
    def _trim_keywords(cls, v: list[str]) -> list[str]:
        return [k.strip() for k in v if k.strip()][:MAX_KEYWORDS]

    @field_validator("entities")
    @classmethod
    def _trim_entities(cls, v: list[Entity]) -> list[Entity]:
        return v[:MAX_ENTITIES]


class CurateBatch(BaseModel):
    """The structured output of one curation pass: one result per source."""

    results: list[CurateResult] = Field(default_factory=list)


CurateFn = Callable[
    [Sequence[SourceItem], Sequence[dict[str, Any]]], Awaitable[list[CurateResult]]
]
"""``(batch, known_entities) -> results`` — the seam the worker calls."""


def check_batch(batch: CurateBatch, refs: Sequence[str]) -> list[str]:
    """Problems with ``batch`` against the refs it must answer (empty = valid).

    Every ref answered exactly once, no unknown refs, every source named at
    least one entity (a source with no entity can never be hinted).
    """
    problems: list[str] = []
    want = set(refs)
    seen: set[str] = set()
    for res in batch.results:
        if res.source_ref not in want:
            problems.append(f"unknown source_ref {res.source_ref!r}")
        elif res.source_ref in seen:
            problems.append(f"source_ref {res.source_ref!r} answered more than once")
        seen.add(res.source_ref)
        if res.source_ref in want and not res.entities:
            problems.append(f"{res.source_ref!r} has no entities")
    missing = [r for r in refs if r not in seen]
    if missing:
        problems.append(f"missing results for: {missing}")
    return problems


# --- prompt --------------------------------------------------------------------

DEFAULT_CURATOR_INSTRUCTIONS = (
    "You maintain a shared vocabulary that lets an assistant recognise when a "
    "conversation is about something it has skills or notes for. For each source "
    "below, name the entities it is mainly about and the ways a person would "
    "refer to them in conversation, plus a few searchable keywords. Reuse an "
    "existing canonical name from the known-entity list whenever a source is "
    "about that entity — do not invent a second name for it. Answer every source "
    "exactly once, citing its exact [ref]. Do not infer anything the source does "
    "not say."
)


def _instructions() -> str:
    try:
        from paf.specs import resolve_background_spec

        spec, _raw = resolve_background_spec("keyword_curator")
        if spec.instructions:
            from typing import cast

            return cast(str, spec.instructions)
    except Exception:  # noqa: BLE001 - best-effort: absent/unreadable spec
        pass
    return DEFAULT_CURATOR_INSTRUCTIONS


def format_known(known: Sequence[dict[str, Any]], *, max_aliases: int = 5) -> str:
    """The known-entity list as prompt lines (``- canonical (aliases: a, b)``)."""
    lines: list[str] = []
    for ent in known:
        aliases = [str(a) for a in (ent.get("aliases") or [])][:max_aliases]
        tail = f" (aliases: {', '.join(aliases)})" if aliases else ""
        lines.append(f"- {ent['canonical']}{tail}")
    return "\n".join(lines)


@dataclass(frozen=True)
class CuratorPrompt:
    """Prompt material for one batch."""

    items: Sequence[SourceItem]
    known: Sequence[dict[str, Any]]

    def render(self) -> str:
        known = format_known(self.known) or "(none yet)"
        sources = "\n\n".join(
            f"[{it.ref}] ({it.source_type}) {it.label}\n{it.text}" for it in self.items
        )
        return (
            f"{_instructions().rstrip()}\n\n"
            f"{KEYWORD_CONVENTION}\n\n"
            f"Known entities (reuse these canonical names):\n{known}\n\n"
            f"Sources:\n{sources}"
        )


# --- model config ----------------------------------------------------------------

DEFAULT_CURATOR_MODEL = "openrouter:deepseek/deepseek-v4-flash-0731"


def _default_model_settings() -> ModelSettings:
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


def _spec_config() -> tuple[str, ModelSettings, int]:
    """``(model, model_settings, retries)`` from the ``keyword_curator`` leaf spec,
    or the defaults when the spec is absent/unreadable."""
    from typing import cast

    from pydantic_ai import ModelSettings as _MS

    try:
        from paf.specs import resolve_background_spec

        spec, raw = resolve_background_spec("keyword_curator")
    except Exception:  # noqa: BLE001 - absent/unreadable spec
        return DEFAULT_CURATOR_MODEL, _default_model_settings(), 5
    retries = raw.get("retries")
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 1:
        retries = 5
    return (
        spec.model or DEFAULT_CURATOR_MODEL,
        cast(_MS, spec.model_settings or _default_model_settings()),
        retries,
    )


def build_agent(refs: Sequence[str], *, model: Any = None):
    """The structured-output curator agent for a batch answering ``refs``.

    An output validator rejects a batch that misses, duplicates, or invents a
    source, so the model retries instead of the worker persisting a bad answer.
    ``model`` overrides the spec's model (tests pass a ``FunctionModel``).
    """
    from pydantic_ai import Agent, ModelRetry

    spec_model, settings, retries = _spec_config()
    agent = Agent(
        model if model is not None else spec_model,
        name="keyword-curator",
        output_type=CurateBatch,
        model_settings=settings,
        retries=retries,
    )

    @agent.output_validator
    def _receipts(output: CurateBatch) -> CurateBatch:
        problems = check_batch(output, refs)
        if problems:
            raise ModelRetry("Invalid answer: " + "; ".join(problems))
        return output

    return agent


async def curate_batch(
    items: Sequence[SourceItem],
    known: Sequence[dict[str, Any]],
    *,
    model: Any = None,
) -> list[CurateResult]:
    """Run the curator over one batch; returns one :class:`CurateResult` per source."""
    from paf.degeneration_guard import run_guarded

    agent = build_agent([it.ref for it in items], model=model)
    result = await run_guarded(agent, CuratorPrompt(items, known).render())
    return list(result.output.results)
