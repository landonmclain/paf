"""Memory tools over the append-only journal.

This module carries the memory tool surface:

- :class:`JournalAppendToolset` — the owner's ONLY tool (``append_memory``).
- :class:`OverrideToolset` — the assistant's ``memory_archive_suggestion``.
- :class:`MemoryWriteResult` — the result shape ``append_memory`` emits.
- shared helpers ``MAIN_FILENAME`` / ``render_memory_prompt`` used by the
  :class:`~paf.capabilities.memory.Memory` capability's injection + guidance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, NotRequired

from pydantic_ai import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from typing_extensions import TypedDict

from ._journal import (
    MEMORY_SCOPES,
    MEMORY_SOURCES,
    WRITE_TYPES,
    JournalStore,
    derive_tier,
)
from ._overrides import OverrideStore

MAIN_FILENAME = "MEMORY.md"
"""The main notebook file injected as bounded user-role context."""

class MemoryWriteResult(TypedDict):
    """Result from a memory append — echoes the recorded fact back so the model
    can confirm its append landed.

    ``memory`` is the appended fact text.
    """

    memory: NotRequired[str | None]
    file: str
    version: str
    replayed: bool
    status: Literal["created", "appended", "updated"]
    diff: str | None


def render_memory_prompt(
    main_content: str,
    subfiles: list[str],
    *,
    heading: str,
    guidance: str,
    main_truncated: bool = False,
    files_truncated: bool = False,
    show_filename: bool = True,
) -> str:
    """Render a complete memory section for injection.

    Bounding is handled upstream, not here: MEMORY.md is capped per-tier by
    count (``tier_caps`` in the index render) and read capped by
    ``max_memory_size``, so ``main_content`` arrives already bounded. This
    function only assembles the markdown and keeps the informational
    truncation markers (older lines dropped / files omitted / notebook over
    the read cap) so the model knows the view is bounded, not exhaustively
    complete.

    ``guidance`` and ``show_filename`` let the caller shape the surface. The
    instruction channel (``get_instructions``) uses both; the per-turn data
    injection passes ``guidance=""`` and ``show_filename=False`` so the
    user-role ``<memory>`` block carries only the snapshot facts — no usage
    preamble (that lives in the system prompt) and no internal filename
    label (pure token waste the model doesn't need).
    """
    source_lines = main_content.rstrip().splitlines()
    shown_files = list(subfiles)
    omitted_files = 0
    return _render_sections(
        heading,
        guidance,
        source_lines,
        0,
        shown_files,
        omitted_files,
        main_truncated,
        files_truncated,
        show_filename,
    )


def _render_sections(
    heading: str,
    guidance: str,
    main_lines: list[str],
    dropped_lines: int,
    subfiles: list[str],
    omitted_files: int,
    main_truncated: bool,
    files_truncated: bool,
    show_filename: bool = True,
) -> str:
    sections = [f"## {heading}"] if heading else []
    if guidance:
        sections.append(guidance)
    if main_lines or dropped_lines or main_truncated:
        lines = list(main_lines)
        if dropped_lines:
            lines.insert(
                0,
                f'... [{dropped_lines} earlier lines; use read_memory("{MAIN_FILENAME}") for the full notebook] ...',
            )
        if main_truncated:
            lines.append(
                "... [notebook exceeds `max_memory_size`; bounded prefix shown] ..."
            )
        if show_filename:
            sections.append(f"### {MAIN_FILENAME}\n\n" + "\n".join(lines))
        else:
            sections.append("\n".join(lines))
    if subfiles or omitted_files or files_truncated:
        listing = [f"- {name}" for name in subfiles]
        if files_truncated:
            listing.append(
                "- ... [more files omitted; use search_memory to find relevant memory]"
            )
        elif omitted_files:
            listing.append(
                f"- ... [{omitted_files} more files; use search_memory to find relevant memory]"
            )
        sections.append("### Other memory files\n\n" + "\n".join(listing))
    return "\n\n".join(sections)


FACT_MAX_CHARS = 400
"""Per-fact cap for extracted facts: one claim, not a paragraph."""

_TIME_WORDS = re.compile(r"\b(currently|as of|right now)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ExtractionContext:
    """What code knows about the slice being extracted (the model never fills it).

    ``ref_dates`` maps every message label in the slice (``run_id:turn_id``) to
    its turn's date (``""`` when the turn carries none) — the receipt list for
    ``source_ref`` and the source of ``born_on``. ``project`` is the
    conversation's project slug (``None`` = untagged conversation).
    """

    ref_dates: dict[str, str] = field(default_factory=dict)
    project: str | None = None


class JournalAppendToolset(FunctionToolset[AgentDepsT]):
    """The owner's ONLY tool: ``append_memory`` -> :class:`JournalStore`.

    The three-layer memory model makes the owner a pure *appender*: it reads
    nothing through tools and edits nothing in place. It inspects the labeled
    conversation slice in its prompt and calls ``append_memory`` once per
    durable fact, writing an immutable record to the append-only journal.

    The model supplies only what it alone can judge — ``type``, ``fact``,
    ``scope``, ``source`` — plus the ``source_ref`` label it copies. Code
    stamps the rest from :class:`ExtractionContext`: ``born_on`` (the turn's
    date), ``project`` (the conversation's), and ``tier`` (``derive_tier``).
    """

    def __init__(
        self, journal: JournalStore, context: ExtractionContext | None = None
    ) -> None:
        super().__init__(id="memory")
        self._journal = journal
        self._context = context or ExtractionContext()
        self.add_function(self.append_memory, name="append_memory")

    async def append_memory(
        self,
        ctx: RunContext[AgentDepsT],
        type: str,
        fact: str,
        scope: str,
        source: str,
        source_ref: str,
    ) -> MemoryWriteResult:
        """Append ONE durable fact to the append-only memory journal.

        The journal is write-once and never edited, moved, or deleted: each call
        records a new immutable entry. Write one fact per call, in the order you
        encounter it.

        Args:
            type: The FIRST that fits — ``identity``, ``rule``, ``preference``,
                ``decision``, ``context``, ``event``.
            fact: ONE complete, self-contained claim, subject inline, under
                ~300 characters, with no time words ("currently", "as of").
            scope: ``global`` if it would still matter in a conversation about
                a different project; otherwise ``project``.
            source: ``stated`` (the user said it), ``observed`` (the assistant
                reported it from its own work), or ``inferred`` (concluded,
                not said outright).
            source_ref: The message's ``{run_id}:{turn_id}`` label, copied
                exactly.
        """
        with ctx.tracer.start_as_current_span(
            "memory.append", record_exception=False, set_status_on_exception=False
        ) as span:
            ctx_ = self._context
            source_ref = (source_ref or "").strip()
            if type not in WRITE_TYPES:
                raise ModelRetry(
                    f"invalid type {type!r}; use the first that fits of "
                    "identity, rule, preference, decision, context, event — "
                    "or skip the fact if none fits"
                )
            if scope not in MEMORY_SCOPES:
                raise ModelRetry(f"invalid scope {scope!r}; use global or project")
            if source not in MEMORY_SOURCES:
                raise ModelRetry(
                    f"invalid source {source!r}; use stated, observed, or inferred"
                )
            fact = (fact or "").strip()
            if len(fact) > FACT_MAX_CHARS:
                raise ModelRetry(
                    f"fact is {len(fact)} characters; split it into separate "
                    f"one-claim facts under ~300 characters each"
                )
            if _TIME_WORDS.search(fact):
                raise ModelRetry(
                    "remove time words (currently / as of / right now) from the "
                    "fact; the date is recorded separately"
                )
            if ctx_.ref_dates and source_ref not in ctx_.ref_dates:
                raise ModelRetry(
                    f"source_ref {source_ref!r} is not a message label in this "
                    "slice; copy the run_id:turn_id label exactly"
                )
            project = ctx_.project if scope == "project" else None
            try:
                self._journal.append(
                    type=type,
                    fact=fact,
                    tier=derive_tier(type, scope),
                    born_on=ctx_.ref_dates.get(source_ref, ""),
                    source_ref=source_ref,
                    scope=scope,
                    project=project,
                    source=source,
                )
            except ValueError as exc:
                raise ModelRetry(str(exc)) from exc
            if span.is_recording():
                span.set_attributes(
                    {
                        "memory.append.type": type,
                        "memory.append.scope": scope,
                        "memory.append.source": source,
                        "memory.append.project": project or "",
                    }
                )
            return {
                "memory": fact,
                "file": MAIN_FILENAME,
                "version": str(self._journal.count()),
                "replayed": False,
                "status": "appended",
                "diff": None,
            }


class OverrideResult(TypedDict):
    """Result of a memory override — echoes the pending archive back so the
    assistant can confirm its suggestion landed (and that it is under review).
    """

    journal_id: int
    op: str
    by: str
    status: Literal["pending_review"]
    reason: str | None


class _CurationToolset(FunctionToolset[AgentDepsT]):
    """Base for the override sidecar toolset (the assistant's).

    Holds the ``OverrideStore`` handle, the agent name (so the eager
    re-render can resolve the spec's per-tier caps), and nothing else — the
    concrete subclass registers its single curation tool.
    """

    def __init__(self, overrides: OverrideStore, *, agent_name: str = "") -> None:
        super().__init__(id="memory")
        self._overrides = overrides
        self._agent_name = agent_name

    def _refresh_render(self) -> None:
        """Re-materialize ``MEMORY.md`` from the journal, honoring the sidecar.

        This makes an override visible on the very next injection instead of
        waiting for the background consolidation pass (every ``consolidate_every``
        turns). Same deterministic ``write_render`` the owner performs; safe to
        call on any override write.
        """
        from ._journal import JournalStore
        from ._render import write_render

        directory = self._overrides.directory
        md_path = directory.rstrip("/\\") + "/" + MAIN_FILENAME
        write_render(JournalStore(directory), md_path, agent_name=self._agent_name)


FactType = Literal["identity", "rule", "preference", "decision", "context", "event"]


class OverrideToolset(_CurationToolset[AgentDepsT]):
    """The assistant's memory toolset: ``memory_archive_suggestion``, plus
    ``list_memory`` / ``search_memory`` for agents on the scoped render.

    ``memory_archive_suggestion`` is the narrow, modify-only curation channel. It
    does NOT append to the journal or touch the index — it writes a **pending**
    intent to the override sidecar (``memory-overrides.jsonl``), which the render
    pass reflects with a ``*(archive pending review)*`` marker until a reviewer
    resolves it (archive / tier_change / reject).

    ``list_memory`` pages through one section (About you or a project) and
    ``search_memory`` ranks facts by a query — both over the scoped render's
    filing, so they reach everything the render's budgets left out. They are
    registered only when ``browse_agent`` is given (the scoped render's agent
    name; the tier-render benchmark path has no browse tools).
    """

    def __init__(
        self,
        overrides: OverrideStore,
        *,
        agent_name: str = "",
        browse_agent: str | None = None,
    ) -> None:
        super().__init__(overrides, agent_name=agent_name)
        self._browse_agent = browse_agent
        self.add_function(
            self.memory_archive_suggestion, name="memory_archive_suggestion"
        )
        if browse_agent:
            self.add_function(self.list_memory, name="list_memory")
            self.add_function(self.search_memory, name="search_memory")

    def _check_project(self, project: str) -> str:
        from paf.conversations import default_project_store

        slugs = [p["slug"] for p in default_project_store().list(include_archived=True)]
        if project != "global" and project not in slugs:
            raise ModelRetry(
                f'Unknown project {project!r}. Use "global" (facts about the user) '
                f"or a project slug: {', '.join(slugs) or '(none registered)'}."
            )
        return project

    async def list_memory(
        self,
        ctx: RunContext[AgentDepsT],
        project: str,
        type: FactType | None = None,
        offset: int = 0,
    ) -> str:
        """List stored memory facts for one section, a page at a time.

        The injected memory shows only what fits; this lists everything stored
        for a section, in the same order (by type, newest first). Events come
        with their dates and are not limited to recent ones.

        Args:
            project: "global" for facts about the user (the About you section),
                or a project slug (e.g. "paf") for that project's facts.
            type: Only this type of fact (optional).
            offset: Skip this many facts (from a previous page's "more" line).
        """
        from ._scoped import list_memory

        return await list_memory(
            self._overrides.directory,
            agent_name=self._browse_agent or self._agent_name,
            project=self._check_project(project),
            type=type,
            offset=offset,
        )

    async def search_memory(
        self,
        ctx: RunContext[AgentDepsT],
        query: str,
        project: str | None = None,
    ) -> str:
        """Search stored memory facts by keywords.

        Finds facts the injected memory left out. Searches facts about the user
        plus this conversation's project, unless you name a project.

        Args:
            query: Keywords to match against the facts.
            project: Search only this section: "global" (facts about the user)
                or a project slug.
        """
        from paf.conversations import project_for_conversation

        from ._scoped import search_memory

        agent = self._browse_agent or self._agent_name
        if project is not None:
            sections = [self._check_project(project)]
        else:
            current = project_for_conversation(agent, ctx.conversation_id)
            sections = ["global", *([current] if current else [])]
        return await search_memory(
            self._overrides.directory, agent_name=agent, query=query, sections=sections
        )

    async def memory_archive_suggestion(
        self,
        ctx: RunContext[AgentDepsT],
        journal_id: int,
        reason: str,
    ) -> OverrideResult:
        """Suggest archiving a memory fact you believe is stale or wrong.

        The fact is addressed by its journal line id -- the ``#N`` shown on the
        line in the injected memory block. You NEVER edit memory directly; this
        records a *pending* archive suggestion that a reviewer must approve
        before it drops out of the injected view. Until then the fact stays
        visible with a ``*(archive pending review)*`` marker.

        Args:
            journal_id: The journal line id of the fact to archive (the ``#N``
                shown at the end of the fact's line in the injected memory block).
            reason: Why this fact is stale/wrong and should be removed. Be
                specific so the reviewer can judge it.
        """
        with ctx.tracer.start_as_current_span(
            "memory.override.suggest_archive",
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                self._overrides.append(
                    op="archive",
                    target_line=journal_id,
                    by="assistant",
                    reason=reason,
                )
            except ValueError as exc:
                raise ModelRetry(str(exc)) from exc
            # Re-render MEMORY.md NOW so the pending-review marker appears on the
            # next injection instead of waiting for the background consolidation
            # pass (which only fires every `consolidate_every` turns). The render
            # is the same deterministic write the owner performs — this just
            # makes the suggestion immediately visible.
            try:
                self._refresh_render()
            except Exception:  # pragma: no cover - render must not fail the turn
                # A render hiccup (e.g. corrupt journal line) must not drop the
                # suggestion — the background pass will retry it later.
                pass
            if span.is_recording():
                span.set_attributes(
                    {
                        "memory.override.op": "archive",
                        "memory.override.target": journal_id,
                    }
                )
            return {
                "journal_id": journal_id,
                "op": "archive",
                "by": "assistant",
                "status": "pending_review",
                "reason": reason,
            }
