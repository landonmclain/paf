"""Memory tools over the append-only journal.

This module carries the memory tool surface:

- :class:`JournalAppendToolset` — the owner's ONLY tool (``append_memory``).
- :class:`OverrideToolset` — the assistant's ``memory_archive_suggestion``.
- :class:`ReviewerToolset` — the reviewer's ``memory_review`` + ``memory_merge``.
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
    MEMORY_TIERS,
    MEMORY_TYPES,
    WRITE_TYPES,
    JournalStore,
    derive_tier,
)
from ._overrides import OverrideStore

MAIN_FILENAME = "MEMORY.md"
"""The main notebook file injected as bounded user-role context."""

DEFAULT_MAX_ITEM_CHARS = 1200


def _merge_item_char_cap() -> int:
    """The reviewer spec's per-item char cap (``max_item_chars``), default 1200.

    Lazy + best-effort: reads the ``memory_reviewer`` leaf spec's top-level
    ``max_item_chars`` knob; falls back to :data:`DEFAULT_MAX_ITEM_CHARS` when the
    spec is absent/unreadable (so non-spec deployments and unit tests keep a
    sane bound). Imported lazily to avoid a specs ↔ memory circular import.
    """
    try:
        from paf.specs import _spec_scalar, get_spec_path
    except Exception:  # noqa: BLE001 - best-effort sugar
        return DEFAULT_MAX_ITEM_CHARS
    try:
        value = _spec_scalar(get_spec_path("memory_reviewer"), "max_item_chars")
    except Exception:  # noqa: BLE001 - absent/unreadable
        return DEFAULT_MAX_ITEM_CHARS
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return DEFAULT_MAX_ITEM_CHARS


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
    """Shared base for the override sidecar toolsets (assistant + reviewer).

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


class ReviewerResult(TypedDict):
    """Result of a reviewer curation verdict — echoes the applied op back."""

    journal_id: int
    op: Literal["archive", "tier_change", "reject"]
    by: Literal["reviewer"]
    status: Literal["applied"]
    reason: str | None
    tier: str | None


class MemoryMergeResult(TypedDict):
    """Result of a reviewer ``memory_merge`` — echoes the canonical record.

    ``merged_id`` is the new journal line id of the appended consolidated fact;
    ``archived`` lists the line ids of the absorbed sources that were archived.
    """

    merged_id: int
    op: Literal["merge"]
    by: Literal["reviewer"]
    status: Literal["applied"]
    fact: str
    tier: str
    archived: list[int]
    reason: str | None


class ReviewerToolset(_CurationToolset[AgentDepsT]):
    """The memory reviewer's curation tools: ``memory_review`` (+ ``memory_merge``).

    The reviewer is a periodic, off-band curator (like the watermark owner,
    cadence-fired) with authority the assistant lacks. It curates the **whole
    rendered memory**: approving/rejecting the assistant's pending archive
    suggestions AND proactively archiving stale facts / changing tiers / merging
    complementary fragments on its own. It writes ``memory_review`` verdicts
    directly (they apply at once, unlike the assistant's pending intents) to the
    same append-only sidecar the assistant writes to, then re-renders.

    When constructed with a writable :class:`JournalStore` (the production
    reviewer path), it ALSO exposes ``memory_merge`` — its one write path, which
    appends a canonical consolidated record then archives the absorbed sources.
    Without a journal (standalone / test construction) it exposes only
    ``memory_review``; merge is unavailable because there is nowhere to write.

    Read input is the **rendered view only** (``read_render``) — it never reads
    the journal or index directly, addressing facts by their stable ``#N`` line id.
    """

    def __init__(
        self,
        overrides: OverrideStore,
        *,
        agent_name: str = "",
        journal: JournalStore | None = None,
    ) -> None:
        super().__init__(overrides, agent_name=agent_name)
        self._journal = journal
        self.add_function(self.memory_review, name="memory_review")
        # Merge (the reviewer's write path) requires a journal to append to; it
        # is registered only when one is provided (the production reviewer path).
        if journal is not None:
            self.add_function(self.memory_merge, name="memory_merge")

    async def memory_review(
        self,
        ctx: RunContext[AgentDepsT],
        journal_id: int,
        op: Literal["archive", "tier_change", "reject"],
        tier: Literal["t1", "t2", "t3"] | None = None,
        reason: str = "",
    ) -> ReviewerResult:
        """Issue a curation verdict on a memory fact (applies immediately).

        The reviewer is the approver: its verdicts act at once (the assistant's
        suggestions stay pending until a reviewer resolves them, or the reviewer
        curates on its own). Addressed by the fact's journal line id -- the
        ``#N`` shown on the line in the rendered memory view.

        Args:
            journal_id: The journal line id of the fact to curate (the ``#N``
                shown at the end of the fact's line in the rendered memory view).
            op: ``archive`` (drop the fact), ``tier_change`` (move it to ``tier``),
                or ``reject`` (keep as-is, clear a pending marker).
            tier: Required for ``tier_change``: the new tier (``t1``/``t2``/``t3``).
            reason: Why this verdict. Be specific; it is recorded in the sidecar.
        """
        with ctx.tracer.start_as_current_span(
            "memory.override.review",
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                self._overrides.append(
                    op=op,
                    target_line=journal_id,
                    by="reviewer",
                    reason=reason,
                    tier=tier,
                )
            except ValueError as exc:
                raise ModelRetry(str(exc)) from exc
            # Reflect the verdict immediately (drop / re-tier / clear marker).
            self._refresh_render()
            if span.is_recording():
                span.set_attributes(
                    {
                        "memory.override.op": op,
                        "memory.override.target": journal_id,
                    }
                )
            return {
                "journal_id": journal_id,
                "op": op,
                "by": "reviewer",
                "status": "applied",
                "reason": reason,
                "tier": tier,
            }

    async def memory_merge(
        self,
        ctx: RunContext[AgentDepsT],
        merge_ids: list[int],
        fact: str,
        tier: str,
        type: str,
        reason: str = "",
    ) -> MemoryMergeResult:
        """Consolidate several complementary facts into ONE canonical memory.

        Use this when several lines are fragments of ONE fact (they share a
        subject and each carries distinct detail) — e.g. separate lines for a
        user's name, role, family, and location. Archiving alone cannot fix such
        clusters (dropping the enriching lines loses data), so ``memory_merge``:
        (1) appends ``fact`` as a NEW journal record — the single canonical line —
        then (2) archives every line in ``merge_ids``. The cluster collapses to
        the one merged record. Applied immediately and re-rendered at once.

        ``fact`` must be the UNION of the sources' existing detail — if it
        contains any claim in none of the merged lines, that is not a merge, do
        not. You consolidate, you do not invent. You may not merge a single line
        (that is a rewrite, not a merge): ``merge_ids`` must contain at least two
        distinct journal line ids.

        Args:
            merge_ids: At least TWO distinct journal line ids (the ``#N`` shown
                at the end of each fact's line in the rendered memory view) whose
                detail folds into the merged record and which then get archived.
            fact: The single consolidated fact — the union of the sources'
                distinct detail, subject inline, canonically worded.
            tier: The merged fact's permanence tier (``t1``/``t2``/``t3``). It is
                a fresh fact with no keeper line, so you assign placement.
            type: The merged fact's semantic kind — one of ``identity``,
                ``preference``, ``fact``, ``event``, ``task``, ``current``.
            reason: Why this merge; recorded on the archived lines (audit).
        """
        if self._journal is None:
            raise ModelRetry("memory_merge requires a journal; reviewer has none")
        with ctx.tracer.start_as_current_span(
            "memory.override.merge",
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            # --- validate merge_ids (>=2 distinct fully-qualified lines) ----
            try:
                ids = [int(i) for i in merge_ids]
            except (TypeError, ValueError):
                raise ModelRetry(
                    f"merge_ids must be a list of line ids, got {merge_ids!r}"
                ) from None
            if len(ids) < 2:
                raise ModelRetry(
                    f"memory_merge requires at least 2 lines to merge, got {len(ids)}; "
                    "a one-line edit is a rewrite, not a merge"
                )
            if len(set(ids)) != len(ids):
                raise ModelRetry(f"merge_ids must be distinct, got {merge_ids!r}")
            if any(i < 1 for i in ids):
                raise ModelRetry(f"merge_ids must be >= 1, got {merge_ids!r}")

            fact = (fact or "").strip()
            if not fact:
                raise ModelRetry("fact must be non-empty")
            # Per-item size cap (Spec-Driven-Background-Agents): a merged fact
            # must not exceed the reviewer spec's `max_item_chars` (default 1200,
            # all tiers). A merge that would produce a longer line is too big —
            # the model must split it into 2-3 shorter facts instead.
            if len(fact) > _merge_item_char_cap():
                raise ModelRetry(
                    f"merged fact exceeds the {_merge_item_char_cap()}-char per-item "
                    "cap — split it into 2-3 smaller facts instead of merging into "
                    "an over-long line"
                )
            if tier not in MEMORY_TIERS:
                raise ModelRetry(
                    f"invalid memory tier {tier!r}; expected one of {sorted(MEMORY_TIERS)}"
                )
            if type not in MEMORY_TYPES:
                raise ModelRetry(
                    f"invalid memory type {type!r}; expected one of {sorted(MEMORY_TYPES)}"
                )

            # (1) Append the canonical merged record. born_on = the NEWEST of the
            # absorbed sources' born_on dates — a merge is a union of existing
            # facts, so its recency is the max of theirs (consistent with the
            # recency-wins rule); falls back to today only if no source has a
            # date. source_ref records exactly which lines it absorbed so it is
            # fully traceable.
            from datetime import UTC, datetime

            today = datetime.now(UTC).date().isoformat()
            born_by_id = {
                line_id: rec.born_on for line_id, rec in self._journal.read_all()
            }
            source_dates = [born_by_id[i] for i in ids if born_by_id.get(i)]
            born_on = max(source_dates) if source_dates else today
            new_id = self._journal.count() + 1
            try:
                self._journal.append(
                    type=type,
                    fact=fact,
                    tier=tier,
                    born_on=born_on,
                    source_ref="merge(" + ",".join(f"#{i}" for i in ids) + ")",
                )
            except ValueError as exc:
                raise ModelRetry(str(exc)) from exc

            # (2) Archive every absorbed source line through the same sidecar the
            # reviewer already writes verdicts to.
            for line_id in ids:
                try:
                    self._overrides.append(
                        op="archive",
                        target_line=line_id,
                        by="reviewer",
                        reason=reason or f"merged into #{new_id}",
                    )
                except ValueError as exc:
                    raise ModelRetry(str(exc)) from exc

            # (3) Fold the new merged record into the derived index (the render is
            # index-aware, so the new line must be indexed or it won't appear),
            # then reflect the merge immediately — the canonical line shows, the
            # absorbed sources drop from the injected view.
            from ._index import update_index

            update_index(self._overrides.directory)
            try:
                self._refresh_render()
            except Exception:  # pragma: no cover - render must not fail the call
                pass

            if span.is_recording():
                span.set_attributes(
                    {
                        "memory.override.op": "merge",
                        "memory.merge.ids": ids,
                        "memory.merge.merged_id": new_id,
                        "memory.merge.tier": tier,
                    }
                )
            return {
                "merged_id": new_id,
                "op": "merge",
                "by": "reviewer",
                "status": "applied",
                "fact": fact,
                "tier": tier,
                "archived": ids,
                "reason": reason,
            }
