"""Memory tools over the append-only journal.

This module carries the memory tool surface:

- :class:`JournalAppendToolset` — the owner's ONLY tool (``append_memory``).
- :class:`OverrideToolset` — the assistant's ``memory_archive_suggestion``.
- :class:`ReviewerToolset` — the reviewer's ``memory_review`` + ``memory_merge``.
- :class:`MemoryWriteResult` — the result shape ``append_memory`` emits.
- shared helpers ``MAIN_FILENAME`` / ``normalize_filename`` / ``list_subfiles`` /
  ``injection_listing_limit`` / ``render_memory_prompt`` used by the
  :class:`~paf.capabilities.memory.Memory` capability's injection + guidance.
"""

from __future__ import annotations

import re
from typing import Literal, NotRequired

from pydantic_ai import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from typing_extensions import TypedDict

from ._journal import MEMORY_TIERS, MEMORY_TYPES, JournalStore
from ._overrides import OverrideStore
from ._store import MemoryStore

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


_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_CHARS_PER_TOKEN = 4
_MAX_SEARCH_QUERY_CHARS = 1_000
_MAX_SEARCH_TERMS = 32
_MIN_FILENAME_LIST_CHARS = 6

# Maximum topic-file names surfaced in the injected memory listing. Tier caps
# bound MEMORY.md itself; this is a separate, small cap on how many other
# memory files are listed (a handful is plenty — the rest are reachable via
# search_memory).
_MAX_INJECTED_SUBFILES = 20


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


def normalize_filename(file: str) -> str:
    """Validate a model-supplied memory filename and normalize it to `<name>.md`."""
    name = file.strip()
    if name and not name.endswith(".md"):
        name = f"{name}.md"
    if not _FILENAME_RE.fullmatch(name) or ".." in name:
        raise ModelRetry(
            f'{file!r} is not a valid memory filename -- use a short name like "postgres-migration.md" '
            "(letters, digits, dots, dashes; no slashes)."
        )
    return name


async def list_subfiles(
    store: MemoryStore, scope: str, *, limit: int
) -> tuple[list[str], bool]:
    """Return a bounded filename list and whether additional paths may exist."""
    prefix = f"{scope}/"
    names: list[str] = []
    request_limit = limit + 2  # one main notebook plus one truncation sentinel
    returned_paths = await store.list_paths(prefix, limit=request_limit)
    paths = returned_paths[:request_limit]
    for path in paths:
        if not path.startswith(prefix):
            raise RuntimeError(
                "memory backend returned a path outside the requested scope"
            )
        name = path.removeprefix(prefix)
        if "/" not in name and name != MAIN_FILENAME and name.endswith(".md"):
            names.append(name)
    names.sort()
    return names[:limit], len(names) > limit or len(returned_paths) >= request_limit


def injected_subfile_limit() -> int:
    """Fixed cap on the topic-file names surfaced in the injected listing."""
    return _MAX_INJECTED_SUBFILES


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


class JournalAppendToolset(FunctionToolset[AgentDepsT]):
    """The owner's ONLY tool: ``append_memory`` -> :class:`JournalStore`.

    The three-layer memory model makes the owner a pure *appender*: it reads
    nothing through tools and edits nothing in place. It inspects the labeled
    conversation slice in its prompt and calls ``append_memory`` once per
    durable fact, writing an immutable record to the append-only journal.
    """

    def __init__(self, journal: JournalStore) -> None:
        super().__init__(id="memory")
        self._journal = journal
        self.add_function(self.append_memory, name="append_memory")

    async def append_memory(
        self,
        ctx: RunContext[AgentDepsT],
        type: str,
        fact: str,
        tier: str,
        born_on: str,
        source_ref: str,
    ) -> MemoryWriteResult:
        """Append ONE durable fact to the append-only memory journal.

        The journal is write-once and never edited, moved, or deleted: each call
        records a new immutable entry, and the memory notebook (``MEMORY.md``)
        is re-rendered from the journal later (recency-wins, dedup handled by the
        render pass — never by you). Write one fact per call, in the order you
        encounter it.

        Args:
            type: One of the six semantic kinds — ``identity``, ``preference``,
                ``fact``, ``event``, ``task``, ``current``.
            fact: A COMPLETE, self-contained, canonically-worded durable
                statement, subject inline ("user prefers dark roast coffee").
            tier: Permanence tier — ``t1`` stable person, ``t2`` goals/current
                work, ``t3`` ephemeral (may age out of the injected view).
            born_on: The ISO date the fact was stated, from the message's label.
            source_ref: The message's ``{run_id}:{turn_id}`` label, so the fact
                can be traced and verified via ``read_turn``.
        """
        with ctx.tracer.start_as_current_span(
            "memory.append", record_exception=False, set_status_on_exception=False
        ) as span:
            if type not in MEMORY_TYPES:
                raise ModelRetry(
                    f"invalid memory type {type!r}; expected one of {sorted(MEMORY_TYPES)}"
                )
            if tier not in MEMORY_TIERS:
                raise ModelRetry(
                    f"invalid memory tier {tier!r}; expected one of {sorted(MEMORY_TIERS)}"
                )
            try:
                self._journal.append(
                    type=type,
                    fact=fact,
                    tier=tier,
                    born_on=born_on,
                    source_ref=source_ref,
                )
            except ValueError as exc:
                raise ModelRetry(str(exc)) from exc
            if span.is_recording():
                span.set_attributes(
                    {"memory.append.type": type, "memory.append.tier": tier}
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


class OverrideToolset(_CurationToolset[AgentDepsT]):
    """The assistant's ONE curation tool: ``memory_archive_suggestion``.

    This is the narrow, modify-only curation channel. It does NOT append to the
    journal or touch the index — it writes a **pending** intent to the override
    sidecar (``memory-overrides.jsonl``), which the render pass reflects with a
    ``*(archive pending review)*`` marker until a reviewer resolves it (archive /
    tier_change / reject). The assistant remains read-only with respect to the
    journal and index.
    """

    def __init__(self, overrides: OverrideStore, *, agent_name: str = "") -> None:
        super().__init__(overrides, agent_name=agent_name)
        self.add_function(
            self.memory_archive_suggestion, name="memory_archive_suggestion"
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

            # (1) Append the canonical merged record. born_on = today (the merge
            # happens now and the canonical line should be current); source_ref
            # records exactly which lines it absorbed so it is fully traceable.
            from datetime import UTC, datetime

            today = datetime.now(UTC).date().isoformat()
            new_id = self._journal.count() + 1
            try:
                self._journal.append(
                    type=type,
                    fact=fact,
                    tier=tier,
                    born_on=today,
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
