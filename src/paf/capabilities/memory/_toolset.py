"""Memory tools over the append-only journal (live M2 surface).

This module carries the LIVE memory tool surface after the legacy rehomed
pydantic-ai builtin was removed:

- :class:`JournalAppendToolset` — the M2 owner's ONLY tool (``append_memory``).
- :class:`MemoryWriteResult` — the result shape both ``append_memory`` and the
  organizer's ``finalize_index`` emit.
- shared helpers ``MAIN_FILENAME`` / ``normalize_filename`` / ``list_subfiles`` /
  ``injection_listing_limit`` / ``render_memory_prompt`` used by the
  :class:`~paf.capabilities.memory.Memory` capability's injection + guidance.

The legacy 4-tool ``MemoryToolset`` (``read_memory`` / ``search_memory`` /
``write_memory`` / ``delete_memory``) and its result types /
`_OWNER_*` no-progress read guards were removed — the assistant is inject-only
(``expose_tools=False``) and the owner appends through the journal. The full
original source is archived at ``tests/retired/memory_toolset_reference.py``.
"""

from __future__ import annotations

import re
from typing import Literal, NotRequired

from pydantic_ai import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from typing_extensions import TypedDict

from ._journal import MEMORY_TIERS, MEMORY_TYPES, JournalStore
from ._store import MemoryStore

MAIN_FILENAME = "MEMORY.md"
"""The main notebook file injected as bounded user-role context."""

_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_CHARS_PER_TOKEN = 4
_MAX_SEARCH_QUERY_CHARS = 1_000
_MAX_SEARCH_TERMS = 32
_MIN_FILENAME_LIST_CHARS = 6


class MemoryWriteResult(TypedDict):
    """Result from a memory append — echoes the recorded fact back so the model
    can confirm its append landed.

    ``memory`` is the appended fact text (used by ``append_memory`` and the
    organizer's ``finalize_index`` to echo what was recorded).
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


def injection_listing_limit(max_tokens: int) -> int:
    """Derive a finite backend listing bound from the complete prompt budget."""
    return max(1, max_tokens * _CHARS_PER_TOKEN // _MIN_FILENAME_LIST_CHARS)


def render_memory_prompt(
    main_content: str,
    subfiles: list[str],
    *,
    heading: str,
    guidance: str,
    max_lines: int,
    max_tokens: int,
    main_truncated: bool = False,
    files_truncated: bool = False,
) -> str:
    """Render a complete memory section within the strict approximate token budget."""
    budget = max_tokens * _CHARS_PER_TOKEN
    rendered_guidance = guidance
    source_lines = main_content.rstrip().splitlines()
    kept_lines = source_lines[-max_lines:] if max_lines else []
    dropped_lines = len(source_lines) - len(kept_lines)

    max_filename_candidates = min(len(subfiles), budget // 4)
    shown_files = subfiles[:max_filename_candidates]
    omitted_files = len(subfiles) - len(shown_files)

    rendered = _render_sections(
        heading,
        rendered_guidance,
        kept_lines,
        dropped_lines,
        shown_files,
        omitted_files,
        main_truncated,
        files_truncated,
    )
    while len(rendered) > budget and shown_files:
        shown_files.pop()
        omitted_files += 1
        rendered = _render_sections(
            heading,
            rendered_guidance,
            kept_lines,
            dropped_lines,
            shown_files,
            omitted_files,
            main_truncated,
            files_truncated,
        )
    while len(rendered) > budget and kept_lines:
        kept_lines.pop(0)
        dropped_lines += 1
        rendered = _render_sections(
            heading,
            rendered_guidance,
            kept_lines,
            dropped_lines,
            shown_files,
            omitted_files,
            main_truncated,
            files_truncated,
        )
    if len(rendered) > budget and dropped_lines:
        dropped_lines = 0
        rendered = _render_sections(
            heading,
            rendered_guidance,
            kept_lines,
            dropped_lines,
            shown_files,
            omitted_files,
            main_truncated,
            files_truncated,
        )
    if len(rendered) > budget and rendered_guidance:
        rendered_guidance = rendered_guidance[
            : max(0, len(rendered_guidance) - (len(rendered) - budget))
        ]
        rendered = _render_sections(
            heading,
            rendered_guidance,
            kept_lines,
            dropped_lines,
            shown_files,
            omitted_files,
            main_truncated,
            files_truncated,
        )
    if len(rendered) <= budget:
        return rendered
    if omitted_files or files_truncated:
        marker = "[Memory files omitted; use search_memory]"
        return marker[:budget]
    return rendered[:budget]


def _render_sections(
    heading: str,
    guidance: str,
    main_lines: list[str],
    dropped_lines: int,
    subfiles: list[str],
    omitted_files: int,
    main_truncated: bool,
    files_truncated: bool,
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
        sections.append(f"### {MAIN_FILENAME}\n\n" + "\n".join(lines))
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
    """The M2 owner's ONLY tool: ``append_memory`` -> :class:`JournalStore`.

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
                span.set_attributes({"memory.append.type": type, "memory.append.tier": tier})
            return {
                "memory": fact,
                "file": MAIN_FILENAME,
                "version": str(self._journal.count()),
                "replayed": False,
                "status": "appended",
                "diff": None,
            }
