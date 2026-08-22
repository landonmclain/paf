"""Memory tools over a scoped, versioned `MemoryStore`."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from typing import TYPE_CHECKING, Literal, NotRequired

from opentelemetry.trace import Span
from pydantic_ai import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from typing_extensions import TypedDict

from ._journal import MEMORY_TYPES, JournalStore
from ._store import (
    MemoryConflictError,
    MemoryMutation,
    MemoryOperation,
    MemorySearchMatch,
    MemorySearchResult,
    MemoryStore,
    SearchableMemoryStore,
)

if TYPE_CHECKING:
    from ._capability import Memory

MAIN_FILENAME = 'MEMORY.md'
"""The main notebook file injected as bounded user-role context."""

_FILENAME_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}')
_CHARS_PER_TOKEN = 4
_MAX_CAS_ATTEMPTS = 16
_MAX_SEARCH_QUERY_CHARS = 1_000
_MAX_SEARCH_TERMS = 32
_MIN_FILENAME_LIST_CHARS = 6
_DIFF_MAX = 8_000  # cap the write-echo diff so a large file can't blow model context
_READ_TRUNCATION_MARKER = (
    '\n\n[Truncated: this file exceeds `max_memory_size`; edit it externally before using `write_memory`.]'
)

# Consecutive-identical tool-call guard, scoped to the memory-owner consolidation
# agent. The owner (and only the owner) has been observed to get stuck in a
# `read_memory`/`search_memory` loop — calling the same tool with the same args
# over and over while the memory content is unchanged, burning model requests
# until the pydantic-ai default `request_limit=50` trips a `UsageLimitExceeded`.
# Same root failure as the identity-write no-op (`old_text == content`), but on
# the read path where nothing raises — every `read_memory` *succeeds*, so the
# built-in ModelRetry/max-retries machinery never engages (it only triggers on
# tool *failure*). This guard tracks the identical-call streak per `Agent.run()`
# (`ctx.run_id`) and, once a streak repeats the same tool+args:
#   * at streak 3–4 it returns a plain *result* (a NUDGE telling the model to
#     call a different tool or finish) — no retry budget is consumed, so the
#     model gets graceful direction instead of a silent death;
#   * at streak >= 5 it raises `ModelRetry`, whose cap is raised to
#     `_OWNER_TOOL_MAX_RETRIES` for these read tools, so pydantic-ai escalates
#     to `UnexpectedModelBehavior` after that many retries and the chunk loop in
#     `_consolidate.py` aborts the chunk as a graceful partial consolidation.
# The streak resets on any different tool call and on each new `Agent.run()`, so
# a legitimate "read the two files once each" / different call is never punished.
_OWNER_AGENT_NAME = 'memory-owner'
# No-progress read-window guard thresholds. The owner consolidates by WRITING;
# reads (`read_memory`/`search_memory`) are only preparation and never change
# state. Once the owner has issued `_OWNER_NUDGE_READS` consecutive read-only
# calls with NO write in between, it is stalling (not converging) — send it a
# redirect nudge. If it keeps reading to `_OWNER_RETRY_READS`, raise ModelRetry
# so pydantic-ai escalates and the chunk loop aborts gracefully.
#
# Escalation math (pydantic-ai semantics): the FIRST ModelRetry raised by the
# guard records the ERROR in the trace; the tool's `retries=_OWNER_TOOL_MAX_RETRIES`
# budget then admits `_OWNER_TOOL_MAX_RETRIES` FURTHER re-invocations before
# pydantic-ai raises `UnexpectedModelBehavior`. So the hard exit lands at
# `_OWNER_RETRY_READS + _OWNER_TOOL_MAX_RETRIES` reads. The soft nudge threads
# (the `_OWNER_NUDGE_READS`..`_OWNER_RETRY_READS` window) are largely cosmetic —
# a stalling model ignores them — the real stop is the ModelRetry + retry tail.
_OWNER_NUDGE_READS = 3  # consecutive reads with no write -> redirect nudge
_OWNER_RETRY_READS = 5  # consecutive reads with no write -> raise ModelRetry
_OWNER_TOOL_MAX_RETRIES = 5  # per-tool `max_retries` for these read tools in the owner
# => hard `UnexpectedModelBehavior` exit at read #10 (5 + 5).


class MemoryWriteResult(TypedDict):
    """Result from `write_memory` — includes the applied diff so the model can
    confirm its edit landed (and detect a mangled structure), mirroring how the
    Hermes `patch` tool returns a diff rather than forcing a re-read.

    ``memory`` is the appended/replaced fact text (used by ``append_memory`` in
    the journal toolset to echo what was recorded); it is unset for the legacy
    write path."""

    memory: NotRequired[str | None]
    file: str
    version: str
    replayed: bool
    status: Literal['created', 'appended', 'updated']
    diff: str | None


class MemoryDeleteResult(TypedDict):
    """Content-free result from `delete_memory`."""

    file: str
    version: str | None
    replayed: bool
    status: Literal['deleted', 'not_found']


class MemorySearchMatchResult(TypedDict):
    """One model-facing, scope-relative search match."""

    file: str
    snippet: str
    score: float


class MemorySearchResponse(TypedDict):
    """Bounded result from `search_memory`."""

    matches: list[MemorySearchMatchResult]
    scanned: int
    truncated: bool


def normalize_filename(file: str) -> str:
    """Validate a model-supplied memory filename and normalize it to `<name>.md`."""
    name = file.strip()
    if name and not name.endswith('.md'):
        name = f'{name}.md'
    if not _FILENAME_RE.fullmatch(name) or '..' in name:
        raise ModelRetry(
            f'{file!r} is not a valid memory filename -- use a short name like "postgres-migration.md" '
            '(letters, digits, dots, dashes; no slashes).'
        )
    return name


def _normalize_search_query(query: str) -> str:
    if len(query) > _MAX_SEARCH_QUERY_CHARS:
        raise ModelRetry(f'Search queries must be at most {_MAX_SEARCH_QUERY_CHARS} characters.')
    normalized = query.strip()
    if not normalized:
        raise ModelRetry('Pass a non-empty search query.')

    terms: list[str] = []
    seen: set[str] = set()
    for term in normalized.split():
        key = term.lower()
        if key in seen:
            continue
        if len(terms) == _MAX_SEARCH_TERMS:
            raise ModelRetry(f'Search queries must contain at most {_MAX_SEARCH_TERMS} unique terms.')
        seen.add(key)
        terms.append(term)
    return ' '.join(terms)


async def list_subfiles(store: MemoryStore, scope: str, *, limit: int) -> tuple[list[str], bool]:
    """Return a bounded filename list and whether additional paths may exist."""
    prefix = f'{scope}/'
    names: list[str] = []
    request_limit = limit + 2  # one main notebook plus one truncation sentinel
    returned_paths = await store.list_paths(prefix, limit=request_limit)
    paths = returned_paths[:request_limit]
    for path in paths:
        if not path.startswith(prefix):
            raise RuntimeError('memory backend returned a path outside the requested scope')
        name = path.removeprefix(prefix)
        if '/' not in name and name != MAIN_FILENAME and name.endswith('.md'):
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
        rendered_guidance = rendered_guidance[: max(0, len(rendered_guidance) - (len(rendered) - budget))]
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
        marker = '[Memory files omitted; use search_memory]'
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
    sections = [f'## {heading}'] if heading else []
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
            lines.append('... [notebook exceeds `max_memory_size`; bounded prefix shown] ...')
        sections.append(f'### {MAIN_FILENAME}\n\n' + '\n'.join(lines))
    if subfiles or omitted_files or files_truncated:
        listing = [f'- {name}' for name in subfiles]
        if files_truncated:
            listing.append('- ... [more files omitted; use search_memory to find relevant memory]')
        elif omitted_files:
            listing.append(f'- ... [{omitted_files} more files; use search_memory to find relevant memory]')
        sections.append('### Other memory files\n\n' + '\n'.join(listing))
    return '\n\n'.join(sections)


def _apply_write(existing: str | None, content: str, old_text: str | None, name: str) -> tuple[str, str]:
    if old_text is None:
        if existing is not None and existing.strip():
            return f'{existing.rstrip()}\n{content.rstrip()}\n', 'appended'
        return f'{content.rstrip()}\n', 'created'
    if existing is None:
        raise ModelRetry(f'There is no memory file named {name!r} to edit -- omit `old_text` to create it.')
    occurrences = existing.count(old_text) if old_text else 0
    if occurrences == 0:
        raise ModelRetry(
            f'`old_text` was not found in {name!r} -- call `read_memory("{name}")` to see its current content.'
        )
    if occurrences > 1:
        raise ModelRetry(
            f'`old_text` appears {occurrences} times in {name!r}; it must match exactly once. '
            'Add surrounding context to make it unique.'
        )
    if old_text == content:
        raise ModelRetry(
            'Replacing a passage with identical text is a no-op and cannot move '
            'it between sections/headers. To relocate a fact, include it in the '
            'correct section within a larger replace, or rewrite the whole file '
            '(in one write, so its old wording is gone).'
        )
    return existing.replace(old_text, content, 1), 'updated'


class JournalAppendToolset(FunctionToolset[AgentDepsT]):
    """The M2 owner's ONLY tool: ``append_memory`` -> :class:`JournalStore`.

    The three-layer memory model makes the owner a pure *appender*: it reads
    nothing through tools and edits nothing in place. It inspects the labeled
    conversation slice in its prompt and calls ``append_memory`` once per
    durable fact, writing an immutable record to the append-only journal. No
    read/search/write/delete surface here — those tools stay on ``MemoryToolset``
    (used by the assistant and the legacy in-place paths).
    """

    def __init__(self, journal: JournalStore) -> None:
        super().__init__(id='memory')
        self._journal = journal
        self.add_function(self.append_memory, name='append_memory')

    async def append_memory(
        self,
        ctx: RunContext[AgentDepsT],
        type: str,
        fact: str,
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
            fact: A plain, self-contained durable statement, subject inline
                ("user prefers X", "project uses Y").
            born_on: The ISO date the fact was stated, from the message's label.
            source_ref: The message's ``{run_id}:{turn_id}`` label, so the fact
                can be traced and verified via ``read_turn``.
        """
        with ctx.tracer.start_as_current_span(
            'memory.append', record_exception=False, set_status_on_exception=False
        ) as span:
            if type not in MEMORY_TYPES:
                raise ModelRetry(
                    f"invalid memory type {type!r}; expected one of {sorted(MEMORY_TYPES)}"
                )
            try:
                self._journal.append(
                    type=type,
                    fact=fact,
                    born_on=born_on,
                    source_ref=source_ref,
                )
            except ValueError as exc:
                raise ModelRetry(str(exc)) from exc
            if span.is_recording():
                span.set_attributes({'memory.append.type': type})
            return {
                'memory': fact,
                'file': MAIN_FILENAME,
                'version': str(self._journal.count()),
                'replayed': False,
                'status': 'appended',
                'diff': None,
            }


class MemoryToolset(FunctionToolset[AgentDepsT]):
    """Scoped read/write/delete/search tools with CAS and durable idempotency.

    The stable `memory` ID lets Temporal and Prefect wrap this static toolset.
    DBOS does not currently turn an ordinary `FunctionToolset` into a durable
    step, so applications requiring DBOS durability must provide that wrapper.
    """

    def __init__(self, capability: Memory[AgentDepsT], *, writable: bool = True) -> None:
        super().__init__(id='memory')
        self._capability = capability
        # Per-run no-progress read-window guard state (see `_guard_repeat`).
        # Counts consecutive read-only calls (`read_memory`/`search_memory`)
        # with no intervening write, reset on each `Agent.run()` and on any
        # `write_memory`/`delete_memory`.
        self._guard_run: str | None = None
        self._reads: list[str] = []  # sig of each consecutive read since last write/reset
        self._guard_wrote: bool = False
        # The read tools get a raised per-tool retry cap in the owner so the
        # `ModelRetry` the guard raises at streak >= 5 has room to redirect
        # before pydantic-ai escalates to `UnexpectedModelBehavior` (aborting the
        # chunk). Per-tool `max_retries` wins over the owner agent's global retry
        # setting (tool_manager derives it from `tool.max_retries` first).
        self.add_function(self.read_memory, name='read_memory', retries=_OWNER_TOOL_MAX_RETRIES)
        self.add_function(
            self.search_memory,
            name='search_memory',
            retries=_OWNER_TOOL_MAX_RETRIES,
        )
        if writable:
            self.add_function(self.write_memory, name='write_memory')
            self.add_function(self.delete_memory, name='delete_memory')

    def _note_memory_write(self, ctx: RunContext[AgentDepsT]) -> None:
        """Reset the no-progress read-window after a write/delete.

        A write is the only *progress* a consolidation owner makes — it mutates
        durable memory. Any `write_memory`/`delete_memory` clears the consecutive
        read count so a healthy read-then-write rhythm is never mistaken for a
        stall.
        """
        run_id = ctx.run_id
        if run_id != self._guard_run:
            self._guard_run = run_id
        self._reads = []
        self._guard_wrote = True

    def _guard_repeat(
        self, ctx: RunContext[AgentDepsT], tool_name: str, *, sig: str
    ) -> str | None:
        """Redirect the memory-owner agent out of a read-only stall.

        The owner (built with ``name=_OWNER_AGENT_NAME``) consolidates by
        WRITING — reads (`read_memory`/`search_memory`) are only preparation and
        never change state. If the owner issues `_OWNER_NUDGE_READS` consecutive
        read-only calls with NO intervening write, it is stalling: the content
        it keeps re-reading is unchanged, and because those calls *succeed*,
        none of pydantic-ai's built-in retry logic fires (it only triggers on
        tool failure). Left alone this burns requests up to the run's
        `request_limit` tripwire — the exact pathology that crashed chunk 8 of
        the seed consolidation. This guard keys on *progress* (writes), not on
        identical args, so it catches every non-converging loop shape —
        identical reads, alternating `read`/`search`, and different-arg churn:

          * at `_OWNER_NUDGE_READS` consecutive reads returns a **nudge** (a plain
            result string) telling the model it is time to write and to stop
            reading. Returning a result costs no retry budget, so the model gets
            graceful direction rather than a silent partial-chunk death;
          * at `_OWNER_RETRY_READS` consecutive reads raises `ModelRetry`.
            Because these read tools are registered with
            `retries=_OWNER_TOOL_MAX_RETRIES`, pydantic-ai escalates the retry
            budget to `UnexpectedModelBehavior` after that many retries, and the
            chunk loop in `_consolidate.py` catches it as a graceful partial
            consolidation.

        Returns ``None`` to let the tool run normally, or a nudge string to
        short-circuit the tool with a redirect. Raises `ModelRetry` to start the
        bounded escalation when the model ignores the nudges.

        The read window resets on each `Agent.run()` (`ctx.run_id`), on any write
        (via `_note_memory_write`), and the sig list keeps the nudge message
        informative about *what* was re-read — but a different sig does NOT reset
        the count, because alternating `read`/`search` is itself a stall.
        """
        # Guard is specific to the consolidation owner; anonymous / recall / live
        # agents never use it.
        owner = getattr(ctx.agent, 'name', None)
        if owner != _OWNER_AGENT_NAME:
            return None
        run_id = ctx.run_id
        if run_id != self._guard_run:
            self._guard_run = run_id
            self._reads = []
            self._guard_wrote = False
        if self._guard_wrote:
            # A write already happened this run; don't punish a fresh read after it.
            self._guard_wrote = False
            self._reads = [sig]
            return None
        self._reads.append(sig)
        n = len(self._reads)
        repeated = any(self._reads.count(s) >= 2 for s in set(self._reads))
        if n >= _OWNER_RETRY_READS:
            raise ModelRetry(
                f'You have made {n} read-only calls ({", ".join(dict.fromkeys(self._reads))}) '
                'with NO write in between. Consolidation only makes progress by WRITING. '
                'You must stop reading and call `write_memory` now (write the fact, update '
                'the notebook, or finish this consolidation chunk).'
            )
        if n >= _OWNER_NUDGE_READS and repeated:
            return (
                f'You have made {n} read-only calls with no write in between, and you '
                f'keep seeing the same content — it is {" / ".join(dict.fromkeys(self._reads))}. '
                'It is time to write. No more reads: call `write_memory` to record the '
                'fact (or the final notebook), or finish this consolidation chunk.'
            )
        return None

    async def write_memory(
        self,
        ctx: RunContext[AgentDepsT],
        content: str,
        file: str = MAIN_FILENAME,
        old_text: str | None = None,
    ) -> MemoryWriteResult:
        """Write persistent memory by appending or uniquely replacing text.

        Omit `old_text` to append, creating the file when necessary. Pass
        `old_text` to replace exactly one matching passage; use an empty
        `content` to remove that passage. Keep short durable facts in
        `MEMORY.md`, and longer or evolving topics in separate files. Update
        stale entries rather than adding contradictory duplicates.

        Write anything you are supposed to keep knowing or tracking — decisions,
        preferences, lessons, and pending actions (errands, pickups, returns,
        deadlines, follow-ups). Persistent memory is cheap and unbounded; history
        is not, so persist a pending item or countable fact (e.g. "3 items to
        pick up") rather than relying on recall from past turns.

        Args:
            ctx: Framework-provided run context.
            content: Text to append, or replacement text for `old_text`.
            file: Memory filename; defaults to `MEMORY.md`.
            old_text: Exact passage to replace, which must occur once.
        """
        capability = self._capability
        self._note_memory_write(ctx)
        name = normalize_filename(file)
        if old_text is None and not content.strip():
            raise ModelRetry('Nothing to write -- pass the text to append, or `old_text` to replace.')
        store, scope = capability.resolve_scope(ctx)
        path = f'{scope}/{name}'
        operation = _operation(ctx, scope, 'write', path, {'content': content, 'file': file, 'old_text': old_text})
        scope_hash = _scope_hash(scope)
        with ctx.tracer.start_as_current_span(
            'memory.write', record_exception=False, set_status_on_exception=False
        ) as span:
            _set_span_base(span, store, scope_hash)
            try:
                previous_mutation = await store.get_operation(operation)
                if previous_mutation is not None:
                    result = _write_result(name, previous_mutation, old_text)
                    _set_span_result(span, 'replayed', replayed=True)
                    return result

                for attempt in range(_MAX_CAS_ATTEMPTS):
                    current = await store.read(path, max_chars=capability.max_memory_size)
                    if current is not None and current.truncated:
                        raise ModelRetry(
                            f'{name!r} exceeds max_memory_size and cannot be changed from partial content; '
                            'edit or replace it through the backing store first.'
                        )
                    updated, status = _apply_write(
                        None if current is None else current.content,
                        content,
                        old_text,
                        name,
                    )
                    if len(updated) > capability.max_memory_size:
                        raise ModelRetry(
                            f'{name!r} would grow to {len(updated)} characters; the limit is '
                            f'{capability.max_memory_size}. Split the content into separate memory files.'
                        )
                    try:
                        mutation = await store.write(
                            path,
                            updated,
                            expected_version=None if current is None else current.version,
                            operation=operation,
                        )
                    except MemoryConflictError:
                        if attempt + 1 == _MAX_CAS_ATTEMPTS:
                            raise ModelRetry(
                                'Memory changed repeatedly while writing; retry the operation.'
                            ) from None
                        continue
                    result = _write_result(
                        name,
                        mutation,
                        old_text,
                        status=status,
                        diff=_build_write_diff(
                            name,
                            None if current is None else current.content,
                            updated,
                            status,
                        ),
                    )
                    _set_span_result(span, 'ok', chars=len(updated), replayed=mutation.replayed)
                    return result
                raise RuntimeError('unreachable CAS retry state')  # pragma: no cover
            except Exception as exc:
                _set_span_error(span, exc)
                raise

    async def read_memory(self, ctx: RunContext[AgentDepsT], file: str) -> str:
        """Read a bounded prefix of one memory file.

        Memory may be stale background context, so verify volatile facts before
        relying on them.

        Args:
            ctx: Framework-provided run context.
            file: Memory filename returned by injection or search.
        """
        name = normalize_filename(file)
        nudge = self._guard_repeat(ctx, 'read_memory', sig=f'read_memory:{name}')
        if nudge is not None:
            return nudge
        store, scope = self._capability.resolve_scope(ctx)
        with ctx.tracer.start_as_current_span(
            'memory.read', record_exception=False, set_status_on_exception=False
        ) as span:
            _set_span_base(span, store, _scope_hash(scope))
            try:
                memory_file = await store.read(f'{scope}/{name}', max_chars=self._capability.max_memory_size)
                if memory_file is None:
                    _set_span_result(span, 'not_found')
                    # Not an error: a missing file on a fresh store is the expected
                    # initial state (e.g. the owner reading before its first write).
                    # Return an informative message the model can act on (create it)
                    # rather than a ModelRetry round-trip.
                    return (
                        f'There is no memory file named {name!r} yet. '
                        f'Create it with `write_memory`. Use `search_memory` to find existing memory.'
                    )
                _set_span_result(span, 'ok', chars=len(memory_file.content))
                return memory_file.content + (_READ_TRUNCATION_MARKER if memory_file.truncated else '')
            except Exception as exc:
                _set_span_error(span, exc)
                raise

    async def delete_memory(self, ctx: RunContext[AgentDepsT], file: str) -> MemoryDeleteResult:
        """Delete a non-main memory file that is no longer useful.

        `MEMORY.md` cannot be deleted; remove or correct its text with
        `write_memory` instead.

        Args:
            ctx: Framework-provided run context.
            file: Memory filename to delete.
        """
        self._note_memory_write(ctx)
        name = normalize_filename(file)
        if name == MAIN_FILENAME:
            raise ModelRetry(f'{MAIN_FILENAME} is the main notebook; edit it with `write_memory` instead.')
        store, scope = self._capability.resolve_scope(ctx)
        path = f'{scope}/{name}'
        operation = _operation(ctx, scope, 'delete', path, {'file': file})
        with ctx.tracer.start_as_current_span(
            'memory.delete', record_exception=False, set_status_on_exception=False
        ) as span:
            _set_span_base(span, store, _scope_hash(scope))
            try:
                previous_mutation = await store.get_operation(operation)
                if previous_mutation is not None:
                    result = _delete_result(name, previous_mutation)
                    _set_span_result(span, 'replayed', replayed=True)
                    return result

                for attempt in range(_MAX_CAS_ATTEMPTS):
                    current = await store.read(path, max_chars=1)
                    try:
                        mutation = await store.delete(
                            path,
                            expected_version=None if current is None else current.version,
                            operation=operation,
                        )
                    except MemoryConflictError:
                        if attempt + 1 == _MAX_CAS_ATTEMPTS:
                            raise ModelRetry(
                                'Memory changed repeatedly while deleting; retry the operation.'
                            ) from None
                        continue
                    result = _delete_result(name, mutation)
                    _set_span_result(span, result['status'], replayed=mutation.replayed)
                    return result
                raise RuntimeError('unreachable CAS retry state')  # pragma: no cover
            except Exception as exc:
                _set_span_error(span, exc)
                raise

    async def search_memory(self, ctx: RunContext[AgentDepsT], query: str) -> str | MemorySearchResponse:
        """Search memory files in the current tenant and agent scope.

        Results contain bounded snippets; call `read_memory` when a larger
        bounded excerpt is relevant.

        Args:
            ctx: Framework-provided run context.
            query: Terms to find in memory filenames and content.
        """
        query = _normalize_search_query(query)
        nudge = self._guard_repeat(ctx, 'search_memory', sig=f'search_memory:{query}')
        if nudge is not None:
            return nudge
        capability = self._capability
        store, scope = capability.resolve_scope(ctx)
        prefix = f'{scope}/'
        # No memory files yet in this scope: don't run an empty search loop.
        # Return a direct, actionable message so the owner knows the notebook
        # is empty and stops searching (a bare `{'matches': []}` made earlier
        # runs repeat `search_memory` with nothing to find).
        existing = await store.list_paths(prefix, limit=1)
        if not existing:
            return (
                f'There are no memory files in this scope yet. Create them '
                f'with `write_memory` (start with `{MAIN_FILENAME}`). There is '
                f'nothing to search until memory has been written.'
            )
        with ctx.tracer.start_as_current_span(
            'memory.search', record_exception=False, set_status_on_exception=False
        ) as span:
            _set_span_base(span, store, _scope_hash(scope))
            try:
                if isinstance(store, SearchableMemoryStore):
                    result = await store.search(
                        prefix,
                        query,
                        limit=capability.max_search_results,
                        max_files=capability.max_search_files,
                        max_chars=capability.max_search_result_chars,
                        max_file_chars=capability.max_memory_size,
                    )
                else:
                    result = await _fallback_search(
                        store,
                        prefix,
                        query,
                        limit=capability.max_search_results,
                        max_files=capability.max_search_files,
                        max_chars=capability.max_search_result_chars,
                        max_file_chars=capability.max_memory_size,
                    )
                matches, bounded_truncated = _bounded_search_matches(
                    result.matches,
                    prefix,
                    limit=capability.max_search_results,
                    max_chars=capability.max_search_result_chars,
                )
                scanned = min(max(result.scanned, 0), capability.max_search_files)
                truncated = result.truncated or bounded_truncated or result.scanned > capability.max_search_files
                _set_span_result(
                    span,
                    'ok',
                    matches=len(matches),
                    scanned=scanned,
                    chars=sum(len(match['snippet']) for match in matches),
                    truncated=truncated,
                )
                return {'matches': matches, 'scanned': scanned, 'truncated': truncated}
            except Exception as exc:
                _set_span_error(span, exc)
                raise


def _operation(
    ctx: RunContext[AgentDepsT],
    scope: str,
    kind: Literal['write', 'delete'],
    path: str,
    model_args: dict[str, str | None],
) -> MemoryOperation:
    tool_call_id = ctx.tool_call_id
    if not tool_call_id:
        raise RuntimeError('memory mutations require a stable tool_call_id')
    run_id = ctx.run_id
    if not run_id:
        raise RuntimeError('memory mutations require a stable run_id')
    operation_id = _digest({'scope': scope, 'run_id': run_id, 'tool_call_id': tool_call_id})
    fingerprint = _digest({'kind': kind, 'path': path, 'model_args': model_args})
    return MemoryOperation(id=operation_id, fingerprint=fingerprint)


def _digest(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _build_write_diff(name: str, before: str | None, after: str, status: str) -> str | None:
    """Return a compact unified diff of a write so the model can confirm it landed.

    Mirrors the Hermes `patch` tool's contract (return the change back to the
    caller) — the model should never have to guess whether its edit applied, or
    whether it mangled a section. For a brand-new file (`created`) there is no
    `before`, so the "diff" is the full new content. Capped ``_DIFF_MAX`` chars
    so a very large file can't blow the model context budget.
    """
    if before is None or not before:
        diff = after if after else status
    else:
        diff = ''.join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f'a/{name}',
                tofile=f'b/{name}',
                n=1,
            )
        )
    if not diff.strip():
        return None
    return diff[:_DIFF_MAX]


def _write_result(
    name: str,
    mutation: MemoryMutation,
    old_text: str | None,
    *,
    status: str | None = None,
    diff: str | None = None,
) -> MemoryWriteResult:
    if mutation.version is None:
        raise RuntimeError('memory write returned no version')  # pragma: no cover
    if status is None:
        status = 'updated' if old_text is not None else ('appended' if mutation.existed else 'created')
    if status == 'created':
        typed_status: Literal['created', 'appended', 'updated'] = 'created'
    elif status == 'appended':
        typed_status = 'appended'
    else:
        typed_status = 'updated'
    return {
        'file': name,
        'version': mutation.version,
        'replayed': mutation.replayed,
        'status': typed_status,
        'diff': diff,
    }


def _delete_result(name: str, mutation: MemoryMutation) -> MemoryDeleteResult:
    status: Literal['deleted', 'not_found'] = 'deleted' if mutation.existed else 'not_found'
    return {'file': name, 'version': mutation.version, 'replayed': mutation.replayed, 'status': status}


def _search_match(match: MemorySearchMatch, prefix: str, snippet: str) -> MemorySearchMatchResult:
    name = match.path.removeprefix(prefix)
    return {'file': name, 'snippet': snippet, 'score': match.score}


async def _fallback_search(
    store: MemoryStore,
    prefix: str,
    query: str,
    *,
    limit: int,
    max_files: int,
    max_chars: int,
    max_file_chars: int,
) -> MemorySearchResult:
    paths = await store.list_paths(prefix, limit=max_files + 1)
    terms = query.lower().split()
    scored: list[tuple[float, str, str]] = []
    scanned = 0
    content_truncated = False
    for path in paths[:max_files]:
        scanned += 1
        if not path.startswith(prefix):
            raise RuntimeError('memory backend returned a path outside the requested scope')
        name = path.removeprefix(prefix)
        if '/' in name or not _FILENAME_RE.fullmatch(name) or '..' in name:
            continue
        memory_file = await store.read(path, max_chars=max_file_chars)
        if memory_file is None:
            continue
        content_truncated = content_truncated or memory_file.truncated
        lower_path = name.lower()
        lower_content = memory_file.content.lower()
        score = float(sum(lower_content.count(term) + 2 * lower_path.count(term) for term in terms))
        if score:
            scored.append((score, path, memory_file.content))
    scored.sort(key=lambda item: (-item[0], item[1]))
    matches = [
        MemorySearchMatch(path=path, snippet=_search_snippet(content, query, max_chars), score=score)
        for score, path, content in scored[:limit]
    ]
    return MemorySearchResult(
        matches=matches,
        scanned=scanned,
        truncated=len(paths) > max_files or len(scored) > len(matches) or content_truncated,
    )


def _search_snippet(content: str, query: str, max_chars: int) -> str:
    lower = content.lower()
    positions = [lower.find(term) for term in query.lower().split()]
    found = [position for position in positions if position >= 0]
    center = min(found) if found else 0
    start = max(0, center - max_chars // 3)
    end = min(len(content), start + max_chars)
    start = max(0, end - max_chars)
    snippet = content[start:end]
    if start and len(snippet) >= 3:
        snippet = f'...{snippet[3:]}'
    if end < len(content) and len(snippet) >= 3:
        snippet = f'{snippet[:-3]}...'
    return snippet


def _bounded_search_matches(
    matches: list[MemorySearchMatch],
    prefix: str,
    *,
    limit: int,
    max_chars: int,
) -> tuple[list[MemorySearchMatchResult], bool]:
    bounded: list[MemorySearchMatchResult] = []
    remaining = max_chars
    truncated = len(matches) > limit
    for match in matches[:limit]:
        if not match.path.startswith(prefix):
            raise RuntimeError('memory search backend returned a result outside the requested scope')
        name = match.path.removeprefix(prefix)
        if '/' in name or not _FILENAME_RE.fullmatch(name) or '..' in name:
            raise RuntimeError('memory search backend returned a non-file or nested result')
        remaining -= len(name)
        if remaining < 0:
            truncated = True
            break
        snippet = match.snippet[:remaining]
        if len(snippet) < len(match.snippet):
            truncated = True
        bounded.append(_search_match(match, prefix, snippet))
        remaining -= len(snippet)
    return bounded, truncated


def _scope_hash(scope: str) -> str:
    return hashlib.sha256(scope.encode()).hexdigest()[:16]


def _set_span_base(span: Span, store: MemoryStore, scope_hash: str) -> None:
    if span.is_recording():
        span.set_attributes({'memory.backend': type(store).__name__, 'memory.scope_hash': scope_hash})


def _set_span_result(span: Span, outcome: str, **attributes: str | int | bool) -> None:
    if span.is_recording():
        span.set_attributes(
            {'memory.outcome': outcome, **{f'memory.{key}': value for key, value in attributes.items()}}
        )


def _set_span_error(span: Span, exc: Exception) -> None:
    if span.is_recording():
        span.set_attributes({'memory.outcome': 'error', 'memory.exception_type': type(exc).__name__})
