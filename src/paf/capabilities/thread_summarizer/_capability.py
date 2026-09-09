"""ThreadSummarizer capability: *intra-conversation* thread-beat summarizer.

Distinct from ``Memory``: ``Memory`` is *cross*-conversation durable-fact
extraction, deduped into a shared journal; the thread summarizer is
*intra*-conversation chronological beats — *what this conversation was doing
across turns and how it evolved*, ``(run, turn)``-anchored and per-conversation.
It is an independent, separately-enabled capability so a profile can run thread
beats without Memory (or vice-versa) — it is its own thing with its own
per-conversation watermarks under ``data/summaries/<agent>/conversations/``.

The summarizer is driven by the SAME cadence + watermark machinery as Memory's
owner, but keyed by conversation. There is no eviction WAL here: the trigger
resolves the unsummarized turn-span directly from the persisted step store
(``resolve_unseen_span(conversation_id=...)``), the worker summarizes the
curated user-prompt + final-answer span, and advances a per-conversation
``MemoryWatermark`` on success. See the spec's re-architecture note (2026-09-08).

Phase 3 (2026-09-09): **in-context present surface.** ``before_model_request``
now mirrors ``Memory``'s injection — once per real user turn, strip any prior
``<thread-summary>`` block (matched by a scope-qualified ``TextContent``
metadata marker), re-inject a bounded render of the per-conversation summaries
(oldest -> newest, each with a relative+absolute recency header + turn count).
Injection is free every turn; the *model call* that produced the content lives
only in the background worker, on its cadence — fully decoupled, exactly like
``<memory>``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, TextContent, UserPromptPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext

from paf.capabilities._turns import _is_user_turn
from paf.capabilities.thread_summarizer._store import (
    ThreadSummaryRecord,
    ThreadSummaryStore,
)

#: Default capture root (name matches memory's per-agent convention).
_DEFAULT_CAPTURE_ROOT = "data/summaries"

#: The injected block's open/close markers (mirror ``<memory>``).
_THREAD_DATA_PREFIX = "\n\n<thread-summary>\n"
_THREAD_DATA_SUFFIX = "\n</thread-summary>"

#: The metadata tag carried by the injected ``TextContent``. Scope-qualified
#: with a conversation hash so this capability strips ONLY its own prior block
#: (never ``<memory>``'s) and stays correct across conversation scopes.
_THREAD_PART_METADATA = "paf.thread-summary.v1"

#: How many summary blocks the injected window shows at most (oldest -> newest).
_THREAD_WINDOW = 5
#: Caps the rendered block's size so `<thread-summary>` + `<memory>` coexist
#: within the turn's context budget (see design D#9).
_THREAD_MAX_CHARS = 4000


def capture_directory_for(agent_name: str | None) -> Path:
    """Per-agent capture dir: ``<CWD>/data/summaries/<agent_name>/``.

    Mirrors ``memory_dir_for``/``DEFAULT_MEMORY_DIR`` (``data/memory/<agent>``).
    The thread summarizer keeps its derived summaries + per-conversation
    watermarks below this root, under ``conversations/<hash(cid)>/``.
    """
    return Path(_DEFAULT_CAPTURE_ROOT) / (agent_name or "agent")


def _fmt_dt(iso: str, fmt: str = "%m/%d %H:%M") -> str:
    """Format an ISO timestamp for the render (safe against bad/empty input)."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return iso
    return dt.strftime(fmt)


def _relative_label(iso: str) -> str:
    """Short human recency like ``2d ago`` / ``3h ago`` / ``20m ago`` / ``now``.

    Anchored on the block's ``ended_at`` (when the summarized turns ended), so
    "X ago" means "the content is X old," not "the worker ran X ago."
    """
    if not iso:
        return "some time ago"
    try:
        ended = datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return "some time ago"
    now = datetime.now(ended.tzinfo) if ended.tzinfo else datetime.now()
    delta = now - ended
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _header(rec: ThreadSummaryRecord) -> str:
    """One summary line's header: ``2d ago (09/07 14:30 → 09/08 09:10) (N turns)``.

    Legacy records (pre-timestamp) fall back to ``summarized_at`` for both
    edges so the header still renders rather than showing an empty range.
    """
    start = rec.started_at or rec.summarized_at
    end = rec.ended_at or rec.summarized_at
    relative = _relative_label(end)
    span = f"{_fmt_dt(start)} → {_fmt_dt(end)}".strip(" →")
    turns = len(rec.refs)
    header = f"{relative} ({span}) ({turns} turns)"
    return header


def render_thread_summary(
    records: list[ThreadSummaryRecord],
    *,
    window: int = _THREAD_WINDOW,
    max_chars: int = _THREAD_MAX_CHARS,
) -> str:
    """Render the in-context ``<thread-summary>`` block body.

    Window = the newest up-to-``window`` blocks, shown **oldest -> newest** so
    the arc reads forward and the last line is "where we left off" (which also
    carries the ``(most recent)`` tag). No ``(run, turn)`` refs and no per-beat
    noise — pure awareness; depth is one footer reminder pointing at the
    existing ``search_conversation`` / ``read_turn`` tools. Oldest blocks are
    trimmed first if the render exceeds ``max_chars`` (a hard token-ish cap so
    ``<thread-summary>`` folds inside the turn budget).
    """
    if not records:
        return ""
    # Newest up to `window` first, then reverse for oldest->newest.
    windowed = records[-window:]
    lines: list[str] = []
    for idx, rec in enumerate(windowed):
        is_newest = idx == len(windowed) - 1
        tag = " (most recent)" if is_newest else ""
        summary = (rec.summary or "").strip().replace("\n", " ")
        if not summary:
            summary = "(no summary)"
        lines.append(f"{_header(rec)}{tag}\n  {summary}")
    # Hard cap: trim oldest blocks first while over budget.
    while len(lines) > 1 and sum(len(line) for line in lines) > max_chars:
        lines.pop(0)
    body = "\n".join(lines)
    footer = "− detail: search_conversation() → read_turn() −"
    if len(body) + len(footer) > max_chars:
        footer = ""
        while len(lines) > 1 and len(body) > max_chars:
            lines.pop(0)
            body = "\n".join(lines)
    return _THREAD_DATA_PREFIX + body + ("\n" + footer if footer else "") + _THREAD_DATA_SUFFIX


@dataclass(init=False)
class ThreadSummarizer(AbstractCapability[AgentDepsT]):
    """Background per-conversation thread summarizer (on/off surface).

    ``@dataclass(init=False)`` is REQUIRED, not cosmetic: pydantic-ai's spec
    loader validates that every custom capability class carries
    ``__dataclass_fields__`` (see ``get_capability_registry``'s
    ``_validate_capability``), and ``init=False`` keeps this type's hand-written
    ``__init__`` intact.

    Phase 1/2 rework (2026-09-08): the capture WAL is gone. This capability is
    the allowlist/on-off surface; the runtime trigger + worker own the cadence
    and per-conversation watermarks.

    Phase 3 (2026-09-09): ``before_model_request`` renders + injects the
    per-conversation `<thread-summary>` block once per real user turn (mirrors
    ``Memory``).
    """

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        capture_directory: str | Path | None = None,
        inject_thread_summary: bool = True,
        window: int = _THREAD_WINDOW,
        max_chars: int = _THREAD_MAX_CHARS,
    ) -> None:
        super().__init__()
        self._capture_directory = (
            Path(capture_directory) if capture_directory is not None else capture_directory_for(agent_name)
        )
        self._conversation_id: str | None = None
        self.inject_thread_summary = inject_thread_summary
        self.window = window
        self.max_chars = max_chars

    @property
    def capture_directory(self) -> Path:
        return self._capture_directory

    def _render(self, conversation_id: str) -> str:
        """Read + render this conversation's summaries ('' when none/absent)."""
        store = ThreadSummaryStore(self._capture_directory, conversation_id)
        records = store.existing_summaries()
        if not records:
            return ""
        try:
            return render_thread_summary(records, window=self.window, max_chars=self.max_chars)
        except Exception:  # a render glitch must never break the turn
            return ""

    def _remove_previous_injection(
        self, messages: list[Any], marker: str
    ) -> None:
        """Drop any prior ``<thread-summary>`` block tagged with ``marker``.

        Mirrors ``Memory._remove_previous_injection``: strip the tagged
        ``TextContent`` from every ``UserPromptPart`` so the block never
        accumulates across turns. The bare metadata match is a compatibility
        strip for anything persisted pre-scope-qualification.
        """
        for index, message in enumerate(messages):
            if not isinstance(message, ModelRequest):
                continue
            parts: list[Any] = []
            changed = False
            for part in message.parts:
                if not isinstance(part, UserPromptPart) or isinstance(
                    part.content, str
                ):
                    parts.append(part)
                    continue
                content = [
                    item
                    for item in part.content
                    if not (
                        isinstance(item, TextContent)
                        and item.metadata in (marker, _THREAD_PART_METADATA)
                    )
                ]
                if len(content) == len(part.content):
                    parts.append(part)
                else:
                    changed = True
                    if content:
                        parts.append(replace(part, content=content))
            if changed:
                messages[index] = replace(message, parts=parts)

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Record the active thread; on a real user-turn start, inject the block.

        Mirrors ``Memory.before_model_request`` exactly: injection happens once
        per **real user turn** (``_is_user_turn`` — injected blocks are NOT turn
        starts), never mid tool-loop, so a stale block from the turn's start
        stays put until the next real user turn strips it and re-injects a fresh
        one. The journal is the background worker's, decoupled from the turn; the
        turn-start-only cadence mirrors memory and avoids re-notice noise.

        Marker is scope-qualified by ``conversation_id`` so this capability only
        strips its own prior block and works across conversation scopes.
        """
        conversation_id = ctx.conversation_id
        self._conversation_id = conversation_id
        if not self.inject_thread_summary or conversation_id is None:
            return request_context
        messages = request_context.messages
        # Inject only on a real user-turn start. Guard empty/edge histories.
        if not messages or not _is_user_turn(messages[-1]):
            return request_context
        scope_hash = hashlib.sha256(str(conversation_id).encode()).hexdigest()[:16]
        marker = f"{_THREAD_PART_METADATA}:{scope_hash}"
        self._remove_previous_injection(messages, marker)
        rendered = self._render(conversation_id)
        if not rendered:
            # Nothing to inject — but the prior block may still be stripped above.
            return request_context
        latest = request_context.messages[-1]
        if not isinstance(latest, ModelRequest):  # pragma: no cover — guaranteed by the agent graph
            return request_context
        part = UserPromptPart([TextContent(rendered, metadata=marker)])
        request_context.messages[-1] = replace(
            latest, parts=[*latest.parts, part]
        )
        return request_context

    @classmethod
    def from_spec(cls, **kwargs: object) -> ThreadSummarizer[AgentDepsT]:
        """Construct from serializable options (``agent_name`` / ``capture_directory``)."""
        agent_name = kwargs.get("agent_name")
        capture_directory = kwargs.get("capture_directory")
        inject_thread_summary = kwargs.get("inject_thread_summary", True)
        window = kwargs.get("window", _THREAD_WINDOW)
        max_chars = kwargs.get("max_chars", _THREAD_MAX_CHARS)
        return cls(
            agent_name=str(agent_name) if agent_name is not None else None,
            capture_directory=str(capture_directory) if capture_directory is not None else None,
            inject_thread_summary=bool(inject_thread_summary),
            window=int(window),
            max_chars=int(max_chars),
        )

    @classmethod
    def get_serialization_name(cls) -> str:
        """The name used by custom capability specs (the allowlist key)."""
        return "ThreadSummarizer"
