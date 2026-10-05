"""The paf-owned `list_turn_summaries` tool — chronological thread-beat recall.

The navigation-first rung of the recall ladder for *what a conversation was
doing across turns*. It reads the per-conversation ``summaries.jsonl`` the
background thread summarizer already persists (``ThreadSummaryStore``) and
returns blocks oldest -> newest, each with its recency header, dense summary,
and ``{at, what}`` beats. The ``at`` refs are ``{run_id}:{turn_id}`` pointers
the model hands straight to ``read_turn`` for the full turn, or
``search_conversation_history`` for more — so this tool is the *timeline*
navigation step, and search stays the last-resort relevance rung.

It is a pure read: no new storage, no model call, deterministic. It complements
the injected ``<thread-summary>`` block (which shows only the newest few dense
summaries) by making the FULL history — including blocks that have fallen off
the injected window — reachable at per-turn granularity.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from ._capability import _header
from ._store import ThreadSummaryStore

LIST_TURN_SUMMARIES_DESCRIPTION = """\
List this conversation's thread-beat summaries in chronological order (oldest \
first) — a timeline of what happened across turns, from the background thread \
summarizer. Each block shows a recency header, a dense summary of the block's \
arc, and its per-turn beats, each carrying an `at: <run_id>:<turn_id>` ref.

Use this to page back through what this conversation did (including blocks that \
have fallen off the injected <thread-summary> window) before reaching for a \
broad search. To drill into one turn, pass a beat's `at` ref to `read_turn` for \
the full turn, or `search_conversation_history` for more.

`offset`/`limit` page through the blocks (default offset=0, limit=10). \
`conversation_id` is accepted for signature stability but is inert: this tool \
always reads the CURRENT conversation, so a value that differs from the active \
conversation is rejected rather than silently reading another one.

When to use:
- You need to recall what this conversation did across earlier turns
- A block you need has fallen off the injected <thread-summary> window
- You want a chronological timeline, not a relevance-ranked search

When NOT to use:
- The information is still in the current conversation context
- You need a specific fact's location (use search_conversation_history)
- You have a beat's `at` ref and only need that one turn (use read_turn)"""


class ThreadSummaryToolset(FunctionToolset[AgentDepsT]):
    """Exposes the `list_turn_summaries` chronological pager over a capture dir.

    Owns no state of its own — just a ``ThreadSummaryStore`` rooted at the
    per-agent capture directory. The active conversation is resolved from
    ``ctx.conversation_id`` at call time (like ``read_turn``), so the tool is
    conversation-scoped and deterministic.
    """

    def __init__(
        self,
        capture_directory: str | Path,
        *,
        tool_id: str = "thread-summary",
    ) -> None:
        super().__init__(id=tool_id)
        self._capture_directory = Path(capture_directory)
        self.add_function(
            self.list_turn_summaries,
            name="list_turn_summaries",
            description=LIST_TURN_SUMMARIES_DESCRIPTION,
        )

    async def list_turn_summaries(
        self,
        ctx: RunContext[AgentDepsT],
        offset: int = 0,
        limit: int = 10,
        conversation_id: str | None = None,
    ) -> str:
        """Page through this conversation's thread-beat summaries, oldest first.

        `offset`/`limit` page the blocks (default offset=0, limit=10). Each block
        renders its recency header, dense summary, and beats with their `at`
        refs. `conversation_id` is inert: it must match the active conversation
        or the call is rejected (cross-conversation recall is not supported yet).
        """
        if offset < 0:
            raise ModelRetry(f"offset must be >= 0, got {offset!r}.")
        if limit < 1:
            raise ModelRetry(f"limit must be >= 1, got {limit!r}.")
        active = ctx.conversation_id
        if conversation_id is not None and conversation_id != active:
            raise ModelRetry(
                "list_turn_summaries reads only the CURRENT conversation; "
                "cross-conversation recall is not supported yet."
            )
        if active is None:
            return "(no active conversation to list summaries for)"
        store = ThreadSummaryStore(self._capture_directory, active)
        records = store.existing_summaries()
        if not records:
            return "(no thread summaries for this conversation yet)"
        page = records[offset : offset + limit]
        if not page:
            return f"(no summaries at offset {offset}; {len(records)} total)"
        lines: list[str] = []
        for rec in page:
            lines.append(f"{_header(rec)}")
            summary = (rec.summary or "").strip().replace("\n", " ")
            if summary:
                lines.append(f"  {summary}")
            beats = rec.beats or []
            if beats:
                lines.append("  beats:")
                for b in beats:
                    at = b.get("at", "")
                    what = (b.get("what", "") or "").strip()
                    lines.append(f"    · {at}  {what}")
        total = len(records)
        shown = offset + len(page)
        if shown < total:
            lines.append(f"(… {total - shown} more blocks; pass offset={shown} to continue)")
        return "\n".join(lines)
