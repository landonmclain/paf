"""The paf-owned `search_conversation_history` toolset (BM25 over persisted history).

This is the wide last-resort rung of the recall ladder. It reads the SAME
`HistorySource` seam as the harness `ConversationSearch` used, but is fully owned
by paf: the corpus is built with the injected-artifact strip (see `_render`), so
marker-tagged `<turn-index>` / `<memory>` blocks never leak into the BM25 index.

Ranking is the dependency-free BM25 core in `_ranking` (re-homed from the harness,
MIT). The tool is a drop-in replacement for the harness `ConversationSearchToolset`:
same `search_conversation_history(query, run_id=None)` signature and same
`scope='conversation'` fail-closed semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.conversation_search._source import HistorySource

from ._ranking import Bm25Params, bm25_rank
from ._render import display_lines, format_messages, is_summary_artifact

SearchScope = Literal["all", "conversation"]
"""How much of the store one search may reach (mirrors the harness type)."""

SEARCH_HISTORY_DESCRIPTION = """\
Search all persisted conversation history: earlier turns of this conversation \
(including messages that context compaction dropped from the live context) and \
past runs persisted in the same store.

When the conversation is compacted, older messages are replaced by a summary in \
the active context, but the persisted step history keeps the originals. Use this \
tool to find specific details the live context no longer carries.

Results are ranked by relevance using BM25 -- rare terms and exact matches \
score higher. Multi-word queries search each word independently. Each result \
names the run it came from; pass `run_id` to search only one run, e.g. a run a \
compaction receipt references as its full transcript.

When to use:
- You need to recall exact details from earlier in the conversation
- The conversation summary or a compaction receipt lacks a detail you need
- You need specific code, file paths, or decisions from a past run

When NOT to use:
- The information is still in the current conversation context
- You have a ref from the turn index and only need one turn (use read_turn)
- You only need a matching one-line summary (use search_index first)
- You need external or real-time information (use web tools instead)"""

_CONVERSATION_SCOPE_NOTE = (
    "\n\nThis search covers only the current conversation; runs belonging to other "
    "conversations in the same store are not reachable."
)


@dataclass
class _RunSection:
    """One run's slice of the corpus, index-aligned per run."""

    label: str
    index_lines: list[str]
    display_lines: list[str]
    # Parallel to `index_lines`/`display_lines`: the source-session id of the
    # message at each line (None when the message carries no provenance). Lets
    # the excerpt label carry `session=<id>` so a retrieval can be scored by
    # LongMemEval ID-membership against `answer_session_ids`. General: any
    # message tagged with `metadata['source_session_id']` surfaces it here.
    session_ids: list[str | None] | None = None


class ConversationSearchToolset(FunctionToolset[AgentDepsT]):
    """A single `search_conversation_history` tool over a `HistorySource`.

    A paf-owned drop-in for the harness `ConversationSearchToolset`. The corpus
    is rendered with the injected-artifact strip (`paf.capabilities.search._render`),
    so it never carries `<turn-index>` / `<memory>` blocks that StepPersistence
    snapshotted into User messages.
    """

    def __init__(
        self,
        source: HistorySource,
        *,
        tool_id: str,
        max_matches: int,
        context_lines: int,
        bm25_k1: float,
        bm25_b: float,
        scope: SearchScope = "all",
    ) -> None:
        super().__init__(id=tool_id)
        if max_matches < 0:
            raise ValueError(f"max_matches must be non-negative, got {max_matches!r}.")
        if context_lines < 0:
            raise ValueError(f"context_lines must be non-negative, got {context_lines!r}.")
        if not math.isfinite(bm25_k1) or bm25_k1 < 0:
            raise ValueError(f"bm25_k1 must be a non-negative finite number, got {bm25_k1!r}.")
        if not 0.0 <= bm25_b <= 1.0:
            raise ValueError(f"bm25_b must be between 0.0 and 1.0, got {bm25_b!r}.")
        self._source = source
        self._max_matches = max_matches
        self._context_lines = context_lines
        self._params = Bm25Params(k1=bm25_k1, b=bm25_b)
        self._scope: SearchScope = scope

        description = SEARCH_HISTORY_DESCRIPTION
        if scope == "conversation":
            description += _CONVERSATION_SCOPE_NOTE
        self.add_function(
            self.search_conversation_history,
            name="search_conversation_history",
            description=description,
        )

    async def search_conversation_history(
        self,
        ctx: RunContext[AgentDepsT],
        query: str,
        run_id: str | None = None,
    ) -> str:
        """Search persisted conversation history using BM25 ranking.

        Args:
            ctx: The active run context.
            query: Text to search for. Multi-word queries search each word
                independently -- rare terms score higher than common ones.
            run_id: Search only this run's history. Use it to resolve a run
                another tool or a compaction receipt referenced.
        """
        conversation_id: str | None = None
        if self._scope == "conversation":
            # Fail closed: filtering on `conversation_id is None` would match every
            # unlabelled run in the store (the exposure scope exists to prevent).
            if ctx.conversation_id is None:
                return _NO_CONVERSATION_MSG
            conversation_id = ctx.conversation_id

        # Durable messages exclude compaction summary artifacts (see `_render`),
        # so the section lines are the original prompts -- like the harness source.
        sections = await self._load_sections(run_id, conversation_id)
        if run_id is not None and not sections:
            return f"No persisted history for run '{run_id}'."
        total_messages = sum(len(section.display_lines) for section in sections)
        if total_messages == 0:
            return _NO_HISTORY_MSG

        # Rank on the untruncated rendering so terms past the display cutoff stay
        # findable; show the truncated rendering. Documents are index-aligned
        # across the flattened sections; context windows stay within one section.
        flat_index_lines: list[str] = []
        locations: list[tuple[int, int]] = []
        for section_idx, section in enumerate(sections):
            for line_idx, line in enumerate(section.index_lines):
                flat_index_lines.append(line)
                locations.append((section_idx, line_idx))

        ranked = bm25_rank(query, flat_index_lines, self._params)
        if not ranked:
            return f"No matches for '{query}' in {total_messages} persisted messages."

        results: list[str] = []
        shown: set[tuple[int, int]] = set()
        for doc_idx, score in ranked:
            if len(results) == self._max_matches:
                break
            section_idx, line_idx = locations[doc_idx]
            if (section_idx, line_idx) in shown:
                continue
            section = sections[section_idx]
            start = max(0, line_idx - self._context_lines)
            end = min(len(section.display_lines), line_idx + self._context_lines + 1)
            shown.update((section_idx, idx) for idx in range(start, end))
            excerpt = "\n".join(section.display_lines[start:end])
            # Surface the matched message's source-session id on the label so a
            # retrieval can be scored by ID-membership (LongMemEval recall@k).
            sid = (
                section.session_ids[line_idx]
                if section.session_ids is not None and line_idx < len(section.session_ids)
                else None
            )
            label = section.label + (f" | session: {sid}" if sid else "")
            results.append(f"[score: {score:.1f} | {label}]\n{excerpt}")

        header = f"Found {len(results)} match(es) for '{query}' in {total_messages} persisted messages:\n\n"
        return header + "\n\n---\n\n".join(results)

    async def _load_sections(
        self,
        run_id: str | None,
        conversation_id: str | None,
    ) -> list[_RunSection]:
        runs = await self._source.list_runs()
        if conversation_id is not None:
            runs = [run for run in runs if run.conversation_id == conversation_id]
        if run_id is not None:
            runs = [run for run in runs if run.run_id == run_id]
        sections: list[_RunSection] = []
        for run in runs:
            messages: list[ModelMessage] = await self._source.run_history(run_id=run.run_id)
            # Drop compaction-summary artifacts from the corpus (derived,
            # not original): mirror the harness `SnapshotHistorySource`.
            messages = [m for m in messages if not is_summary_artifact(m)]
            if not messages:
                continue
            label = f"run: {run.run_id}"
            if run.conversation_id is not None:
                label += f" | conversation: {run.conversation_id}"
            sessions = [
                (getattr(m, "metadata", None) or {}).get("source_session_id")
                if getattr(m, "metadata", None)
                else None
                for m in messages
            ]
            sections.append(
                _RunSection(
                    label=label,
                    index_lines=format_messages(messages, truncate=False),
                    display_lines=display_lines(messages),
                    session_ids=sessions,
                )
            )
        return sections


_NO_CONVERSATION_MSG = (
    "Conversation-scoped search needs a conversation id, and this run has none. "
    "Pass `conversation_id=` to `Agent.run(...)` so the search can tell this "
    "conversation's history from other conversations in the same store."
)

_NO_HISTORY_MSG = (
    "No persisted conversation history to search yet. History accumulates "
    "as runs persist their steps (pair this capability with `StepPersistence` "
    "on a shared store)."
)
