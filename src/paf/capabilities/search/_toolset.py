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

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.conversation_search._source import HistorySource

from ._ranking import Bm25Params, bm25_rank, recency_boost
from ._reader import READ_TURN_DESCRIPTION, ReadTurnToolset, TurnFinder
from ._render import display_lines, format_messages, is_summary_artifact
from ..turn_index._generate import user_anchor_indices

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
names the run it came from and carries a `turn: <run_id>:<turn_id>` ref; pass \
that ref to `read_turn` to pull the FULL matched turn (do not re-search for a \
detail -- a hit's ref expands to the original content). Pass `run_id` to search \
only one run, e.g. a run a compaction receipt references as its full transcript.

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

# ---------------------------------------------------------------------------
# Search loop guard (recall ladder)
# ---------------------------------------------------------------------------
# An agent that issues N consecutive `search_conversation_history` calls with NO
# intervening `read_turn` (full-turn expansion) is looping on a theme — re-running
# near-identical searches in the hope of a different snippet while the context only
# grows with each result. That is the exact pathology that burned one recall to the
# request limit (24 searches, zero read_turn). `read_turn` after a search IS retrieval
# progress (it expands a hit to full content), so it resets the streak. The guard keys
# on progress (did it expand a hit?), not identical args, so it catches every
# non-converging loop shape — identical queries, alternating phrases, and churn.
_SEARCH_NUDGE = 3  # consecutive searches w/o read_turn -> return a redirect nudge
_SEARCH_RETRY = 5  # consecutive searches w/o read_turn -> raise ModelRetry
# Per-tool retry cap so the `ModelRetry` the guard raises has room to redirect before
# pydantic-ai escalates to `UnexpectedModelBehavior` (a spurious hard abort).
_SEARCH_TOOL_RETRIES = 5


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
    # Parallel to `index_lines`/`display_lines`: the message's source datetime
    # as ISO (`metadata['source_date']`, None when absent). Lets a search hit
    # label carry `date=<iso>` so the model can resolve "newer wins" / "which
    # happened first" directly from the hit metadata instead of re-searching to
    # discover the ordering. General: any message tagged with
    # `metadata['source_date']` surfaces it here.
    dates: list[str | None] | None = None
    # Parallel to `display_lines`: the `{run_id}:{turn_id}` ref of the TURN each
    # line belongs to (turn_id = ordinal of the message's user-prompt anchor, the
    # same stable ref the turn index emits). Lets a search hit hand `read_turn`
    # the same pointer format the index uses, so a hit can be expanded to its
    # full turn via the shared read tool.
    turn_refs: list[str | None] | None = None


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
        recency_weight: float = 0.1,
        recency_half_life_days: float = 30.0,
        turn_weight: float = 0.1,
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
        if recency_weight < 0:
            raise ValueError(f"recency_weight must be >= 0, got {recency_weight!r}.")
        if recency_half_life_days <= 0:
            raise ValueError(
                f"recency_half_life_days must be > 0, got {recency_half_life_days!r}."
            )
        if turn_weight < 0:
            raise ValueError(f"turn_weight must be >= 0, got {turn_weight!r}.")
        self._source = source
        self._max_matches = max_matches
        self._context_lines = context_lines
        self._params = Bm25Params(k1=bm25_k1, b=bm25_b)
        self._recency_weight = recency_weight
        self._recency_half_life_days = recency_half_life_days
        self._turn_weight = turn_weight
        self._scope: SearchScope = scope
        self._finder = TurnFinder(source)
        # Per-run search-loop guard state (see `_guard_search`): counts consecutive
        # `search_conversation_history` calls with no `read_turn` in between, reset
        # on each `Agent.run()` and on any `read_turn`. `read_turn` is the expand-a-hit
        # step that constitutes retrieval progress, so reaching it clears the streak.
        self._s_run: str | None = None
        self._sigs: list[str] = []  # query sig of each consecutive search since last read_turn/reset

        description = SEARCH_HISTORY_DESCRIPTION
        if scope == "conversation":
            description += _CONVERSATION_SCOPE_NOTE
        self.add_function(
            self.search_conversation_history,
            name="search_conversation_history",
            description=description,
            retries=_SEARCH_TOOL_RETRIES,
        )
        # Expose the shared full-content read tool (owned by `search`). Because
        # search is the permanent retrieval layer, its search hits can hand the
        # model a `run_id:turn_id` ref to expand -- the model reaches the full
        # turn without any index. Same tool/ref format as the index uses.
        self.add_function(self.read_turn, name="read_turn", description=READ_TURN_DESCRIPTION)

    async def read_turn(
        self,
        ctx: RunContext[AgentDepsT],
        ref: str,
    ) -> str:
        """Resolve a `run_id:turn_id` ref into its full durable turn via pydantic-ai persistence."""
        # Expanding a search hit to its full turn is retrieval PROGRESS — it clears
        # the search streak so a healthy read-search-read rhythm is never a stall.
        self._reset_search_streak(ctx)
        run_id, _, _ = ref.partition(":")
        return await self._finder.resolve(
            ref, run_id=run_id, conversation_id=ctx.conversation_id
        )

    def _reset_search_streak(self, ctx: RunContext[AgentDepsT]) -> None:
        run_id = ctx.run_id
        self._s_run = run_id
        self._sigs = []

    def _bump_search_streak(self, ctx: RunContext[AgentDepsT], sig: str) -> int:
        """Record a `search_conversation_history` call; return the running streak.

        Resets on `Agent.run()` (new run id). Does NOT reset on a different query sig
        — alternating query phrasing is itself a stall. A `read_turn` in between resets
        the streak (see `_reset_search_streak`).
        """
        run_id = ctx.run_id
        if run_id != self._s_run:
            self._s_run = run_id
            self._sigs = []
        self._sigs.append(sig)
        return len(self._sigs)

    def _guard_search(
        self, ctx: RunContext[AgentDepsT], sig: str
    ) -> str | None:
        """Nudge/ModelRetry an agent stuck re-searching without reading a hit.

        Keys on *retrieval progress* (`read_turn`), not identical args, so it catches
        every non-converging search loop (identical queries, alternating phrases,
        query churn). Returns a non-None nudge string to short-circuit the tool with a
        redirect at `_SEARCH_NUDGE`; raises `ModelRetry` at `_SEARCH_RETRY` so the
        bounded retry escalation stops a token-burning loop before the request limit.
        Returns ``None`` to let the search run normally.
        """
        n = self._bump_search_streak(ctx, sig)
        if n >= _SEARCH_RETRY:
            raise ModelRetry(
                f'You have run {n} searches without calling `read_turn` on any hit '
                '({\"' + '", "'.join(dict.fromkeys(self._sigs)) + '"}). Re-searching the same '
                'theme will not change the result — the transcript is static. STOP searching. '
                'Call `read_turn("<hit ref>")` on your best hit to expand it, then answer from '
                'the full turn; or answer now from the evidence you already have.'
            )
        if n >= _SEARCH_NUDGE:
            return (
                f'You have run {n} searches with no `read_turn` in between '
                '({\"' + '", "'.join(dict.fromkeys(self._sigs)) + '"}). You are looping on '
                'this theme. Pick your best hit and call `read_turn` on its turn ref to '
                'expand it — or answer now.'
            )
        return None

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
        # Loop guard: nudge / ModelRetry an agent re-searching without reading a hit.
        nudge = self._guard_search(ctx, sig=f"search:{query.strip()}")
        if nudge is not None:
            return nudge
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

        # Fold recency in as a conservative nudge, mirroring the turn-index
        # ranker (see turn_index/_toolset.py). Pure BM25 is recency-blind, so on
        # an exact-value question ("what was my PREVIOUS best?") the newer turn
        # about the SAME subject can outrank the older turn holding the asked-for
        # value; the model then reasons from the wrong hit. Anchor to the newest
        # dated candidate so ranking stays deterministic (no wall-clock): the boost
        # is in [0, recency_weight], and the default weight (0.1) stays well below
        # one BM25 IDF unit, so it only breaks genuine near-ties and never overrides
        # a real relevance gap. Undated candidates get no boost (treated as stale).
        #
        # The date is SESSION-granular (one datetime per source session, propagated
        # to every message in it), so it cannot order turns WITHIN a session. To
        # recover within-session recency we also fold ``turn_id`` (ordinal of the
        # message's user-prompt anchor within its run = one run per session) in as a
        # small, separately-weighted, NORMALIZED nudge: ``turn_weight * tid/max_tid``
        # where the denominator is the largest turn_id among candidate hits in the
        # SAME run. [0, turn_weight], so it only separates messages that share a
        # session (equal date) and never disturbs the cross-session date ranking or
        # a real BM25 relevance gap.
        blended: list[dict] = []  # {doc_idx, score, date, run, tid}
        for doc_idx, score in ranked:
            section_idx, line_idx = locations[doc_idx]
            section = sections[section_idx]
            dso = (
                section.dates[line_idx]
                if section.dates is not None and line_idx < len(section.dates)
                else None
            )
            run, tid = None, None
            if section.turn_refs is not None and line_idx < len(section.turn_refs):
                tref = section.turn_refs[line_idx]
                if tref and ":" in tref:
                    run, tid_str = tref.rsplit(":", 1)
                    if tid_str.isdigit():
                        tid = int(tid_str)
            blended.append(
                {"doc_idx": doc_idx, "score": score, "date": dso, "run": run, "tid": tid}
            )
        anchor_ts = max(
            (b["date"] for b in blended if b["date"] and b["date"] != ""),
            default="",
        )
        if self._recency_weight and anchor_ts:
            for b in blended:
                if b["date"] and b["date"] != "":
                    b["score"] += self._recency_weight * recency_boost(
                        b["date"], anchor_ts, self._recency_half_life_days
                    )
        if self._turn_weight:
            # Within-session normalization: for each candidate that carries a turn
            # ordinal, divide by the largest turn_id among candidate hits in the
            # SAME run so long/short sessions are on the same [0,1] footing.
            max_tid_by_run: dict[str, int] = {}
            for b in blended:
                if b["run"] is not None and b["tid"] is not None:
                    max_tid_by_run[b["run"]] = max(
                        max_tid_by_run.get(b["run"], -1), b["tid"]
                    )
            for b in blended:
                if b["run"] is not None and b["tid"] is not None:
                    m = max_tid_by_run.get(b["run"])
                    if m:
                        b["score"] += self._turn_weight * (b["tid"] / m)
        blended.sort(key=lambda b: b["score"], reverse=True)

        results: list[str] = []
        shown: set[tuple[int, int]] = set()
        for item in blended:
            if len(results) == self._max_matches:
                break
            doc_idx = item["doc_idx"]
            score = item["score"]
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
            # Surface the matched message's turn ref so the model can expand it to
            # the FULL turn via `read_turn` -- the same `run_id:turn_id` format the
            # index emits. This is the "expand a search hit to full content" path
            # that was missing (and caused non-terminating retrieval loops on
            # full-content questions like LongMemEval's "children's book" item).
            tref = (
                section.turn_refs[line_idx]
                if section.turn_refs is not None and line_idx < len(section.turn_refs)
                else None
            )
            label += (f" | turn: {tref}" if tref else "")
            dso = (
                section.dates[line_idx]
                if section.dates is not None and line_idx < len(section.dates)
                else None
            )
            # Surface the message's source datetime (ISO) so the model can
            # resolve "newer wins" / "which happened first" from the hit itself,
            # instead of re-searching to recover ordering (a loop trigger).
            label += (f" | date: {dso}" if dso else "")
            header_line = f"[score: {score:.1f} | {label}]"
            # Give the model an explicit, copy-paste-able ref for the shared tool.
            hint = f'\n  -> read_turn("{tref}") for the full turn' if tref else ""
            results.append(f"{header_line}{hint}\n{excerpt}")

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
            dates = [
                (getattr(m, "metadata", None) or {}).get("source_date")
                if getattr(m, "metadata", None)
                else None
                for m in messages
            ]
            # Map each message to the `{run_id}:{turn_id}` ref of the TURN it
            # belongs to (turn_id = ordinal of the message's user-prompt anchor,
            # the SAME stable ref the turn index emits). format_messages is
            # one-line-per-message and message-aligned, so turn_refs[i] reaches
            # messages[i]'s turn. Non-user-query messages (assistant responses,
            # tool traffic) inherit their containing turn. Messages before any
            # anchor have no turn ref (None).
            run_ref = run.run_id
            anchors = user_anchor_indices(messages)
            turn_refs: list[str | None] = []
            for idx, _m in enumerate(messages):
                # Find the greatest anchor index <= idx; that anchor's ordinal is
                # the turn_id.
                tid = None
                for a in range(len(anchors) - 1, -1, -1):
                    if anchors[a] <= idx:
                        tid = a
                        break
                turn_refs.append(f"{run_ref}:{tid}" if tid is not None else None)
            sections.append(
                _RunSection(
                    label=label,
                    index_lines=format_messages(messages, truncate=False),
                    display_lines=display_lines(messages),
                    session_ids=sessions,
                    dates=dates,
                    turn_refs=turn_refs,
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
