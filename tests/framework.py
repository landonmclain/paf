from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from pydantic_ai import Agent, AgentRunResultEvent, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    SlidingWindowCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence

from paf.capabilities.search import ConversationSearch, SnapshotHistorySource
from paf.capabilities.turn_index import (
    TurnIndexCapability,
    TurnIndexStore,
    TurnSummarizer,
)
from paf.observability.instrumentation import (
    _json_dumps,
    root_span,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUBJECT_MODEL = os.environ.get(
    "BEAM_LITE_SUBJECT_MODEL",
    "openrouter:deepseek/deepseek-v4-flash-0731:nitro",
)
MAX_LINES = int(os.environ.get("LOCOMO_MAX_LINES", "10"))  # injected index tail budget

# --- eviction knobs ---------------------------------------------------------
# target_tokens bounds the LIVE context window (TieredCompaction prunes messages
# as the request overflows it). It does NOT touch the on-disk TurnIndexStore or
# the persisted history -- so it only throttles in-context recognition, it does
# not (and cannot) affect search_index / search_conversation_history recall.
# 400 was far too tight: it forced compaction to spuriously evict evidence
# turns mid-run. 5000 is a middle dial: enough to keep ordinary exchanges in
# context while still evicting early evidence turns -> keeps retrieval pressure
# on (vs 16000/40 which padded context and suppressed search-tool use -- see
# the 20260810-202602 run). Override via env for other dials.
LOCOMO_TARGET_TOKENS = int(os.environ.get("LOCOMO_TARGET_TOKENS", "5000"))
LOCOMO_SW_MAX_MESSAGES = int(os.environ.get("LOCOMO_SW_MAX_MESSAGES", "6"))
LOCOMO_SW_KEEP = int(os.environ.get("LOCOMO_SW_KEEP", "3"))

# Retrieval-ladder knob. DEFAULT (unset) = NO turn index: the TurnIndexCapability
# block is removed from the pipeline and ConversationSearch flips to
# add_instructions=True so the plain BM25 search_conversation_history (plus the
# read-only Memory notebook) is the retrieval ladder. This is the current intended
# default. Set ABLATE_TURN_INDEX=0 to OPT BACK IN to the TurnIndex-owned ladder
# (the 20-Q gated baseline, 0.8333 / recall 0.275, run lmem-20260812-133841).
ABLATE_TURN_INDEX = os.environ.get("ABLATE_TURN_INDEX", "1") == "1"

# Memory presence knob: MEMORY=1 wires the durable Memory capability into the
# harness (inject_memory=True, subject READ-ONLY / writable=False per the M2
# owner model) and swaps the plain SlidingWindowCompaction tier for
# MemoryConsolidationWindow, so the owner sub-agent (writable=True) folds each
# about-to-evict slice into a run-scoped memory store.
# DEFAULT (unset) = ON: SEARCH + MEMORY (no turn index) is the intended standard
# arm. Set MEMORY=0 for a no-memory ablation run (`MEMORY=0 ...`).
MEMORY = os.environ.get("MEMORY", "1") == "1"

# Per-question memory store for the running question (run_lmem only). The
# subject agent is built once but must read the SAME per-question notebook the
# consolidation owner writes. run_lmem sets this to `index_dir.parent /
# "memory" / <question_id>` before each recall; tiny_lmem (single
# conversation) leaves it None so the subject defaults to `mem_dir`.
_CURRENT_MEMORY_DIR: Path | None = None

# Single system prompt owns ALL static guidance — identity, answer rules
# (CONFLICT/INFERENCE), and the retrieval ladder. Capability `get_instructions`
# blocks are OFF (ConversationSearch add_instructions=False, Memory
# guidance="") so this STRING is the one source of recall steering. The only
# thing outside it is the dynamic `MEMORY.md` body, injected as user-role
# context per-request by `before_model_request` (can't live in a static
# system_prompt because it changes per-question).
#
# Design rules baked in:
# - MEMORY > RECENT > inference ordering (CONFLICT).
# - INFERENCE softens refusing: hedge when context strongly implies one answer.
# - Ladder is 2 rungs (in-context, then search_conversation_history) plus
#   read_turn to expand a hit's full turn. `search_memory` is deliberately NOT
#   mentioned: the notebook is always injected in-context, so pointing at a
#   separate notebook tool only muddies the waters.
# - No UNANSWERABLE rule: INFERENCE already covers the honest-refusal case.
SUBJECT_SYSTEM_PROMPT = (
    "You are an assistant with access to your full conversational history "
    "through a retrieval ladder. Answer from the evidence you actually have — "
    "the message history in context and anything you retrieve. "
    "CONFLICT — memory is the source of truth for any fact it covers. Your "
    "durable notebook is the consolidated, most-recently-stated record; trust "
    "it over any raw past turn or your inference. Do not second-guess a "
    "user-stated value (a personal best, a count, an amount). Only when the "
    "notebook is silent on the fact does a recent turn's statement stand as "
    "evidence — and prefer the user's own most recent word over your "
    "reconstruction. "
    "INFERENCE — when the retrieved context does not name the asked-for fact "
    "explicitly, but the history strongly and single-mindedly implies one "
    "specific answer — or the entire session establishes one unambiguous "
    "context — do not stop at \"that's not on record\". Give the implied answer "
    "in hedged form — \"Most likely X, but you never told me the exact Y\" — so "
    "the user gets the useful answer while you stay honest about the "
    "uncertainty. Reserve a flat \"not in our history\" refusal for when the "
    "history is genuinely ambiguous or empty. The hedged inference is preferred "
    "over a bare refusal whenever the evidence points at a single answer. "
    "RETRIEVAL — work cheapest-first; never tool-call for what you already "
    "have: "
    "1. IN CONTEXT — a durable-fact notebook and the conversation so far are "
    "injected into every request. Answer from them if present; never tool-call "
    "for something already in front of you. "
    "2. PAST TURN — call `search_conversation_history` (a wide search over the "
    "raw transcript of past turns) when the answer is NOT already in context. "
    "Search returns a `run_id:turn_id` ref for each hit. "
    "3. READ FULL TURN — when a search hit gives you a ref but not the full "
    "detail, call `read_turn` on that ref to pull the turn's complete content. "
    "Skip rungs that don't apply. "
    "MEMORY — you carry a read-only durable notebook injected in your context. "
    "You never write to or delete from it; a separate consolidation owner is "
    "the sole writer. Never claim something was remembered just because you "
    "read it; only the owner persists facts."
)


def _build_agent(
    store: FileStepStore,
    index_dir: Path,
    *,
    memory_dir: Path | None = None,
    enable_summarizer: bool = True,
    use_memory_owner: bool = True,
) -> Agent:
    """Wire the full PAF pipeline (retrieval ladder ablatable).

    The durable `Memory` capability is included by default (`MEMORY=1`), plus
    the consolidation window that fires the owner on each eviction, targeting
    ``memory_dir`` (default ``index_dir.parent / "memory"``) so a run never
    touches repo `data/memory`. Set `MEMORY=0` for a no-memory ablation.

    Per-question gating (used by run_lmem): when a question already has durable
    turn-index summaries (`enable_summarizer=False`) the `TurnIndexCapability`
    is wired with ``summarizer=None`` so `after_model_request` never calls the
    LLM summarizer — the index is already populated for that question. When it
    has a durable memory snapshot (`use_memory_owner=False`) the plain
    `SlidingWindowCompaction` replaces `MemoryConsolidationWindow` so the owner
    sub-agent never fires. Building the agent per-question is cheap (both are
    config-only dataclasses — no model init at build time).

    Default: a single `system_prompt=` (SUBJECT_SYSTEM_PROMPT) owns ALL static
    guidance — identity, the CONFLICT/INFERENCE answer rules, and the retrieval
    ladder (in-context -> search_conversation_history -> read_turn). Every
    capability instruction block is OFF: ConversationSearch has
    add_instructions=False and Memory uses guidance="" (only the dynamic
    MEMORY.md body still injects), so the prompt IS the one source of recall
    steering. `search_memory` is deliberately not mentioned in the prompt (the
    notebook is always injected in-context).

    ABLATE_TURN_INDEX (default 1): the TurnIndexCapability block is omitted
    entirely; the ladder is purely ConversationSearch/search per the prompt.
    Set ABLATE_TURN_INDEX=0 to include the index tail + search_index/read_turn
    tools (instructions still off; the prompt names read_turn as rung 3).

    With MEMORY=1: the durable `Memory` capability is loaded (a plain capability,
    subject READ-ONLY / writable=False per the M2 owner model, so the subject
    can't contaminate durable memory) and the plain SlidingWindowCompaction tier
    is replaced by MemoryConsolidationWindow — the owner sub-agent (writable=True,
    the sole writer) folds each about-to-evict slice into ``memory_dir`` before
    the window drops it. Default memory dir is a run-scoped sibling of
    ``index_dir``.
    """
    source = SnapshotHistorySource(store)
    caps = [
        StepPersistence(store=store),
        # ConversationSearch: BM25 wide fallback. By default add_instructions=False
        # because the TurnIndex ladder names search_conversation_history as its LAST
        # rung; under ablation it flips to True -- it IS the ladder now.
        ConversationSearch(
            source=source,
            scope="conversation",
            # Instructions ALWAYS off — the single system prompt (SUBJECT_SYSTEM_PROMPT)
            # owns the retrieval ladder. This tool is just registered; the prompt names it.
            add_instructions=False,
        ),
    ]
    if not ABLATE_TURN_INDEX:
        # TurnIndex: injected tail index (max_lines bounds the tail), plus
        # read_turn / search_index. Instructions ON — this is the SOLE
        # system-prompt retrieval ladder (memory removed). ALWAYS-on real LLM
        # summarizer: sentence + key_terms + outcome. Matches the reference in
        # tiny_lmem. TurnSummarizer defaults its model to SUBJECT_MODEL;
        # explicit keeps tool/model in one place.
        caps.append(
            TurnIndexCapability(
                source=source,
                index_store=TurnIndexStore(index_dir),
                max_lines=MAX_LINES,
                # Instructions off — SUBJECT_SYSTEM_PROMPT owns the ladder.
                add_instructions=False,
                summarizer=(
                    TurnSummarizer(
                        model=SUBJECT_MODEL, budget_chars=160, max_key_terms=6
                    )
                    if enable_summarizer
                    else None
                ),
            )
        )
    sw_tier: SlidingWindowCompaction
    if MEMORY:
        # Memory arm: durable notebook injected every request. Per the M2 owner
        # model (_consolidate.py, design note §2c), the subject is READ-ONLY
        # (writable=False) — it can never contaminate durable memory. The
        # consolidation owner (writable=True) is the sole writer, folding each
        # about-to-evict slice into the store at the eviction boundary.
        from paf.capabilities.memory import Memory
        from paf.capabilities.memory._window import MemoryConsolidationWindow

        mem_dir = memory_dir or (index_dir.parent / "memory")
        # Per-question isolation (run_lmem): the subject's memory store must
        # track the SAME dir the consolidation owner writes. run_lmem sets
        # `_CURRENT_MEMORY_DIR` to `memory/<question_id>` before each recall
        # turn; a store_resolver (resolved per-run, unlike the static store)
        # lets one agent serve every question's notebook without a rebuild.
        # For a single-conversation runner `_CURRENT_MEMORY_DIR` stays
        # None so the store defaults to `mem_dir` as before.
        from paf.capabilities.memory import MarkdownStorage as _MarkdownStorage

        def _mem_store(_ctx):
            root = _CURRENT_MEMORY_DIR or mem_dir
            return _MarkdownStorage(root)

        caps.append(
            Memory(
                store=_MarkdownStorage(mem_dir),
                store_resolver=_mem_store,
                agent_name="assistant",
                inject_memory=True,
                writable=False,
                # guidance="" disables the Memory instruction block entirely —
                # the single system prompt owns all static guidance. Only the
                # dynamic MEMORY.md body still injects (before_model_request).
                guidance="",
            )
        )
        # Consolidation window: same knobs as the plain sliding window, but
        # folds the about-to-evict slice into the shared run-scoped memory dir
        # before dropping it.
        # Consolidation window: same knobs as the plain sliding window, but
        # folds the about-to-evict slice into the shared run-scoped memory dir
        # before dropping it. `use_memory_owner=False` (durable memory exists
        # for the question) keeps the plain window so the owner never fires.
        if use_memory_owner:
            sw_tier = MemoryConsolidationWindow(
                max_messages=LOCOMO_SW_MAX_MESSAGES,
                keep_messages=LOCOMO_SW_KEEP,
                memory_directory=str(mem_dir),
                memory_agent="assistant",
            )
        else:
            sw_tier = SlidingWindowCompaction(
                max_messages=LOCOMO_SW_MAX_MESSAGES,
                keep_messages=LOCOMO_SW_KEEP,
            )
    else:
        sw_tier = SlidingWindowCompaction(
            max_messages=LOCOMO_SW_MAX_MESSAGES,
            keep_messages=LOCOMO_SW_KEEP,
        )
    caps.append(
        # Context compaction: no-summary ladder, LOW target so it trips hard and
        # evicts early evidence turns out of live context -> forces retrieval.
        TieredCompaction(
            target_tokens=LOCOMO_TARGET_TOKENS,
            tiers=[
                ClampOversizedMessages(
                    max_part_tokens=50_000,
                    keep_head_chars=2_000,
                    keep_tail_chars=2_000,
                ),
                ClearToolResults(max_tokens=1, keep_pairs=4),
                sw_tier,
            ],
        ),
    )
    return Agent(
        SUBJECT_MODEL,
        name="locomo-subject",
        # Single system prompt owns all retrieval guidance (ladder + answer
        # rules + read-only memory contract). Capability instructions are off.
        system_prompt=SUBJECT_SYSTEM_PROMPT,
        capabilities=caps,
        model_settings={"temperature": 0},
    )

def _run_turn(agent, prompt: str, cid: str, name: str, session: str) -> dict:
    """Run one turn; return {text, tools, tool_outputs, usage} over a root span.

    Graceful tool-cut (bender guard): the whole recall loop is wrapped in
    `capture_run_messages()`. If the subject runs away calling retrieval tools
    and hits `UsageLimitExceeded`, instead of crashing the run we hand the
    captured history (up to the breach) to a ZERO-tool fallback agent that must
    produce a final answer. `RECALL_REQUEST_LIMIT` (default 50) lets a test
    force the cap low to trip the bender cheaply; a normal run never exceeds
    50 requests so the fallback is a dead path unless the model loops.
    """
    tools: list[str] = []
    tool_outputs: list[str] = []
    text: str = ""
    usage: dict = {
        "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
        "cache_write_tokens": 0, "total_tokens": 0, "requests": 0,
    }
    bailed: bool = False

    from pydantic_ai import capture_run_messages

    # Zero-tool fallback: plain Agent with NO capabilities = no tools, so it
    # is physically incapable of calling search_index / read_turn and must
    # answer from whatever the subject retrieved before the cut.
    cutoff_agent = Agent(
        os.environ.get("RECALL_CUTOFF_MODEL", SUBJECT_MODEL),
        system_prompt=(
            "The previous run was cut off because it called too many tools. "
            "You are finishing its answer. Use the conversation history above "
            "ONLY — do not call any tools. Give the single most likely answer."
        ),
    )
    request_limit = int(os.environ.get("RECALL_REQUEST_LIMIT", "50"))
    # Default ON: cap tool calls per recall turn at 25 so a bender can't run
    # forever (or hit the 50-request wall and crash the whole run). 25 is
    # enough for these benchmarks' retrieval needs; on a genuine runaway the
    # UsageLimitExceeded path hands the accumulated context to a zero-tool
    # fallback for a graceful answer instead of crashing. Override via
    # RECALL_TOOL_CALLS_LIMIT (e.g. 0/'' to disable).
    tool_calls_limit = os.environ.get("RECALL_TOOL_CALLS_LIMIT", "25")
    _ul: dict = {"request_limit": request_limit}
    if tool_calls_limit:
        _ul["tool_calls_limit"] = int(tool_calls_limit)

    async def run():
        nonlocal text, bailed
        with root_span(
            name,
            **{
                "langfuse.observation.input": _json_dumps(
                    {"prompt": prompt, "conversation_id": cid}
                ),
                "langfuse.session.id": session,
            },
        ) as _span:
            with capture_run_messages() as history:
                try:
                    async with agent.run_stream_events(
                        prompt, conversation_id=cid,
                        usage_limits=UsageLimits(**_ul),
                    ) as es:
                        async for event in es:
                            if isinstance(event, FunctionToolCallEvent):
                                tools.append(str(event.part.tool_name or ""))
                            elif isinstance(event, FunctionToolResultEvent):
                                tpart = getattr(event, "part", None)
                                c = getattr(tpart, "content", None)
                                if c is None:
                                    c = getattr(event, "content", None)
                                if isinstance(c, str) and c.strip():
                                    tool_outputs.append(c)
                            elif isinstance(event, AgentRunResultEvent):
                                u = event.result.usage
                                usage["input_tokens"] = getattr(u, "input_tokens", 0) or 0
                                usage["output_tokens"] = getattr(u, "output_tokens", 0) or 0
                                usage["cache_read_tokens"] = getattr(u, "cache_read_tokens", 0) or 0
                                usage["cache_write_tokens"] = getattr(u, "cache_write_tokens", 0) or 0
                                usage["total_tokens"] = getattr(u, "total_tokens", 0) or 0
                                usage["requests"] = getattr(u, "requests", 0) or 0
                                text = event.result.output
                except UsageLimitExceeded:
                    # Bender: cut the model off AND force a graceful answer.
                    # The captured history ends mid-tool-call (the cap fired
                    # before a tool ran), so repair the dangling tool calls
                    # (synthesized 'interrupted' returns) so the fallback can
                    # be handed a valid history + a fresh user prompt without
                    # triggering "Cannot provide a new user prompt when the
                    # message history contains unprocessed tool calls."
                    bailed = True
                    from pydantic_ai._agent_graph import _repair_dangling_tool_calls
                    repaired = _repair_dangling_tool_calls(
                        history, repair_last_response=True
                    )
                    res = await cutoff_agent.run(
                        "Conclude your answer now, based only on the context above.",
                        message_history=repaired,
                    )
                    text = res.output
                    usage["requests"] = len(history) if history else 1

    asyncio.run(run())
    out = {"text": text, "tools": tools, "tool_outputs": tool_outputs, "usage": usage, "bailed": bailed}
    return out
