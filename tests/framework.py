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

# Ablation knob: with ABLATE_TURN_INDEX=1 the TurnIndexCapability block is removed
# from the pipeline and ConversationSearch flips to add_instructions=True so the
# plain BM25 search_conversation_history becomes the SOLE retrieval ladder.
# Baseline (unset) = the TurnIndex-owned ladder we benchmarked at the 20-Q gated
# baseline (0.8333 / recall 0.275, run lmem-20260812-133841).
ABLATE_TURN_INDEX = os.environ.get("ABLATE_TURN_INDEX", "0") == "1"

# Memory presence knob: MEMORY=1 wires the durable Memory capability into the
# harness (inject_memory=True, subject READ-ONLY / writable=False per the M2
# owner model) and swaps the plain SlidingWindowCompaction tier for
# MemoryConsolidationWindow, so the owner sub-agent (writable=True) folds each
# about-to-evict slice into a run-scoped memory store. Mirrors ABLATE_TURN_INDEX:
# a module-level env bool, default ON now (memory is the intended standard arm).
# Every runner that shares this builder (run_lmem / tiny_lmem) is byte-identical
# without it. Turn it OFF for a no-memory ablation run: `MEMORY=0 ...`.
MEMORY = os.environ.get("MEMORY", "1") == "1"

# Per-question memory store for the running question (run_lmem only). The
# subject agent is built once but must read the SAME per-question notebook the
# consolidation owner writes. run_lmem sets this to `index_dir.parent /
# "memory" / <question_id>` before each recall; tiny_lmem (single
# conversation) leaves it None so the subject defaults to `mem_dir`.
_CURRENT_MEMORY_DIR: Path | None = None

# Read-only guidance for the subject's Memory surface (writable=False). The
# subject can recall from MEMORY.md but never writes to it (the consolidation
# owner is the sole writer), so it must NOT be told to `write_memory` / CAPTURE.
# TurnIndex already owns the CONFLICT/UNANSWERABLE rules and the recall ladder,
# so this is deliberately minimal and read-only. Kept harness-local (an env
# constant) rather than a source change: `guidance=` overrides the default.
# Opens with a separator so, once concatenated with the turn-index ladder, the
# memory instructions read as a visually distinct section (NOT another rung).
_READONLY_MEMORY_GUIDANCE = (
    "\n"
    "You carry a read-only durable notebook (MEMORY.md) injected below. You "
    "never write or delete from it — memory upkeep happens elsewhere (the "
    "consolidation owner is the only writer). Recall from it; do not maintain "
    "it. For recall use the turn-index ladder (search_memory is read-only).\n"
    "When the notebook (MEMORY.md) conflicts with a raw turn or earlier "
    "statement you recalled, MEMORY WINS: the durable store is the consolidated, "
    "most-recently-stated record. Prefer it over inconsistent one-off turns "
    "(e.g. a stale count or an earlier figure the user later corrected).\n"
)


def _build_agent(
    store: FileStepStore,
    index_dir: Path,
    *,
    memory_dir: Path | None = None,
    enable_summarizer: bool = True,
    use_memory_owner: bool = True,
) -> Agent:
    """Wire the full PAF pipeline (memory-free; retrieval ladder ablatable).

    The durable `Memory` capability is intentionally NOT included by default
    ("memory is not ready / not part of tests near-term" — 2026-08-11). With
    `MEMORY=1` it *is* included (plus the consolidation window that fires the
    owner on each eviction), targeting ``memory_dir`` (default
    ``index_dir.parent / "memory"``) so a run never touches repo `data/memory`.

    Per-question gating (used by run_lmem): when a question already has durable
    turn-index summaries (`enable_summarizer=False`) the `TurnIndexCapability`
    is wired with ``summarizer=None`` so `after_model_request` never calls the
    LLM summarizer — the index is already populated for that question. When it
    has a durable memory snapshot (`use_memory_owner=False`) the plain
    `SlidingWindowCompaction` replaces `MemoryConsolidationWindow` so the owner
    sub-agent never fires. Building the agent per-question is cheap (both are
    config-only dataclasses — no model init at build time).

    Default (ABLATE_TURN_INDEX unset): turn-index owns the ladder --
    TurnIndexCapability.add_instructions=True is the ONE retrieval steering,
    and ConversationSearch stays add_instructions=False so the two don't
    double-announce `search_conversation_history` (the index ladder names it as
    its last rung).

    With ABLATE_TURN_INDEX=1: the TurnIndexCapability block is dropped entirely
    and ConversationSearch flips to add_instructions=True, so the plain BM25
    `search_conversation_history` becomes the SOLE retrieval ladder. This is the
    "conversation search alone" ablation vs the 20-Q gated TurnIndex baseline
    (lmem-20260812-133841: overall 0.8333, recall 0.275, 288,052 tokens).

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
            add_instructions=ABLATE_TURN_INDEX,
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
                add_instructions=True,
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
                guidance=_READONLY_MEMORY_GUIDANCE,
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
