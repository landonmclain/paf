"""LoCoMo sample conversation test — full PAF memory pipeline.

Runtime end-to-end: replay LoCoMo `session_1` (D1:1-D1:18, two-speaker chat)
through the FULL enabled PAF pipeline, then cold-start the answerable
questions and score *evidence recall@k* against LoCoMo's gold `evidence`
dia_ids.

Pipeline exercised (mirrors `ASSISTANT_PROFILE` from `capabilities/profiles.py`,
all over ONE isolated temp store so the run is self-contained and reproducible):

  - StepPersistence          writes full history to a FileStepStore
  - ConversationSearch       BM25 wide fallback over that same store
  - TurnIndexCapability      one-line injected tail index (max_lines bounds the
                             injected tail) + read_turn / search_index; its
                             _INSTRUCTIONS are the SOLE system-prompt ladder
  - TieredCompaction         the no-summary ladder (build_compaction, target
                             LOCOMO_TARGET_TOKENS) -- pruning live context so
                             old turns are NOT in context and real retrieval is
                             required

Retrieval steering comes from ONE place: `TurnIndexCapability.get_instructions()`
(recognition -> search_index -> read_turn -> search_conversation_history).
ConversationSearch is built with `add_instructions=False` because the index
ladder already names it as its last rung. The durable `Memory` capability is
**not** part of the harness (memory is not ready / not in tests as of 2026-08-11).

The `_build_agent(store, index_dir)` here is the SHARED builder used by
`run_lmem.py`, `run_lmem_memoff.py`, and `tiny_lmem.py` -- edit it in one place.

Seed phase: replay each of the 18 session_1 turns as a `{speaker}: {text}` user
message plus an instruction to persist durable facts via `write_memory` (capture
discipline). TurnIndex auto-writes a one-liner per completed turn; StepPersistence
persists the full history; compaction prunes live context as it overflows the
1200-token budget.

Recall phase: ask the answerable questions cold-start against the SAME
conversation_id so persisted history / index / memory are reachable, then score
evidence recall@k against gold `evidence` dia_ids.

Outputs a JSON result to tests/results/.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

from pydantic_ai import Agent, AgentRunResultEvent
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
    setup_observability,
    shutdown_observability,
)
from tests import locomo_scoring as scoring  # noqa: E402  (module under tests/)

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
DATA = Path(__file__).parent.parent / "data" / "benchmarks" / "locomo10.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Multi-conversation run bounds (small by default -- see LoCoMo-Testing.md).
LOCOMO_CONV_LIMIT = int(os.environ.get("LOCOMO_CONV_LIMIT", "3"))
LOCOMO_MAX_QUESTIONS = int(os.environ.get("LOCOMO_MAX_QUESTIONS", "30"))  # across all convs


def _load_samples(n: int | None = None) -> list[dict]:
    """Return the first `n` LoCoMo samples (default: LOCOMO_CONV_LIMIT)."""
    data = json.loads(DATA.read_text())
    if n is None:
        n = LOCOMO_CONV_LIMIT
    return [s for s in data[:n] if s.get("conversation")]


# Full session_1 as the capture window: D1:1..D1:18 (18 turns).
def _seed_turns(conv: dict) -> list[dict]:
    return list(conv["session_1"])


# The answerable questions whose gold `evidence` lies within session 1.
def _recall_questions(qa: list[dict]) -> list[dict]:
    out = []
    for q in qa:
        evs = q.get("evidence") or []
        if evs and all(e.startswith("D1:") for e in evs):
            out.append(q)
    return out


def _gold_answer(q: dict) -> str:
    """LoCoMo gold: plain `answer`, else the `adversarial_answer` variant."""
    return str(q.get("answer") or q.get("adversarial_answer") or "")


# ---------------------------------------------------------------------------
# Agent assembly (full pipeline over ONE isolated temp store)
# ---------------------------------------------------------------------------


def _build_agent(store: FileStepStore, index_dir: Path) -> Agent:
    """Wire the full PAF pipeline (memory-free; turn-index owns the ladder).

    The durable `Memory` capability is intentionally NOT included ("memory is
    not ready / not part of tests near-term" — 2026-08-11). Retrieval steering
    lives solely in TurnIndexCapability._INSTRUCTIONS, which is the ONE
    guidance source; ConversationSearch stays add_instructions=False because
    the index ladder already names search_conversation_history as its last rung.
    """
    source = SnapshotHistorySource(store)
    caps = [
        StepPersistence(store=store),
        # ConversationSearch: BM25 wide fallback. No instructions — the
        # TurnIndex ladder already names it as the LAST rung; a profile that
        # owns the ladder suppresses this short hint.
        ConversationSearch(
            source=source, scope="conversation", add_instructions=False
        ),
        # TurnIndex: injected tail index (max_lines bounds the tail), plus
        # read_turn / search_index. Instructions ON — this is the SOLE
        # system-prompt retrieval ladder (memory removed). ALWAYS-on real LLM
        # summarizer: sentence + key_terms + outcome. Matches the reference in
        # tiny_lmem. TurnSummarizer defaults its model to SUBJECT_MODEL;
        # explicit keeps tool/model in one place.
        TurnIndexCapability(
            source=source,
            index_store=TurnIndexStore(index_dir),
            max_lines=MAX_LINES,
            add_instructions=True,
            summarizer=TurnSummarizer(
                model=SUBJECT_MODEL, budget_chars=160, max_key_terms=6
            ),
        ),
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
                SlidingWindowCompaction(
                    max_messages=LOCOMO_SW_MAX_MESSAGES,
                    keep_messages=LOCOMO_SW_KEEP,
                ),
            ],
        ),
    ]
    return Agent(SUBJECT_MODEL, name="locomo-subject", capabilities=caps)


_SEED_INSTRUCTION = (
    "\n\n[This is a conversation from a chat transcript. Person names are "
    "Caroline (trans woman) and Melanie. Your job, WHILE KEEPING THE CONVERSATION "
    "NATURAL: persist any durable, reusable facts about either person into memory "
    "using `write_memory`. Use concise facts (e.g. 'Melanie painted a lake sunrise "
    "in 2022'). Do NOT persist transient small talk. Acknowledge briefly.]"
)


# ---------------------------------------------------------------------------
# Turn runner (shared with beam-lite conventions)
# ---------------------------------------------------------------------------


def _run_turn(agent, prompt: str, cid: str, name: str, session: str) -> dict:
    """Run one turn; return {text, tools, tool_outputs, usage} over a root span."""
    tools: list[str] = []
    tool_outputs: list[str] = []
    text: str = ""
    usage: dict = {
        "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
        "cache_write_tokens": 0, "total_tokens": 0, "requests": 0,
    }

    async def run():
        nonlocal text
        with root_span(
            name,
            **{
                "langfuse.observation.input": _json_dumps(
                    {"prompt": prompt, "conversation_id": cid}
                ),
                "langfuse.session.id": session,
            },
        ) as _span:
            async with agent.run_stream_events(prompt, conversation_id=cid) as es:
                async for event in es:
                    if isinstance(event, FunctionToolCallEvent):
                        tools.append(str(event.part.tool_name or ""))
                    elif isinstance(event, FunctionToolResultEvent):
                        # Tool results live on the ToolReturnPart; event.content is
                        # unreliable (esp. after ClearToolResults-style compaction).
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

    asyncio.run(run())
    return {"text": text, "tools": tools, "tool_outputs": tool_outputs, "usage": usage}


# ---------------------------------------------------------------------------
# Scoring (LoCoMo-faithful; see locomo_scoring.py)
# ---------------------------------------------------------------------------


def _score_question(
    recall: dict,
    q: dict,
    id_to_text: dict[str, str],
) -> dict:
    """Score one recall row: answer-F1 (per category) + retrieval-recall (tools)."""
    gold = set(q.get("evidence") or [])
    answer = _gold_answer(q)
    category = int(q.get("category") or 0)  # 0 -> unknown -> plain F1 fallback

    # Metric 1: LoCoMo answer F1, category-dispatched.
    recall["answer_f1"] = scoring.score_answer(recall["text"], answer, category)
    # Metric 2: retrieval -- did the recall machinery actually surface the
    # evidence turn content (tool output OR memory/context that the answer
    # reproduced)? Independent of how the model phrased the final answer.
    recall["retrieval_recall"], recall["retrieval_surfaced"] = (
        scoring.tool_surfaced_evidence(
            recall.get("tool_outputs", []), gold, id_to_text,
            answer_text=recall.get("text") or "",
        )
    )
    recall["gold_evidence"] = sorted(gold)
    return recall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    samples = _load_samples()
    if not samples:
        raise RuntimeError("No conversations found in dataset")

    session = f"locomo-{time.strftime('%Y%m%d-%H%M%S')}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    traced = setup_observability()
    if not traced:
        print("[run_locomo] WARNING: Langfuse tracing NOT configured — run will "
              "produce JSON/artifacts but NO Langfuse traces.")

    # Distribute the total question budget across conversations (round-robin by
    # pool size so a big conversation doesn't starve the rest).
    per_conv_pools = []
    for s in samples:
        conv = s["conversation"]
        qs = _recall_questions(s["qa"])
        per_conv_pools.append((conv, qs))
    budget_left = LOCOMO_MAX_QUESTIONS
    per_conv_limits = []
    n_conv = len(per_conv_pools)
    for idx, (conv, qs) in enumerate(per_conv_pools):
        # Even base share, remainder to the first conversations.
        alloc = budget_left // (n_conv - idx)
        alloc = min(alloc, len(qs))
        per_conv_limits.append(alloc)
        budget_left -= alloc

    all_seed_rows: list[dict] = []
    all_recall_rows: list[dict] = []

    # Persist the FULL evidence trail for this run — step-persistence (full
    # turn history), the turn index (per-turn one-liners + key_terms), and the
    # memory markdown — under a run-scoped dir alongside the report. Scores
    # without the underlying data are unfalsifiable: this lets us audit *why*
    # the model retrieved (or failed to retrieve) after the fact, instead of
    # discarding it in a tempdir (the original flaw — see 2026-08-11).
    # Clean up old artifacts with `rm -rf tests/results/locomo-*/`; keep recent
    # ones for audit while iterating.
    artifact_dir = RESULTS_DIR / session
    artifact_dir.mkdir(parents=True, exist_ok=True)
    store = FileStepStore(directory=artifact_dir / "store")
    index_dir = artifact_dir / "index"
    agent = _build_agent(store, index_dir)
    # Write the agent spec once so the artifact dir is self-describing.
    (artifact_dir / "meta.json").write_text(json.dumps({
        "subject": SUBJECT_MODEL,
        "data": str(DATA),
        "eviction": {
            "target_tokens": LOCOMO_TARGET_TOKENS,
            "sw_max_messages": LOCOMO_SW_MAX_MESSAGES,
            "sw_keep": LOCOMO_SW_KEEP,
        },
        "max_lines_index_tail": MAX_LINES,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))

    for (conv, pool), limit in zip(per_conv_pools, per_conv_limits):
        cid = f"locomo-{uuid.uuid4().hex[:8]}"
        seed = _seed_turns(conv)
        questions = pool[:limit] if limit else []

        # ---- Seed: replay session_1 turns for this conversation ---------
        id_to_text: dict[str, str] = {}
        for i, turn in enumerate(seed, start=1):
            prompt = f"{turn['speaker']}: {turn['text']}{_SEED_INSTRUCTION}"
            row = _run_turn(
                agent, prompt, cid,
                f"locomo.seed.{i}.session_1", session,
            )
            row["dia_id"] = turn["dia_id"]
            id_to_text[turn["dia_id"]] = f"{turn['speaker']}: {turn['text']}"
            row["conv_id"] = cid
            all_seed_rows.append(row)

        # ---- Recall: cold-start the answerable questions ---------------
        for qi, q in enumerate(questions, start=1):
            row = _run_turn(
                agent, q["question"], cid,
                f"locomo.recall.{qi}.session_1", session,
            )
            row["question"] = q["question"]
            row["gold_answer"] = _gold_answer(q)
            row["category"] = q.get("category")
            row["gold_evidence"] = sorted(q.get("evidence") or [])
            row["conv_id"] = cid
            _score_question(row, q, id_to_text)
            all_recall_rows.append(row)

    recall_rows = all_recall_rows
    seed_rows = all_seed_rows

    # Subset the model actually had the material to answer: gold evidence
    # surfaced by retrieval (retrieval_recall == 1). "Ignore the ones it
    # can't answer" -- the fidelity-bound questions never retrieved their
    # gold turn, so we report them separately from the answerable subset.
    answerable = [r for r in recall_rows if r["retrieval_recall"] >= 1.0]

    result = {
        "meta": {
            "subject": SUBJECT_MODEL,
            "session": session,
            "n_conversations": len(per_conv_pools),
            "data": str(DATA),
            "conv_limits": {i: l for i, l in enumerate(per_conv_limits)},
            "max_lines_index_tail": MAX_LINES,
            "eviction": {
                "target_tokens": LOCOMO_TARGET_TOKENS,
                "sw_max_messages": LOCOMO_SW_MAX_MESSAGES,
                "sw_keep": LOCOMO_SW_KEEP,
            },
            "traced": traced,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "seed": {
            "n_turns": len(seed_rows),
            "tokens": {
                "input": sum(r["usage"]["input_tokens"] for r in seed_rows),
                "output": sum(r["usage"]["output_tokens"] for r in seed_rows),
                "total": sum(r["usage"]["total_tokens"] for r in seed_rows),
            },
        },
        "recall": {
            "n_questions": len(recall_rows),
            "questions": [
                {
                    "question": r["question"],
                    "conv_id": r["conv_id"],
                    "gold_answer": r["gold_answer"],
                    "category": r["category"],
                    "answer": r["text"],
                    "tools": r["tools"],
                    "tool_outputs": r["tool_outputs"],
                    "gold_evidence": r["gold_evidence"],
                    "answer_f1": r["answer_f1"],
                    "retrieval_recall": r["retrieval_recall"],
                    "retrieval_surfaced": r["retrieval_surfaced"],
                    "tokens": r["usage"],
                }
                for r in recall_rows
            ],
            "answer_f1_mean": round(
                sum(r["answer_f1"] for r in recall_rows) / len(recall_rows), 4
            ) if recall_rows else None,
            "retrieval_recall_rate": round(
                sum(r["retrieval_recall"] for r in recall_rows) / len(recall_rows), 4
            ) if recall_rows else None,
            # The number that matters: F1 on questions whose gold evidence
            # was actually retrieved (the fidelity-fair subset).
            "answerable": {
                "n": len(answerable),
                "answer_f1_mean": round(
                    sum(r["answer_f1"] for r in answerable) / len(answerable), 4
                ) if answerable else None,
            },
            "tokens": {
                "input": sum(r["usage"]["input_tokens"] for r in recall_rows),
                "output": sum(r["usage"]["output_tokens"] for r in recall_rows),
                "total": sum(r["usage"]["total_tokens"] for r in recall_rows),
            },
        },
    }

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"locomo-{ts}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n[written] {out}")
    if traced:
        shutdown_observability()  # flush buffered OTel spans before process exit
    return result


if __name__ == "__main__":
    main()
