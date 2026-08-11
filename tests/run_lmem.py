"""LongMemEval sample run — full PAF memory pipeline.

Analogue of `run_locomo.py`, but for LongMemEval's `lme_oracle.json` (the
"oracle" config = evidence-bearing sessions only, ~1-6 sessions / ~24 msgs per
question). Each question's `haystack_sessions` are seeded into ONE isolated
store through the FULL enabled PAF pipeline, then the question is asked
cold-start and scored.

Pipeline (mirrors run_locomo.py / ASSISTANT_PROFILE, one isolated temp store):

  - StepPersistence          full history to a FileStepStore
  - ConversationSearch       BM25 wide fallback over that store
  - TurnIndexCapability      injected tail index + read_turn / search_index
  - Memory (markdown)        durable notebook, inject_memory=True
  - TieredCompaction         the no-summary ladder (target_tokens default 5000)
                             -- prunes live context so old sessions must be
                             retrieved, not recognised in-context

DATE HANDLING (the thing LoCoMo couldn't do):
  LongMemEval sessions are dated (`haystack_dates`) and the question has a
  `question_date` anchor. The oracle file's sessions are NOT date-sorted, so we
  sort them by `haystack_dates` and prefix each session's turns with its
  datetime in the seed prompt. This gives temporal-reasoning / knowledge-update
  questions a real date anchor to resolve "yesterday"/"last week"/recency
  against -- the exact capture-fidelity gap that capped LoCoMo's answer-F1.

SCORING (two columns, per LoCoMo two-metric design):
  1. Judge      -- reuse the C1-C4 stable LLM judge (_judge in test_beam_lite).
                    Judge model defaults to gpt-4o-mini via OpenRouter (Landon's
                    call: no new routing, reuse the repo OPENROUTER_API_KEY).
                    Gold `answer` is the single ground-truth atom.
  2. Retrieval  -- independent of the judge: did the search tools surface the
                    content of the `answer_session_ids`/`has_answer` turns?

Outputs JSON to tests/results/lmem-<ts>.json.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent

# Reuse the C1-C4 stable LLM judge (judge != answerer). Judge model override via
# env so we don't touch test_beam_lite's default for its own runs.
from tests.test_beam_lite import _judge

# Reuse the LoCoMo harness plumbing (agent assembly + turn runner).
from tests.run_locomo import (
    _build_agent,
    _run_turn,
)

from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    SlidingWindowCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence

from paf.capabilities.memory import Memory
from paf.capabilities.search import ConversationSearch, SnapshotHistorySource
from paf.capabilities.turn_index import TurnIndexCapability, TurnIndexStore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUBJECT_MODEL = os.environ.get("BEAM_LITE_SUBJECT_MODEL", "openrouter:deepseek/deepseek-v4-flash-0731:nitro")
# Landon's call: gpt-4o-mini on OpenRouter, reuse the repo OPENROUTER_API_KEY.
JUDGE_MODEL = os.environ.get("LME_JUDGE_MODEL", "openrouter:openai/gpt-4o-mini")

MAX_LINES = int(os.environ.get("LOCOMO_MAX_LINES", "10"))
# Same middle dial as the LoCoMo sweep finding (see run_locomo.py docstring):
# wide enough to keep ordinary exchanges in-context, tight enough that old
# evidence sessions are evicted -> retrieval pressure stays ON.
LOCOMO_TARGET_TOKENS = int(os.environ.get("LOCOMO_TARGET_TOKENS", "5000"))
LOCOMO_SW_MAX_MESSAGES = int(os.environ.get("LOCOMO_SW_MAX_MESSAGES", "6"))
LOCOMO_SW_KEEP = int(os.environ.get("LOCOMO_SW_KEEP", "3"))

DATA = Path(__file__).parent.parent / "data" / "benchmarks" / "lme_oracle.json"
RESULTS_DIR = Path(__file__).parent / "results"

# Multi-question run bounds (small by default). A stratified ~15-Q sample
# across all 6 types + some abstention is a good first validation.
LME_MAX_QUESTIONS = int(os.environ.get("LME_MAX_QUESTIONS", "15"))
LME_MAX_TURNS_PER_QUESTION = int(os.environ.get("LME_MAX_TURNS_PER_QUESTION", "120"))  # cap seeding cost


@dataclass
class _Challenge:
    """Minimal C1-C4 Challenge-shaped wrapper for _judge's interface."""

    id: str
    mechanism: str
    query: str
    ground_truth_atoms: tuple[str, ...]
    expected_tool: str = "none"


# ---------------------------------------------------------------------------
# Sampling / date handling
# ---------------------------------------------------------------------------


def _load_questions(n: int | None = None) -> list[dict]:
    """Return a stratified sample of LongMemEval questions (across all types).

    Reserves a small number of slots for ABSTENTION questions (question_id
    ends `_abs`) so the harness exercises the abstention-judging path, then
    fills the rest via round-robin across `question_type`.
    """
    data = json.loads(DATA.read_text())
    if n is None:
        n = LME_MAX_QUESTIONS
    abstention_pool = [q for q in data if str(q["question_id"]).endswith("_abs")]
    factual = [q for q in data if not str(q["question_id"]).endswith("_abs")]
    n_abs = min(2, max(0, n // 5), len(abstention_pool))
    sample: list[dict] = list(abstention_pool[:n_abs])
    n_fact = n - n_abs

    # Stratified round-robin across question_type so every class is represented.
    by_type: dict[str, list[dict]] = {}
    for q in factual:
        by_type.setdefault(q["question_type"], []).append(q)
    types = sorted(by_type)
    idx = {t: 0 for t in types}
    while len(sample) < n and len(sample) < n_abs + n_fact:
        progressed = False
        for t in types:
            if len(sample) >= n_abs + n_fact:
                break
            pool = by_type[t]
            if idx[t] < len(pool):
                sample.append(pool[idx[t]])
                idx[t] += 1
                progressed = True
        if not progressed:
            break
    return sample


def _parse_dt(s: str) -> tuple:
    """Parse '2023/04/10 (Mon) 23:07' into a sortable tuple."""
    # 2023/04/10 (Mon) 23:07
    date_part, _, time_part = s.partition("(")
    y, m, d = (int(x) for x in date_part.strip().split("/"))
    hh, mm = (0, 0)
    if time_part.strip():
        try:
            hh, mm = (int(x) for x in time_part.strip().split(":"))
        except ValueError:
            pass
    return (y, m, d, hh, mm)


def _seed_turns(q: dict) -> tuple[list[dict], list[str]]:
    """Return datestamped seed turns (sorted by date) + their session id per turn.

    Each turn becomes {text, dia_id}. We prefix the session's datetime onto the
    first turn of each session so the model sees date context that resolves
    relative-time references (the LoCoMo gap). dia_id = f"{session}_t{i}".
    """
    sessions = q["haystack_sessions"]
    dates = q["haystack_dates"]
    session_ids = q["haystack_session_ids"]
    # zip sessions with (date, session_id) and sort by parsed date.
    tagged = []
    for sess, dstr, sid in zip(sessions, dates, session_ids):
        tagged.append((_parse_dt(dstr), dstr, sid, sess))
    tagged.sort(key=lambda x: x[0])
    turns: list[dict] = []
    for _dt, dstr, sid, sess in tagged:
        for i, turn in enumerate(sess):
            content = turn.get("content", "")
            prefix = f"[date: {dstr}] " if i == 0 else ""
            turns.append(
                {
                    "text": f"{prefix}{turn.get('role', 'user')}: {content}",
                    "dia_id": f"{sid}_t{i}",
                }
            )
    return turns, q.get("answer_session_ids") or []


# ---------------------------------------------------------------------------
# Retrieval signal (independent of the judge)
# ---------------------------------------------------------------------------


def _retrieval(recall: dict, gold_session_texts: list[str]) -> dict:
    """Did the retrieval tools surface the answer sessions' content?

    A gold session counts as surfaced if its has_answer turn's *content*
    reaches the model — either in a tool output or reproduced in the final
    answer. We use token-overlap F1 (reuse locomo_scoring) rather than a raw
    first-80-chars substring, because LongMemEval's answer often sits near the
    END of a long evidence turn (e.g. "...I set a new PB of 25:50"). A probe
    scores surfaced if F1(probe, any_tool_output) >= 0.30 OR
    F1(probe, answer) >= 0.35 (same thresholds as the LoCoMo tool_surfaced).
    """
    from tests.locomo_scoring import f1_score

    if not gold_session_texts:
        return {"retrieval_recall": 1.0, "surfaced": []}
    chunks = [t for t in recall.get("tool_outputs", []) if t and t.strip()]
    ans = (recall.get("text") or "").strip()
    surfaced = []
    for probe in gold_session_texts:
        p = probe.strip()
        if not p:
            continue
        best_tool = max((f1_score(p, c) for c in chunks), default=0.0)
        ans_f1 = f1_score(p, ans) if ans else 0.0
        if best_tool >= 0.30 or ans_f1 >= 0.35:
            surfaced.append(p)
    return {
        "retrieval_recall": round(len(surfaced) / len(gold_session_texts), 4),
        "surfaced": surfaced,
    }


def _gold_session_probes(q: dict) -> list[str]:
    """Distinctive text per answer session (has_answer turn, else first user turn)."""
    out = []
    sessions = q["haystack_sessions"]
    for sess in sessions:
        for turn in sess:
            if turn.get("has_answer") and turn.get("content"):
                out.append(turn["content"])
                break
        else:
            # no has_answer flag in this session -> first user turn
            for turn in sess:
                if turn.get("role") == "user" and turn.get("content"):
                    out.append(turn["content"])
                    break
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    questions = _load_questions()
    if not questions:
        raise RuntimeError("No questions loaded from LongMemEval data")

    session = f"lmem-{time.strftime('%Y%m%d-%H%M%S')}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_seed_rows: list[dict] = []
    all_recall_rows: list[dict] = []

    # Persist the FULL evidence trail for this run — step-persistence (full
    # turn history), the turn index (per-turn one-liners + key_terms), and the
    # memory markdown — under a run-scoped dir alongside the report. Scores
    # without the underlying data are unfalsifiable: this lets us audit *why*
    # the model retrieved (or failed to retrieve) after the fact, instead of
    # discarding it in a tempdir (the original flaw — see 2026-08-11).
    # Clean up old artifacts with `rm -rf tests/results/lmem-*/`; keep recent
    # ones for audit while iterating.
    artifact_dir = RESULTS_DIR / session
    artifact_dir.mkdir(parents=True, exist_ok=True)
    store = FileStepStore(directory=artifact_dir / "store")
    index_dir = artifact_dir / "index"
    memory_dir = artifact_dir / "memory"
    agent = _build_agent(store, index_dir, memory_dir)
    # Write the agent spec once so the artifact dir is self-describing.
    (artifact_dir / "meta.json").write_text(json.dumps({
        "subject": SUBJECT_MODEL,
        "judge": JUDGE_MODEL,
        "data": str(DATA),
        "eviction": {
            "target_tokens": LOCOMO_TARGET_TOKENS,
            "sw_max_messages": LOCOMO_SW_MAX_MESSAGES,
            "sw_keep": LOCOMO_SW_KEEP,
        },
        "max_lines_index_tail": MAX_LINES,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))

    for qi, q in enumerate(questions, start=1):
        cid = f"lmem-{uuid.uuid4().hex[:8]}"
        turns, answer_sids = _seed_turns(q)
        # Cap seeding cost (LongMemEval oracle is small, but be safe).
        turns = turns[:LME_MAX_TURNS_PER_QUESTION]

        # ---- Seed the haystack sessions ------------------------------
        for i, turn in enumerate(turns, start=1):
            row = _run_turn(
                agent, turn["text"], cid,
                f"lmem.seed.{qi}.{i}", session,
            )
            row["dia_id"] = turn["dia_id"]
            row["conv_id"] = cid
            all_seed_rows.append(row)

        # ---- Recall: cold-start the question -------------------------
        qctx = f"The date now is {q['question_date']}. " if q.get("question_date") else ""
        row = _run_turn(
            agent, qctx + q["question"], cid,
            f"lmem.recall.{qi}", session,
        )
        row["question"] = q["question"]
        row["question_type"] = q["question_type"]
        row["gold_answer"] = q["answer"]
        row["question_date"] = q.get("question_date")
        row["answer_session_ids"] = answer_sids
        row["question_id"] = q.get("question_id", f"q{qi}")
        row["conv_id"] = cid

        # Judge: reuse C1-C4 _judge, gold answer as the single atom.
        ch = _Challenge(
            id=q["question_id"],
            mechanism=q["question_type"],
            query=q["question"],
            ground_truth_atoms=(q["answer"],),
        )
        datum = {
            "tools_invoked": row["tools"],
            "answer": row["text"],
            "arm": row["question_type"],
        }
        row["judged"] = _judge(ch, datum, session)
        row["judge_score"] = sum(
            a["score"] for a in row["judged"]["atoms"]
        ) / len(row["judged"]["atoms"]) if row["judged"]["atoms"] else 0.0
        row["grounding"] = row["judged"]["grounding"]

        # Retrieval signal (independent of judge).
        row["retrieval"] = _retrieval(row, _gold_session_probes(q))

        all_recall_rows.append(row)

    recall_rows = all_recall_rows
    seed_rows = all_seed_rows

    # Abstention questions: question_id ends with _abs. Correct behaviour is to
    # REFUSE. The C1-C4 judge compares vs the gold `answer` string (which for an
    # abstention q is the explanation, not a fact) -- so we report these
    # separately and don't mix them into the judge_mean for factual questions.
    # (Faithful abstention grading is a future refinement; for now it's a flag.)
    abstention_rows = [r for r in recall_rows if str(r.get("question_id", "")).endswith("_abs")]
    factual_rows = [r for r in recall_rows if r not in abstention_rows]

    result = {
        "meta": {
            "subject": SUBJECT_MODEL,
            "judge": JUDGE_MODEL,
            "session": session,
            "n_questions": len(recall_rows),
            "data": str(DATA),
            "max_lines_index_tail": MAX_LINES,
            "eviction": {
                "target_tokens": LOCOMO_TARGET_TOKENS,
                "sw_max_messages": LOCOMO_SW_MAX_MESSAGES,
                "sw_keep": LOCOMO_SW_KEEP,
            },
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
            "n_abstention": len(abstention_rows),
            "n_factual": len(factual_rows),
            "questions": [
                {
                    "question_id": r["question_id"],
                    "question": r["question"],
                    "question_type": r["question_type"],
                    "conv_id": r["conv_id"],
                    "gold_answer": r["gold_answer"],
                    "question_date": r["question_date"],
                    "answer": r["text"],
                    "tools": r["tools"],
                    "tool_outputs": r["tool_outputs"],
                    "answer_session_ids": r["answer_session_ids"],
                    "judge_score": r["judge_score"],
                    "grounding": r["grounding"],
                    "judged": r["judged"],
                    "retrieval": r["retrieval"],
                    "tokens": r["usage"],
                }
                for r in recall_rows
            ],
            # Judge mean over FACTUAL questions only (abstention handled separately).
            "judge_mean": round(
                sum(r["judge_score"] for r in factual_rows) / len(factual_rows), 4
            ) if factual_rows else None,
            "abstention_judge_mean": round(
                sum(r["judge_score"] for r in abstention_rows) / len(abstention_rows), 4
            ) if abstention_rows else None,
            "retrieval_recall_rate": round(
                sum(r["retrieval"]["retrieval_recall"] for r in recall_rows) / len(recall_rows), 4
            ) if recall_rows else None,
            "tokens": {
                "input": sum(r["usage"]["input_tokens"] for r in recall_rows),
                "output": sum(r["usage"]["output_tokens"] for r in recall_rows),
                "total": sum(r["usage"]["total_tokens"] for r in recall_rows),
            },
        },
    }

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"lmem-{ts}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n[written] {out}")
    return result


if __name__ == "__main__":
    main()