"""Isolated single-question run of the GM rotation question (7161e7e2) with MEMORY OFF.

Builds the same full PAF pipeline as run_lmem._build_agent EXCEPT the Memory
capability is removed (no inject_memory, no <memory> block, no read_memory /
write_memory tools). Everything else (StepPersistence, ConversationSearch,
TurnIndex + real TurnSummarizer, TieredCompaction) is identical.

Purpose: isolate whether the durable-memory injection is what feeds the model
the "Backup/Relief, Admon=B on Sunday" rotation that contradicts the gold
(Admon = 8am-4pm Day Shift Sunday). With memory off, the model can only see the
seeded conversation + turn index + search tools.

Usage:
    PYTHONPATH="" .venv/bin/python tests/run_lmem_memoff.py
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from pydantic_ai import Agent
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

from tests.run_locomo import _run_turn
from tests.run_lmem import (
    _lme_judge,
    _retrieval,
    _seed_turns,
    _gold_session_probes,
)

SUBJECT_MODEL = os.environ.get("BEAM_LITE_SUBJECT_MODEL", "openrouter:deepseek/deepseek-v4-flash-0731:nitro")
JUDGE_MODEL = os.environ.get("LME_JUDGE_MODEL", "openrouter:openai/gpt-4o-mini")
MAX_LINES = int(os.environ.get("LOCOMO_MAX_LINES", "10"))
LOCOMO_TARGET_TOKENS = int(os.environ.get("LOCOMO_TARGET_TOKENS", "5000"))
LOCOMO_SW_MAX_MESSAGES = int(os.environ.get("LOCOMO_SW_MAX_MESSAGES", "6"))
LOCOMO_SW_KEEP = int(os.environ.get("LOCOMO_SW_KEEP", "3"))
DATA = Path(__file__).parent.parent / "data" / "benchmarks" / "lme_oracle.json"
RESULTS_DIR = Path(__file__).parent / "results"

TARGET_QID = "7161e7e2"


def _build_agent_memoff(store: FileStepStore, index_dir: Path) -> Agent:
    """Same as run_locomo._build_agent but WITHOUT the Memory capability."""
    source = SnapshotHistorySource(store)
    caps = [
        StepPersistence(store=store),
        ConversationSearch(source=source, scope="conversation", add_instructions=False),
        TurnIndexCapability(
            source=source,
            index_store=TurnIndexStore(index_dir),
            max_lines=MAX_LINES,
            add_instructions=False,
            summarizer=TurnSummarizer(model=SUBJECT_MODEL, budget_chars=160, max_key_terms=6),
        ),
        TieredCompaction(
            target_tokens=LOCOMO_TARGET_TOKENS,
            tiers=[
                ClampOversizedMessages(max_part_tokens=50_000, keep_head_chars=2_000, keep_tail_chars=2_000),
                ClearToolResults(max_tokens=1, keep_pairs=4),
                SlidingWindowCompaction(max_messages=LOCOMO_SW_MAX_MESSAGES, keep_messages=LOCOMO_SW_KEEP),
            ],
        ),
    ]
    return Agent(SUBJECT_MODEL, name="locomo-subject", capabilities=caps)


def main() -> dict:
    data = json.loads(DATA.read_text())
    q = next((x for x in data if str(x.get("question_id")) == TARGET_QID), None)
    if q is None:
        raise RuntimeError(f"question {TARGET_QID} not found in {DATA}")

    session = f"lmem-memoff-{time.strftime('%Y%m%d-%H%M%S')}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    traced = setup_observability()
    print(f"[memoff] tracing={'ON' if traced else 'OFF'} session={session}", flush=True)
    print(f"[memoff] question {TARGET_QID} type={q['question_type']} date={q.get('question_date')}", flush=True)

    artifact_dir = RESULTS_DIR / session
    artifact_dir.mkdir(parents=True, exist_ok=True)
    store = FileStepStore(directory=artifact_dir / "store")
    index_dir = artifact_dir / "index"
    agent = _build_agent_memoff(store, index_dir)

    cid = f"lmem-{uuid.uuid4().hex[:8]}"
    turns, answer_sids = _seed_turns(q)
    print(f"[memoff] seeding {len(turns)} turns into conv {cid}", flush=True)

    seed_rows = []
    for i, turn in enumerate(turns, start=1):
        row = _run_turn(agent, turn["text"], cid, f"lmem.seed.1.{i}", session)
        row["dia_id"] = turn["dia_id"]
        seed_rows.append(row)
        if i % 5 == 0 or i == len(turns):
            print(f"[memoff]   seeded {i}/{len(turns)}", flush=True)

    qctx = f"The date now is {q['question_date']}. " if q.get("question_date") else ""
    row = _run_turn(agent, qctx + q["question"], cid, "lmem.recall.1", session)
    row["question"] = q["question"]
    row["question_type"] = q["question_type"]
    row["gold_answer"] = q["answer"]
    row["question_date"] = q.get("question_date")
    row["answer_session_ids"] = answer_sids
    row["question_id"] = q.get("question_id", TARGET_QID)
    row["conv_id"] = cid

    row["judged"] = _lme_judge(
        q["question_type"], q["question"], q["answer"], row["text"],
        q.get("question_id", TARGET_QID), span_name="lmem.judge.1", session=session,
    )
    row["autoeval_label"] = {"model": JUDGE_MODEL, "label": row["judged"]["label"]}
    row["retrieval"] = _retrieval(row, _gold_session_probes(q))

    print(f"[memoff] RECALL tools={row['tools']}", flush=True)
    print(f"[memoff] RECALL answer: {row['text'][:400]}", flush=True)
    print(f"[memoff] judge={'CORRECT' if row['judged']['label'] else 'WRONG'} "
          f"reasoning={row['judged'].get('reasoning','')[:200]}", flush=True)
    print(f"[memoff] retrieval={row['retrieval']}", flush=True)

    result = {
        "meta": {"session": session, "question_id": TARGET_QID, "memory": "OFF",
                 "subject": SUBJECT_MODEL, "judge": JUDGE_MODEL, "traced": traced},
        "recall": row,
        "seed_n_turns": len(seed_rows),
    }
    out = RESULTS_DIR / f"lmem-memoff-{TARGET_QID}.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"[memoff] wrote {out}", flush=True)
    shutdown_observability()
    return result


if __name__ == "__main__":
    main()
