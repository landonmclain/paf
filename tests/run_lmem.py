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

SCORING (two columns, per LongMemEval two-metric design):
  1. Judge      -- OFFICIAL LongMemEval binary yes/no: `get_anscheck_prompt` in
                    tests/locomo_scoring.py (VERBATIM per-task templates, incl. the
                    abstention prompt), judged by gpt-4o-mini via OpenRouter at
                    temperature 0; label = 'yes' in response. Gold `answer` is the
                    reference. Reports per-task / task-averaged / overall /
                    abstention accuracy, mirroring print_qa_metrics.py.
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
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    RunRecord,
)

from paf.capabilities.turn_index import (
    Turn,
    TurnIndexStore,
    TurnSummarizer,
    TurnSummary,
    summarize_turn_append,
)
from paf.capabilities.turn_index._extract import build_turn_context
from paf.observability.instrumentation import (
    _json_dumps,
    root_span,
    setup_observability,
    shutdown_observability,
)

# Official LongMemEval judge scaffold (verbatim prompts + label parsing).
from tests.locomo_scoring import (
    LME_TYPES,
    get_lme_anscheck_prompt,
    lme_is_abstention,
    lme_parse_label,
)

# Reuse the LoCoMo harness plumbing (agent assembly + turn runner).
from tests.run_locomo import (
    _build_agent,
    _run_turn,
)

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
# Durable seed summaries (committed, see snapshot_seed_index.py). The harness
# REQUIRES these per-question; missing = error, no silent regeneration.
GENERATED_DIR = Path(__file__).parent.parent / "data" / "benchmarks" / "generated"

# Multi-question run bounds (small by default). A stratified ~15-Q sample
# across all 6 types + some abstention is a good first validation.
LME_MAX_QUESTIONS = int(os.environ.get("LME_MAX_QUESTIONS", "15"))
LME_MAX_TURNS_PER_QUESTION = int(os.environ.get("LME_MAX_TURNS_PER_QUESTION", "120"))  # cap seeding cost


def _run_sync_under_span(agent, prompt: str, span_name: str, session: str):
    """Run a sync agent call inside a root span (matches the seed/recall span convention)."""
    with root_span(
        span_name,
        **{
            "langfuse.observation.input": _json_dumps({"prompt": prompt}),
            "langfuse.session.id": session,
        },
    ):
        return agent.run_sync(prompt)


def _lme_judge(question_type: str, question: str, gold_answer: str, response: str, qid: str, span_name: str = "lmem.judge", session: str = "") -> dict:
    """Official LongMemEval binary judge (gpt-4o-mini): yes/no on the ANSCHECK prompt.

    Uses the VERBATIM per-task prompt from evaluate_qa.py (`get_lme_anscheck_prompt`),
    adapted to the paf Agent (judge != answerer). Structured output asks the judge
    for an explicit `reasoning` alongside the verdict, so we can audit *why* each
    question was marked yes/no -- not just the label. `label` is kept a bool so the
    official aggregation (`autoeval_label`, `_acc` sum) is unchanged.
    The judge run is wrapped in a `root_span` named `span_name` (default
    `lmem.judge`) so its Langfuse trace name matches the seed/recall convention
    (`lmem.seed.N.M` / `lmem.recall.N` / `lmem.judge.N`) instead of defaulting to
    the agent name `lme-judge run`.
    Returns {label, correct, raw, reasoning, prompt}.
    """
    from typing import Literal

    from pydantic import BaseModel

    class _JudgeVerdict(BaseModel):
        label: Literal["yes", "no"]
        reasoning: str

    abstention = lme_is_abstention(qid)
    prompt = get_lme_anscheck_prompt(
        question_type, question, gold_answer, response, abstention=abstention
    )
    judge = Agent(
        JUDGE_MODEL,
        name="lme-judge",
        output_type=_JudgeVerdict,
        instructions=(
            "You are the official LongMemEval answer judge. Answer the question "
            "exactly as asked: with a structured verdict `label` of 'yes' or 'no' "
            "plus a concise `reasoning` explaining why the response matches (or "
            "fails to match) the gold answer per the task rubric. `label` must be "
            "exactly 'yes' or 'no'."
        ),
    )
    result = judge.run_sync(prompt) if not session else _run_sync_under_span(judge, prompt, span_name, session)
    verdict = result.output
    raw = verdict.label if verdict is not None else ""
    label = lme_parse_label(raw)
    reasoning = getattr(verdict, "reasoning", "") if verdict is not None else ""
    if verdict is None:
        reasoning = "ERROR: judge returned no structured verdict (fallback to raw output)"
        raw = str(result.output or "").strip()
        label = lme_parse_label(raw)
    return {"label": label, "correct": label, "raw": raw, "reasoning": reasoning, "prompt": prompt}


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
    for sess, dstr, sid in zip(sessions, dates, session_ids, strict=True):
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
# Direct-persist seed (no agent loop): write raw LME turns straight into the
# store + turn index, then let the summarizer build the one-liners.
# ---------------------------------------------------------------------------


def _raw_turns(q: dict) -> list[dict]:
    """Raw LME turns across all haystack sessions, in chronological order.

    Each turn is the ORIGINAL `{role, content}` record (NOT flattened into a
    `user: {role}: {content}` prompt -- that flattening was the corrupt-seed
    bug: it made the store hold `User: assistant: ...` + the seed agent's own
    tool calls, and the search tools faithfully returned the garbage).
    """
    sessions = q["haystack_sessions"]
    dates = q["haystack_dates"]
    session_ids = q["haystack_session_ids"]
    tagged = []
    for sess, dstr, sid in zip(sessions, dates, session_ids, strict=True):
        tagged.append((_parse_dt(dstr), dstr, sid, sess))
    tagged.sort(key=lambda x: x[0])
    out: list[dict] = []
    for _dt, dstr, sid, sess in tagged:
        for i, turn in enumerate(sess):
            out.append(
                {
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                    "dia_id": f"{sid}_t{i}",
                    "date": dstr if i == 0 else "",
                }
            )
    return out


def _transcript_messages(
    raw_turns: list[dict], run_id: str, cid: str
) -> list[ModelMessage]:
    """Build pydantic-ai ModelMessages with TRUE roles (user/assistant).

    Each LME turn becomes a `ModelRequest` (user) or `ModelResponse`
    (assistant) so the persisted history is a faithful transcript -- no
    `User: assistant:` nesting, no tool-call pollution.
    """
    messages: list[ModelMessage] = []
    for t in raw_turns:
        text = t["content"]
        if t["role"] == "assistant":
            messages.append(
                ModelResponse(
                    parts=[TextPart(content=text)],
                    run_id=run_id,
                    conversation_id=cid,
                )
            )
        else:
            messages.append(
                ModelRequest(
                    parts=[UserPromptPart(content=text)],
                    run_id=run_id,
                    conversation_id=cid,
                )
            )
    return messages


def _user_turn_units(messages: list[ModelMessage]) -> list[tuple[int, list[ModelMessage]]]:
    """Group transcript messages into user-turn units: (turn_id, [msgs]).

    `turn_id` = ordinal of the user-prompt anchor among the run's user
    prompts -- exactly the ref model `read_turn('{run_id}:{turn_id}')` walks
    (`_TurnFinder._render_turn` slices from the turn_id-th anchor to the next
    user anchor). A unit is [user ModelRequest, *following assistant
    ModelResponses]. A leading assistant message with no preceding user
    anchor opens the first unit so no transcript content is dropped.
    """
    units: list[tuple[int, list[ModelMessage]]] = []
    pending: list[ModelMessage] = []
    for m in messages:
        if isinstance(m, ModelRequest):
            units.append((len(units), [m] + pending))
            pending = []
        elif units:
            units[-1][1].append(m)
        else:
            pending.append(m)  # leading assistant message (defensive)
    if pending and not units:
        units.append((0, pending))
    return units


async def _persist_seed_direct(
    store: FileStepStore,
    index_store: TurnIndexStore,
    summarizer: TurnSummarizer | None,
    raw_turns: list[dict],
    cid: str,
    run_id: str,
    *,
    precomputed: dict[int, TurnSummary] | None = None,
) -> list[dict]:
    """Persist raw LME turns directly (no agent) and index each user turn.

    Writes ONE run record + ONE snapshot holding the full transcript (so
    `read_turn` / `search_conversation_history` resolve), then summarizes
    each user-turn unit with the REAL summarizer into the turn index. No LLM
    chat agent runs during seed -- the only model calls are the summarizer's
    (the index builder), which is the whole point: ~zero seed token burn and
    a CLEAN store (no `User: assistant:` nesting, no tool-call pollution).

    `precomputed` is the REUSE seam: when a turn_id has a durable `TurnSummary`
    in it, that summary is used instead of calling the summarizer (so re-runs
    reproduce the exact same index with ZERO summarizer tokens). When a turn_id
    is absent from `precomputed`, the summarizer is called as usual -- keeping
    this function the single mechanics path for the gate (`check_direct_seed`)
    and the live-summarizer validation (`tiny_lmem`).
    """
    messages = _transcript_messages(raw_turns, run_id, cid)
    await store.register_run(
        RunRecord(run_id=run_id, conversation_id=cid, agent_name="locomo-subject")
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id=run_id,
            step_index=0,
            conversation_id=cid,
            agent_name="locomo-subject",
            messages=messages,
        )
    )

    rows: list[dict] = []
    units = _user_turn_units(messages)
    missing = sorted(tid for tid, _ in units if tid not in (precomputed or {}))
    if missing and summarizer is None:
        # No durable summary AND no live summarizer to build one -> cannot
        # index these turns. This is the STRICT guard: when a live summarizer
        # is available (fallback path), missing turns are summarized on the fly
        # instead so new questions can be indexed without a committed snapshot.
        raise ValueError(
            f"stale durable seed summaries: no summary for turn_id(s) {missing} "
            "and no live summarizer. Regenerate with snapshot_seed_index.py, "
            "or pass a summarizer to build them live."
        )
    if missing and summarizer is not None:
        print(
            f"[lmem]   _persist_seed_direct: {len(missing)} turn(s) missing "
            f"durable summary -> summarized live: {missing[:5]}{'...' if len(missing) > 5 else ''}",
            flush=True,
        )
    for turn_id, msgs in units:
        turn = Turn(
            run_id=run_id,
            conversation_id=cid,
            turn_id=turn_id,
            messages=msgs,
        )
        rendered = build_turn_context(turn.messages)
        if (precomputed or {}).get(turn_id) is not None:
            summary = (precomputed or {})[turn_id]
        elif summarizer is None:
            raise RuntimeError(  # unreachable: both guards above cover this
                "_persist_seed_direct: summarizer is None with no precomputed summary"
            )
        else:
            summary = await summarizer.summarize(rendered)
        appended, one_line = await summarize_turn_append(
            turn=turn, index_store=index_store, summary=summary
        )
        rows.append(
            {
                "text": one_line,
                "tools": [],
                "tool_outputs": [],
                "usage": {
                    "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                    "cache_write_tokens": 0, "total_tokens": 0, "requests": 0,
                },
                "dia_id": turn.ref,
                "conv_id": cid,
                "direct": True,
                "appended": appended,
            }
        )
    return rows


def _load_precomputed_summaries(
    question_id: str, *, strict: bool = True
) -> dict[int, TurnSummary]:
    """Load durable seed summaries for `question_id`.

    Reads `GENERATED_DIR/<question_id>.jsonl` (committed via
    `snapshot_seed_index.py`). In STRICT mode (default) a missing file raises
    -- there is NO silent fallback to regenerating: the durable summaries are
    the source of truth for a reproducible run. With `strict=False` (new
    questions that don't have durable summaries yet) a missing file returns
    `{}` so `_persist_seed_direct` falls through to its live summarizer --
    the index is built on the fly that run and can be snapshotted afterward.
    Stale/partial coverage is caught by `_persist_seed_direct` (every
    user-turn unit must have a summary, or it's summarized live).
    """
    path = GENERATED_DIR / f"{question_id}.jsonl"
    if not path.exists():
        if strict:
            raise FileNotFoundError(
                f"durable seed summaries missing: {path}. Run "
                "`tests/snapshot_seed_index.py` to generate them for this question."
            )
        return {}
    out: dict[int, TurnSummary] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[int(rec["turn_id"])] = TurnSummary(
            sentence=rec["summary"],
            key_terms=rec.get("key_terms") or [],
            outcome=rec["outcome"],
        )
    return out


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

    traced = setup_observability()
    if not traced:
        print("[run_lmem] WARNING: Langfuse tracing NOT configured — run will "
              "produce JSON/artifacts but NO Langfuse traces.")
    else:
        print(f"[run_lmem] tracing ON → Langfuse (session={session})", flush=True)
    print(f"[run_lmem] questions={len(questions)} subject={SUBJECT_MODEL} "
          f"judge={JUDGE_MODEL} max_lines={MAX_LINES} "
          f"target_tokens={LOCOMO_TARGET_TOKENS} "
          f"real_summarizer=TurnSummarizer@{SUBJECT_MODEL}", flush=True)

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
    agent = _build_agent(store, index_dir)
    # Direct-seed writes (no agent loop) share the SAME store + index the
    # recall agent reads: the index store and real summarizer are constructed
    # here so seed rows land exactly where read_turn/search_index will look.
    index_store = TurnIndexStore(index_dir)
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
        raw_turns = _raw_turns(q)
        # Cap seeding cost (LongMemEval oracle is small, but be safe).
        raw_turns = raw_turns[:LME_MAX_TURNS_PER_QUESTION]
        print(
            f"[lmem] q{qi}/{len(questions)} {q['question_type']} "
            f"{q.get('question_id', '?')} seeding {len(raw_turns)} turns — "
            f"recall: {q['question'][:48]!r}",
            flush=True,
        )

        # ---- Seed the haystack sessions ------------------------------
        # DIRECT-PERSIST SEED (no agent loop): raw LME turns are written
        # straight into the store + turn index with their TRUE roles. The
        # index summaries come from the DURABLE committed store
        # (`data/benchmarks/generated/`, see snapshot_seed_index.py), STRICT:
        # a missing/stale file is a hard error -- no silent regeneration, so
        # re-runs reproduce the exact same index with ZERO summarizer tokens.
        # The store still persists the full transcript for read_turn /
        # search_conversation_history resolution.
        run_id = f"seed-{cid}"
        # Durable summaries where they exist (strict), else FALL BACK to a
        # live summarizer for the index. New questions (5+) have no committed
        # summaries yet, so build their index on the fly this run, then
        # snapshot it afterward (snapshot_seed_index.py) so next time it's
        # durable and strictly reused. Q1-4 (committed) stay zero-token.
        precomputed = _load_precomputed_summaries(
            q["question_id"], strict=False
        )
        live_summarizer: TurnSummarizer | None = None
        if not precomputed:
            live_summarizer = TurnSummarizer(
                model=SUBJECT_MODEL, budget_chars=160, max_key_terms=6
            )
            print(
                f"[lmem]   q{qi} NO durable summaries -> live summarizer "
                f"(TurnSummarizer@{SUBJECT_MODEL})",
                flush=True,
            )
        seed_rows = asyncio.run(
            _persist_seed_direct(
                store, index_store, live_summarizer, raw_turns, cid, run_id,
                precomputed=precomputed,
            )
        )
        all_seed_rows.extend(seed_rows)
        print(
            f"[lmem]   q{qi} seeded {len(seed_rows)} user-turn units "
            f"({'durable, 0 summarizer' if not live_summarizer else 'LIVE summarized'})",
            flush=True,
        )

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
        row["answer_session_ids"] = q.get("answer_session_ids") or []
        row["question_id"] = q.get("question_id", f"q{qi}")
        row["conv_id"] = cid

        # Judge: official LongMemEval binary (yes/no ANSCHECK prompt).
        row["judged"] = _lme_judge(
            q["question_type"], q["question"], q["answer"], row["text"], q.get("question_id", f"q{qi}"),
            span_name=f"lmem.judge.{qi}", session=session,
        )
        row["autoeval_label"] = {"model": JUDGE_MODEL, "label": row["judged"]["label"]}

        # Retrieval signal (independent of judge).
        row["retrieval"] = _retrieval(row, _gold_session_probes(q))

        all_recall_rows.append(row)
        print(
            f"[lmem] q{qi}/{len(questions)} DONE type={q['question_type']} "
            f"judge={'CORRECT' if row['judged'].get('label') else 'wrong'} "
            f"retrieved={row['retrieval'].get('tool_surfaced', '?')} "
            f"({row['text'][:60]!r})",
            flush=True,
        )

    recall_rows = all_recall_rows
    seed_rows = all_seed_rows

    # Abstention questions: question_id ends with _abs. Correct behaviour is to
    # REFUSE (the official abstention prompt grades this). We keep them separate
    # from factual questions, and report Abstention Accuracy independently --
    # exactly as print_qa_metrics.py does (abstention is sliced out of "all_acc").
    abstention_rows = [r for r in recall_rows if lme_is_abstention(r.get("question_id", ""))]
    factual_rows = [r for r in recall_rows if not lme_is_abstention(r.get("question_id", ""))]

    # ---- Official LongMemEval aggregation (mirrors print_qa_metrics.py) ----
    # Per-task accuracy is the mean of the BINARY labels within each question type;
    # abtention is sliced OUT of all_acc and reported alone. Then:
    #   Overall Accuracy    = mean over ALL non-abstention questions
    #   Task-averaged Acc   = mean of the 6 per-task means
    #   Abstention Accuracy = mean over abstention questions
    def _acc(rows: list[dict]) -> float | None:
        if not rows:
            return None
        return round(sum(r["autoeval_label"]["label"] for r in rows) / len(rows), 4)

    per_task = {t: _acc([r for r in factual_rows if r["question_type"] == t]) for t in LME_TYPES}
    deline_by_task: list[float] = [v for v in (per_task[t] for t in LME_TYPES) if v is not None]
    task_averaged = round(sum(deline_by_task) / len(deline_by_task), 4) if deline_by_task else None
    overall = _acc(factual_rows)
    abstention = _acc(abstention_rows)

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
                    "judged": r["judged"],
                    "autoeval_label": r["autoeval_label"],
                    "retrieval": r["retrieval"],
                    "tokens": r["usage"],
                }
                for r in recall_rows
            ],
            # Official metrics: per-task accuracy, task-averaged, overall, abstention.
            "per_task_accuracy": per_task,
            "task_averaged_accuracy": task_averaged,
            "overall_accuracy": overall,
            "abstention_accuracy": abstention,
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
    if traced:
        shutdown_observability()  # flush buffered OTel spans before process exit
    return result


if __name__ == "__main__":
    main()
