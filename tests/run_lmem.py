"""LongMemEval sample run — full PAF memory pipeline.

Uses the shared harness builder in `tests/framework.py`, pointed at LongMemEval's
`lme_oracle.json` (the
"oracle" config = evidence-bearing sessions only, ~1-6 sessions / ~24 msgs per
question). Each question's `haystack_sessions` are seeded into ONE isolated
store through the FULL enabled PAF pipeline, then the question is asked
cold-start and scored.

Pipeline (mirrors tests/framework.py / ASSISTANT_PROFILE, one isolated temp store):

  - StepPersistence          full history to a FileStepStore
  - ConversationSearch       BM25 wide fallback + read_turn over that store
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
                    tests/lme_scoring.py (VERBATIM per-task templates, incl. the
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
import re
import resource
import time
from datetime import datetime
from pathlib import Path

# LongMemEval seeds are stored UTC (``_iso_date`` stamps ``+00:00``) but
# shown in the user's local zone; a session at 2023-05-21T00:00Z would render
# as 2023-05-20 in America/Chicago and break temporal questions. Force UTC
# BEFORE any paf import resolves the zone — ``paf.clock.local_zone()`` is
# lru_cache'd on first use, so setting it here (module top) is the only point
# that reliably sticks for the whole run. Env beats .env for pydantic-settings,
# so this overrides any TIMEZONE in .env too.
os.environ["TIMEZONE"] = "UTC"

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

from paf.observability.instrumentation import (
    _json_dumps,
    root_span,
    setup_observability,
    shutdown_observability,
)
from paf.specs import load_agent_spec

# The benchmark's project slug (stamped on extraction AND forced on recall so the
# scoped render shows the lmem section even though per-question conversations are
# never registered).
from tests._lmem_consolidate import LME_PROJECT

# Reuse the LoCoMo harness plumbing (agent assembly + turn runner).
from tests.framework import (
    JUDGE_PROVIDER_ORDER,
    JUDGE_PROVIDER_QUANTIZATIONS,
    LME_SAMPLER_SEED,
    MEMORY,
    THREAD_SUMMARY,
    _build_agent,
    _openrouter_model_settings,
    _run_turn,
)

# Official LongMemEval judge scaffold (verbatim prompts + label parsing).
from tests.lme_scoring import (
    LME_TYPES,
    get_lme_anscheck_prompt,
    lme_is_abstention,
    lme_parse_label,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Landon's call: gpt-4o-mini on OpenRouter, reuse the repo OPENROUTER_API_KEY.
# The subject model comes from the spec (specs/longmemeval.yaml), not here.
JUDGE_MODEL = "openrouter:openai/gpt-4o-mini"

# Benchmark data file — v2 memory-system variant copy of the oracle split
# (evidence-only sessions, ~1.9 sessions / ~22 turns per question). Hardcoded
# (not env/spec): the renamed copy also forces a fresh durable-memory dir below,
# so the new (v2) memory system's runs never clobber the v1 snapshots that the
# 0.9468/0.934 baselines came from.
LME_DATA_FILE = "memory-v2-lme_oracle.json"
DATA = Path(__file__).parent.parent / "data" / "benchmarks" / LME_DATA_FILE
RESULTS_DIR = Path(__file__).parent / "results"

# Durable per-question MEMORY snapshots (committed, mirroring generated/ so a
# fresh clone re-runs with ZERO owner/consolidation tokens). The owner writes
# <stem>/<qid>/ at seed time; _persist_durable_memory snapshots the result here
# and _restore_durable_memory copies it back to skip re-consolidation on later
# runs. The dir is namespaced by the data-file stem (the filename already
# carries the `memory-` prefix, so no extra prefix here) so separate splits
# never clobber each other's durable memory. PAF_FRESH_MEMORY=1 forces
# re-consolidation (and re-persists); combining with LME_QIDS narrows the
# refresh to specific questions.
# Durable memory dir is namespaced by dataset filename stem (see the skill):
# copying/renaming the dataset forces a fresh memory dir. The thread-summarizer
# durable dir is namespaced separately (`-summaries`) so summary snapshots never
# collide with memory snapshots.
MEMORY_DURABLE_DIR = Path(__file__).parent.parent / "data" / "benchmarks" / DATA.stem
SUMMARIES_DURABLE_DIR = (
    Path(__file__).parent.parent / "data" / "benchmarks" / f"{DATA.stem}-summaries"
)

# Multi-question run bounds (small by default). A stratified ~15-Q sample
# across all 6 types + some abstention is a good first validation.
LME_MAX_QUESTIONS = int(os.environ.get("LME_MAX_QUESTIONS", "15"))


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


def _lme_judge(
    question_type: str,
    question: str,
    gold_answer: str,
    response: str,
    qid: str,
    span_name: str = "lmem.judge",
    session: str = "",
) -> dict:
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
        model_settings=_openrouter_model_settings(
            JUDGE_PROVIDER_ORDER,
            JUDGE_PROVIDER_QUANTIZATIONS,
            seed=LME_SAMPLER_SEED,
            base={"temperature": 0},
        ),
        instructions=(
            "You are the official LongMemEval answer judge. Answer the question "
            "exactly as asked: with a structured verdict `label` of 'yes' or 'no' "
            "plus a concise `reasoning` explaining why the response matches (or "
            "fails to match) the gold answer per the task rubric. `label` must be "
            "exactly 'yes' or 'no'."
        ),
    )
    result = (
        judge.run_sync(prompt)
        if not session
        else _run_sync_under_span(judge, prompt, span_name, session)
    )
    verdict = result.output
    raw = verdict.label if verdict is not None else ""
    label = lme_parse_label(raw)
    reasoning = getattr(verdict, "reasoning", "") if verdict is not None else ""
    if verdict is None:
        reasoning = (
            "ERROR: judge returned no structured verdict (fallback to raw output)"
        )
        raw = str(result.output or "").strip()
        label = lme_parse_label(raw)
    return {
        "label": label,
        "correct": label,
        "raw": raw,
        "reasoning": reasoning,
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# Sampling / date handling
# ---------------------------------------------------------------------------


def _load_questions(n: int | None = None) -> list[dict]:
    """Return a stratified sample of LongMemEval questions (across all types).

    Reserves a small number of slots for ABSTENTION questions (question_id
    ends `_abs`) so the harness exercises the abstention-judging path, then
    fills the rest via round-robin across `question_type`.

    `LME_QIDS` (space/comma-separated) overrides sampling entirely: run ONLY
    the named question_ids, in the given order. Useful for snapshotting a
    specific question set (e.g. `LME_QIDS="gpt4_70e84552_abs gpt4_93159ced_abs"`)
    so their durable seed summaries get persisted without a full run.
    """
    data = json.loads(DATA.read_text())
    qids = os.environ.get("LME_QIDS")
    if qids:
        want = [q.strip() for q in qids.replace(",", " ").split() if q.strip()]
        by_id = {str(q["question_id"]): q for q in data}
        out = []
        for wid in want:
            if wid in by_id:
                out.append(by_id[wid])
            else:
                print(
                    f"[run_lmem] WARNING: LME_QIDS unknown id {wid!r}, skipped",
                    flush=True,
                )
        return out
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


def _iso_date(s: str | None) -> str | None:
    """Convert an LME date ('2023/04/10 (Mon) 23:07') to ISO-8601 (''/None -> None).

    The consolidation owner's ``append_memory`` needs a ``born_on`` in ISO form
    (lexically sortable == chronologically sortable). ``_parse_dt`` drops the time
    when a weekday like ``(Mon)`` sits between the date and the time, so this
    helper parses the date+time directly with a regex instead of ``_parse_dt``.
    Returns ``None`` when there is no date so callers drop the ``source_date``
    metadata key rather than emit a bogus value.
    """
    if not s:
        return None
    m = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})(?:[^0-9]*?(\d{1,2}):(\d{2}))?", s)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    hh = int(m.group(4)) if m.group(4) else 0
    mm = int(m.group(5)) if m.group(5) else 0
    return f"{y:04d}-{mo:02d}-{d:02d}T{hh:02d}:{mm:02d}:00+00:00"


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
                    # Every turn of the session carries the session's date so the
                    # owner can label each message with a `born_on` (not just the
                    # first turn). `_transcript_messages` uses it for the
                    # `source_date` metadata seam.
                    "date": dstr,
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

    Each message carries the SOURCE SESSION's id in `metadata` under the
    generic `source_session_id` key (derived from `dia_id` = `{sid}_t{i}`).
    This is the provenance seam the search tools surface, so a retrieval can
    be scored as LongMemEval ID-membership: did a surfaced ref's session id
    appear in the retrieved set (`answer_session_ids`)? It is general -- any
    pipeline that persists turns from a dialogue can tag them the same way.
    """
    messages: list[ModelMessage] = []
    prev_sid: str | None = None
    for t in raw_turns:
        text = t["content"]
        dia_id = t.get("dia_id", "")
        # dia_id == "{sid}_t{i}"; the source session id is the part before the
        # last "_t<i>" (index may itself contain '_'); strip the _t suffix.
        sid = dia_id.rsplit("_t", 1)[0] if dia_id else None
        # Propagate the session's datetime into the metadata of EVERY message so
        # the consolidation owner can label each message with a `born_on` (the
        # `source_date` seam `resident_consolidate_text` reads). This is what
        # keeps the date alive for non-first turns -- it was previously dropped.
        iso_date = _iso_date(t.get("date"))
        metadata: dict = {"source_date": iso_date} if iso_date else {}
        if sid:
            metadata["source_session_id"] = sid
        # The session's datetime ALSO rides inline on the FIRST turn of each
        # session (session-id change) so `search_conversation_history` /
        # `read_turn` re-surface the time anchor the model needs to resolve
        # "how long ago"/"which first" against. Only the first turn: prefixing
        # every message would pollute content and duplicate the source_date
        # already in metadata.
        is_fresh_session = sid is not None and sid != prev_sid
        if is_fresh_session and t.get("date"):
            text = f"[date: {t['date']}] {text}"
        prev_sid = sid
        if t["role"] == "assistant":
            messages.append(
                ModelResponse(
                    parts=[TextPart(content=text)],
                    run_id=run_id,
                    conversation_id=cid,
                    metadata=metadata,
                )
            )
        else:
            messages.append(
                ModelRequest(
                    parts=[UserPromptPart(content=text)],
                    run_id=run_id,
                    conversation_id=cid,
                    metadata=metadata,
                )
            )
    return messages


async def _persist_seed_direct(
    store: FileStepStore,
    raw_turns: list[dict],
    cid: str,
    run_id: str,
) -> tuple[list[dict], list[ModelMessage]]:
    """Persist raw LME turns directly into the store (no agent, no index).

    Returns ``(rows, messages)`` — ``rows`` is empty (there is no turn-index
    seed-row list anymore) and ``messages`` is the full `ModelMessage`
    transcript that was persisted (``raw_turns`` flattened). ``messages`` is
    returned so callers can hand the *same* haystack to downstream consumers
    (the memory-owner consolidation) without rebuilding it — and, importantly,
    so they can assert it contains only pre-question content.

    Writes ONE run record + ONE snapshot holding the full transcript so
    `search_conversation_history`/BM25 and `read_turn('{run_id}:{turn_id}')`
    resolve — no LLM chat agent runs and no summarizer is called during seed.
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
    return [], messages


def _copy_tree(src: Path, dst: Path) -> int:
    """Recursively copy all files under ``src`` into ``dst`` (idempotent)."""
    if not src.is_dir():
        return 0
    n = 0
    for item in src.rglob("*"):
        if item.is_file():
            rel = item.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.read_bytes())
            n += 1
    return n


def _persist_durable_memory(qid: str, live_dir: Path) -> int:
    """Snapshot a question's freshly-consolidated memory store to the durable dir.

    Copies every file under ``live_dir`` (the ``memory/<qid>/`` the owner just
    wrote — ``MEMORY.md``, any topic files, and the ``.memory-state.json`` CAS
    sidecar) into ``data/benchmarks/memory/<qid>/``. Idempotent overwrite, so
    re-running on unchanged input just re-stamps the same bytes. Call it only
    after a REAL owner consolidation (never after a restore-skip), otherwise it
    would duplicate what was already durable. Returns files copied.
    """
    snap = MEMORY_DURABLE_DIR / qid
    if not live_dir.is_dir():
        return 0
    n = _copy_tree(live_dir, snap)
    if n:
        print(
            f"[lmem]     persisted {n} durable memory files -> {snap}",
            flush=True,
        )
    return n


def _restore_durable_memory(qid: str, live_dir: Path) -> int:
    """Copy a durable memory snapshot back into the live store; 0 if none.

    With no snapshot (fresh question, or `PAF_FRESH_MEMORY` forced) returns 0
    so the caller falls through to a real owner consolidation. Otherwise the
    durable snapshot IS the source of truth for a reproducible run: copy all
    files (MD bodies + the CAS sidecar) back into ``memory/<qid>/`` so the
    subject's normal ``store_resolver`` path reads the exact consolidated
    notebook — ZERO owner tokens for the question.
    """
    snap = MEMORY_DURABLE_DIR / qid
    if not snap.is_dir():
        return 0
    n = _copy_tree(snap, live_dir)
    if n:
        print(
            f"[lmem]     REUSED {n} durable memory files -> {live_dir} "
            "(0 owner tokens)",
            flush=True,
        )
    return n


def _persist_durable_summaries(qid: str, live_dir: Path) -> int:
    """Snapshot a question's freshly-summarized thread store to the durable dir.

    Copies every file under ``live_dir`` (the ``summaries/<qid>/`` the seed-time
    summarizer wrote — ``conversations/<hash(cid)>/summaries.jsonl`` + the
    thread watermark) into ``data/benchmarks/<stem>-summaries/<qid>/``.
    Idempotent overwrite, so re-running on unchanged input just re-stamps the
    same bytes. Call it only after a REAL summarizer pass (never after a
    restore-skip). Returns files copied.
    """
    snap = SUMMARIES_DURABLE_DIR / qid
    if not live_dir.is_dir():
        return 0
    n = _copy_tree(live_dir, snap)
    if n:
        print(
            f"[lmem]     persisted {n} durable summary files -> {snap}",
            flush=True,
        )
    return n


def _restore_durable_summaries(qid: str, live_dir: Path) -> int:
    """Copy a durable thread-summary snapshot back into the live store; 0 if none.

    With no snapshot (fresh question, or a forced refresh) returns 0 so the
    caller falls through to a real summarizer pass. Otherwise the durable
    snapshot IS the source of truth for a reproducible run: copy all files back
    into ``summaries/<qid>/`` so the subject's ThreadSummarizer capability reads
    the exact summaries — ZERO summarizer tokens for the question.
    """
    snap = SUMMARIES_DURABLE_DIR / qid
    if not snap.is_dir():
        return 0
    n = _copy_tree(snap, live_dir)
    if n:
        print(
            f"[lmem]     REUSED {n} durable summary files -> {live_dir} "
            "(0 summarizer tokens)",
            flush=True,
        )
    return n


def _extract_surfaced_session_ids(recall: dict) -> list[str]:
    """Pull the source-session ids the search tools surfaced this turn.

    Both `search_conversation_history` (``| session: ...``) and `read_turn`
    tag their matched refs with the evidence session id, which is the generic
    `source_session_id` provenance carried on the seeded messages (see
    `_transcript_messages`). We collect the union across all tool
    outputs — the "retrieved top-k" any ranked retriever would return -- and
    return them in first-seen order. Empty when a run's outputs carry no such
    tags (e.g. older artifacts), which is the fallback trigger for the text
    probe.
    """
    import re

    ids: list[str] = []
    seen: set[str] = set()
    pat = re.compile(r"(?:session|session_id)[:=]\s*([A-Za-z0-9_]+)")
    for out in recall.get("tool_outputs", []) or []:
        if not isinstance(out, str):
            continue
        for m in pat.finditer(out):
            sid = m.group(1)
            if sid and sid not in seen:
                seen.add(sid)
                ids.append(sid)
    return ids


def _id_membership_recall(recall: dict, gold_ids: list[str]) -> dict:
    """LongMemEval recall@k by ID-membership over the surfaced retrieval set.

    Official semantics (print_retrieval_metrics.py): recall is independent of
    the final answer and measured from the retrieval log -- whether the ground
    truth `answer_session_ids` appear among the ids the retriever surfaced.
    `k` = the number of ids surfaced (the analog of top-k; `search_conversation_history`
    returns rank-ordered refs per call).

    - `recall_all@k` = fraction of questions where ALL gold ids are surfaced
    - `ndcg_any@k`   = any gold id surfaced, position-weighted by rank

    Abstention (`_abs`) questions are excluded by the caller, matching the
    official evaluator's `'_abs' not in question_id` filter.
    """
    surfaced = _extract_surfaced_session_ids(recall)
    found = [g for g in gold_ids if g in surfaced]
    k = len(surfaced)
    recall_all = round(len(found) / len(gold_ids), 4) if gold_ids else 1.0
    # NDCG_any@k: give any-gold-hit a position-weight; simplified (DCG of a
    # single-query binary relevance list vs ideal). We report the recall_all
    # primarily and ndcg_any as the "any gold surfaced" signal.
    ndcg_any = 1.0 if found else 0.0
    return {
        "retrieval_recall": recall_all,
        "ndcg_any": ndcg_any,
        "k": k,
        "surfaced": surfaced,
        "method": "id-membership",
    }


def _retrieval(recall: dict, gold_session_texts: list[str] | None = None) -> dict:
    """Did the search tools surface the evidence for this question?

    LongMemEval ID-membership recall@k. The question's `answer_session_ids` are
    the ground truth; a search result counts if its surfaced `session=<id>`
    (from `source_session_id` provenance) names an evidence session. Independent
    of the final answer, `_abs` excluded by the caller. When a run carries no
    `answer_session_ids` there is nothing to score -> recall 0.
    """
    gold_ids = recall.get("answer_session_ids") or []
    return _id_membership_recall(recall, gold_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> dict:
    # Long full-suite runs open many transient file/socket handles (step-persistence
    # store, durable memory, search corpus, Langfuse tracing). The default soft
    # RLIMIT_NOFILE of 1024 is too low and the process hits EMFILE (Errno 24) mid-run
    # (~q336 on a 472-Q run). Raise the soft limit to 65536, capped at the hard limit.
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if _soft < 65536:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))

    spec = load_agent_spec("longmemeval")

    questions = _load_questions()
    if not questions:
        raise RuntimeError("No questions loaded from LongMemEval data")

    session = f"lmem-{time.strftime('%Y%m%d-%H%M%S')}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    traced = setup_observability()
    if not traced:
        print(
            "[run_lmem] WARNING: Langfuse tracing NOT configured — run will "
            "produce JSON/artifacts but NO Langfuse traces."
        )
    else:
        print(f"[run_lmem] tracing ON → Langfuse (session={session})", flush=True)

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

    # ---- Incremental persistence (crash-resilience, 2026-08-18) ----
    # Each completed question's `row` is appended as JSONL to
    # `lmem-<session>.partial.jsonl` the moment it's judged, so a crash (e.g. the
    # raw httpx.ReadError at q328/472 on 2026-08-18 that discarded 328 completed
    # questions) never loses finished work. With `LME_RESUME_PARTIAL=1` the run
    # READS an existing partial file back into `all_recall_rows` and skips any
    # question_id already present, turning an interruption into a resumable one.
    # The final `lmem-<ts>.json` is still written at the end with all rows; the
    # partial is a durable side-channel, not the report.
    partial_path = RESULTS_DIR / f"{session}.partial.jsonl"
    resume_partial = os.environ.get("LME_RESUME_PARTIAL", "0") == "1"
    _partial_fh = None

    # Resume: if a partial JSONL exists for THIS question set and resume is on,
    # replay it into all_recall_rows (completion order) and build a skip-set of
    # question_ids so the loop below only re-runs what's unfinished. Saved rows
    # are still weighted into the final aggregation exactly as if freshly judged.
    _done_qids: set[str] = set()
    if resume_partial and partial_path.exists():
        _n_done = 0
        with partial_path.open() as _pf:
            for _ln in _pf:
                _ln = _ln.strip()
                if not _ln:
                    continue
                try:
                    _prev = json.loads(_ln)
                except json.JSONDecodeError:
                    continue  # trailing partial line from a kill mid-write
                _qid = _prev.get("question_id")
                if _qid:
                    _done_qids.add(str(_qid))
                    all_recall_rows.append(_prev)
                    _n_done += 1
        print(
            f"[run_lmem] LME_RESUME_PARTIAL=1: replayed {_n_done} completed "
            f"question(s) from {partial_path.name}; {len(questions)} requested, "
            f"{len(questions) - _n_done} remaining",
            flush=True,
        )
    # Write the agent spec once so the artifact dir is self-describing.
    (artifact_dir / "meta.json").write_text(
        json.dumps(
            {
                "subject": spec.model,
                "judge": JUDGE_MODEL,
                "data": str(DATA),
                "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
    )

    # The recall agent is built PER-QUESTION (inside the loop below) so its
    # memory-owner can be gated off when that question already has a durable
    # snapshot -- never call a model when the durable store already has the
    # answer.
    for qi, q in enumerate(questions, start=1):
        cid = f"lmem-{q['question_id']}"
        # Resume: skip any question whose row was already replayed from the
        # partial file (LME_RESUME_PARTIAL=1), so an interrupted run picks up
        # exactly where it left off instead of re-answering completed QIDs.
        if _done_qids and str(q.get("question_id", "")) in _done_qids:
            print(
                f"[lmem] q{qi}/{len(questions)} {q.get('question_id', '?')} "
                f"SKIPPED (already in partial, resumed)",
                flush=True,
            )
            continue
        raw_turns = _raw_turns(q)
        # NO per-question turn cap: seed the FULL haystack so the owner sees
        # every session and can distill facts that live later in the stack
        # (e.g. a gold answer dated late in a long single-session-user run).
        # A cap truncated the owner context mid-haystack and silently dropped
        # late facts from durable memory.
        print(
            f"[lmem] q{qi}/{len(questions)} {q['question_type']} "
            f"{q.get('question_id', '?')} seeding {len(raw_turns)} turns — "
            f"recall: {q['question'][:48]!r}",
            flush=True,
        )

        # ---- Seed the haystack sessions ------------------------------
        # DIRECT-PERSIST SEED (no agent loop): raw LME turns are written
        # straight into the store with their TRUE roles. The store persists
        # the full transcript for `read_turn('{run_id}:{turn_id}')` /
        # `search_conversation_history` resolution. No summarizer runs; no
        # turn index exists anymore.
        run_id = f"seed-{cid}"
        seed_rows, seed_msgs = asyncio.run(
            _persist_seed_direct(
                store,
                raw_turns,
                cid,
                run_id,
            )
        )
        all_seed_rows.extend(seed_rows)
        print(
            f"[lmem]   q{qi} seeded {len(raw_turns)} raw turns (raw persist, "
            f"no index / no summarizer)",
            flush=True,
        )

        # ---- Memory consolidation (MEMORY=1): owner folds the haystack ----
        # The eviction window is trim-only (TurnBoundSlidingWindow) and never
        # runs the owner — direct-persist never flows through a live eviction
        # anyway. Instead, run the owner over the FULL seeded haystack
        # (`seed_msgs`) HERE, before recall, so the recall turn's <memory>
        # injection (which re-reads the store every request) renders a
        # POPULATED notebook.
        #   - HONEST-MEMORY GUARD: seed_msgs is built from q["haystack_sessions"]
        #     ONLY (see _raw_turns). The recall question/answer turn is asked
        #     AFTER seeding (_run_turn) and is NOT in this list — assert that so
        #     a future refactor can't leak the answer turn into the owner.
        #   - Owner writes into mem_dir = artifact_dir/"memory" — the same
        #     store _build_agent(MEMORY=1) reads (matches its default).
        # Pre-declare the per-question durable-memory state so the recall-agent
        # gating below always sees `restored` / `fresh_memory` bound, even when
        # MEMORY is off (then no durable memory exists -> owner enabled).
        fresh_memory = bool(os.environ.get("PAF_FRESH_MEMORY"))
        # LME_NO_REVIEWER=1 skips the pairwise reviewer pass below
        # (extractor-only memory, no archive/supersede curation) — the
        # reviewer ablation arm. Requires PAF_FRESH_MEMORY=1 so memory is regenerated without
        # the reviewer (existing durable snapshots have curation baked in).
        no_reviewer = bool(os.environ.get("LME_NO_REVIEWER"))
        restored = 0
        if MEMORY:
            from paf.capabilities.memory._view import JournalViewStore

            # HONEST-MEMORY GUARD (Landon: owner reads up to the question, not
            # further). raw_turns is built purely from q["haystack_sessions"]
            # (see _raw_turns), so the recall question/answer turn is asked
            # AFTER seeding and must NOT appear in the seed. Assert that so a
            # future refactor can't leak the answer turn into the owner.
            _seed_content = "\n".join(
                (t.get("content") or "") for t in raw_turns if isinstance(t, dict)
            )
            assert q["question"] not in _seed_content, (
                "seed turns must not contain the recall question (look-ahead leak)"
            )
            mem_dir = artifact_dir / "memory" / q["question_id"]

            # DURABLE MEMORY REUSE: if a committed snapshot exists for this
            # question (and we're not forcing fresh), copy it into the live
            # store and SKIP the owner entirely -- 0 owner tokens. The durable
            # snapshot is the source of truth for a reproducible run; refresh
            # it via PAF_FRESH_MEMORY=1 (all) or LME_QIDS=<qid> PAF_FRESH_MEMORY=1
            # (specific).
            restored = (
                _restore_durable_memory(q["question_id"], mem_dir)
                if not fresh_memory
                else 0
            )
            if restored:
                print(
                    f"[lmem]   q{qi} memory REUSED from durable snapshot "
                    f"(skipped owner)",
                    flush=True,
                )
            else:
                if fresh_memory:
                    print(
                        f"[lmem]   q{qi} PAF_FRESH_MEMORY=1 -> re-consolidating "
                        f"(durable snapshot overwritten)",
                        flush=True,
                    )
                store_mem = JournalViewStore(mem_dir)
                # Chunked seed consolidation: folds the haystack in token-bounded,
                # message-boundary-aligned chunks into durable memory so
                # single-mention thematic clusters survive the owner's single-shot
                # budget.
                from tests._lmem_consolidate import (
                    consolidate_conversation_chunked,
                )

                consolidation = consolidate_conversation_chunked
                chunk_label = "chunked"
                # M2 three-layer model: the owner appends immutable facts to an
                # append-only journal (`append_memory` is its ONLY tool), and
                # MEMORY.md is rendered from the journal AT RECALL (below), not
                # per chunk. The assistant's Memory injection reads MEMORY.md
                # from this same dir, so rendering here makes it populate.
                from paf.capabilities.memory._journal import JournalStore
                from paf.capabilities.memory._render import write_render

                journal = JournalStore(mem_dir)
                asyncio.run(
                    consolidation(
                        seed_msgs,
                        store=store_mem,
                        agent_name="assistant",
                        namespace="",
                        journal=journal,
                    )
                )
                print(
                    f"[lmem]   q{qi} consolidated {len(seed_msgs)} seed msgs "
                    f"({chunk_label}) (up to question) into {mem_dir}",
                    flush=True,
                )
                # Render the journal -> MEMORY.md at recall: the subject's
                # injected notebook is the recall presentation of the journal.
                _rendered = write_render(journal, str(mem_dir / "MEMORY.md"))
                print(
                    f"[lmem]   q{qi} rendered {journal.count()} journal records "
                    f"-> MEMORY.md ({len(_rendered)} chars) for recall",
                    flush=True,
                )
                # Pairwise reviewer pass before recall: every fact in this
                # fresh journal vs the older facts it might duplicate or
                # update (start_after=0 — the whole journal is new). It
                # re-renders MEMORY.md when it changes anything.
                # LME_NO_REVIEWER=1 skips this (extractor-only memory).
                if no_reviewer:
                    print(
                        f"[lmem]   q{qi} LME_NO_REVIEWER=1 -> skipped reviewer "
                        f"(extractor-only memory)",
                        flush=True,
                    )
                else:
                    from paf.capabilities.memory._pairwise import review_new_facts

                    _report = asyncio.run(
                        review_new_facts(
                            str(mem_dir), start_after=0, agent_name="assistant"
                        )
                    )
                    print(
                        f"[lmem]   q{qi} pairwise reviewer: {_report.summary()}",
                        flush=True,
                    )
                    from paf.capabilities.memory._cascade import STATS, screen_enabled

                    if screen_enabled():
                        print(f"[lmem]   q{qi} {STATS.summary()}", flush=True)
                _persist_durable_memory(q["question_id"], mem_dir)

        # ---- Thread summarizer: fold the haystack into per-conversation beats --
        # Mirrors the memory owner: run the seed-time summarizer over the SAME
        # seeded haystack (seed_msgs) so the recall turn's `<thread-summary>`
        # injection + `list_turn_summaries` tool read a POPULATED store. Durable
        # snapshot reuse (0 summarizer tokens) mirrors the memory path. The cid
        # is stable per question (see above), so the per-conversation summary
        # store keyed by it is reproducible across runs.
        if THREAD_SUMMARY:
            from tests._lmem_summarize import summarize_conversation

            summaries_dir = artifact_dir / "summaries" / q["question_id"]
            restored_s = _restore_durable_summaries(q["question_id"], summaries_dir)
            if restored_s:
                print(
                    f"[lmem]   q{qi} thread summaries REUSED from durable snapshot "
                    "(skipped summarizer)",
                    flush=True,
                )
            else:
                n_blocks = asyncio.run(
                    summarize_conversation(
                        seed_msgs,
                        capture_root=summaries_dir,
                        conversation_id=cid,
                        agent_name="assistant",
                    )
                )
                print(
                    f"[lmem]   q{qi} summarized {n_blocks} thread block(s) "
                    f"-> {summaries_dir}",
                    flush=True,
                )
                _persist_durable_summaries(q["question_id"], summaries_dir)

        # ---- Recall: cold-start the question -------------------------
        # Point the subject's Memory store at THIS question's notebook so
        # search_memory/read_memory/inject read exactly what the owner just
        # consolidated (memory/<question_id>). Set on the framework module
        # global (a `from ... import` local alias would NOT update the module
        # global the store_resolver closure reads).
        # The recall agent is built per-question; durable-memory state still
        # gates the seed-time owner skip (below), and the agent uses the
        # trim-only turn-bounded window — no eviction-time consolidation.
        if MEMORY:
            import tests.framework as _rl

            _rl._CURRENT_MEMORY_DIR = artifact_dir / "memory" / q["question_id"]
        if THREAD_SUMMARY:
            import tests.framework as _rl

            _rl._CURRENT_SUMMARIES_DIR = (
                artifact_dir / "summaries" / q["question_id"]
            )
        agent = _build_agent(
            store,
            memory_dir=artifact_dir / "memory",
            summaries_dir=artifact_dir / "summaries",
            # v2 scoped-recall arm: force the lmem project section and anchor
            # the recent-history window to the question date (the 2023 seeds
            # would otherwise all fall outside now_local()'s 30-day window).
            render_project=LME_PROJECT,
            render_now=(
                datetime.fromisoformat(_iso_date(q.get("question_date")))
                if q.get("question_date")
                else None
            ),
        )
        qctx = (
            f"The date now is {q['question_date']}. " if q.get("question_date") else ""
        )
        row = _run_turn(
            agent,
            qctx + q["question"],
            cid,
            f"lmem.recall.{qi}",
            session,
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
            q["question_type"],
            q["question"],
            q["answer"],
            row["text"],
            q.get("question_id", f"q{qi}"),
            span_name=f"lmem.judge.{qi}",
            session=session,
        )
        row["autoeval_label"] = {"model": JUDGE_MODEL, "label": row["judged"]["label"]}

        # Retrieval signal (independent of judge).
        if not row.get("tools"):
            # No tool calls => the answer came from in-context context (injected
            # <memory>, system scaffold, or the seed itself). That is NOT a failed
            # retrieval -- retrieval was never attempted because it wasn't needed.
            # Skip scoring (exclude from the aggregate) rather than scoring 0.0,
            # which would conflate "didn't need to look" with "couldn't find".
            row["retrieval"] = {
                "retrieval_recall": None,
                "k": 0,
                "surfaced": [],
                "method": "skipped-no-tools",
            }
        else:
            row["retrieval"] = _retrieval(row)

        all_recall_rows.append(row)
        # Crash-resilience: persist this judged row immediately so a later crash
        # (transport error, OOM, manual kill) never loses already-finished work.
        if _partial_fh is None:
            _partial_fh = partial_path.open("a", encoding="utf-8")  # append mode
        _partial_fh.write(json.dumps(row) + "\n")
        _partial_fh.flush()
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
    abstention_rows = [
        r for r in recall_rows if lme_is_abstention(r.get("question_id", ""))
    ]
    factual_rows = [
        r for r in recall_rows if not lme_is_abstention(r.get("question_id", ""))
    ]

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

    per_task = {
        t: _acc([r for r in factual_rows if r["question_type"] == t]) for t in LME_TYPES
    }
    deline_by_task: list[float] = [
        v for v in (per_task[t] for t in LME_TYPES) if v is not None
    ]
    task_averaged = (
        round(sum(deline_by_task) / len(deline_by_task), 4) if deline_by_task else None
    )
    overall = _acc(factual_rows)
    abstention = _acc(abstention_rows)

    result = {
        "meta": {
            "subject": spec.model,
            "judge": JUDGE_MODEL,
            "session": session,
            "n_questions": len(recall_rows),
            "data": str(DATA),
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
            "retrieval_recall_rate": (
                round(
                    sum(
                        r["retrieval"]["retrieval_recall"]
                        for r in recall_rows
                        if r["retrieval"].get("retrieval_recall") is not None
                    )
                    / len(
                        [
                            r
                            for r in recall_rows
                            if r["retrieval"].get("retrieval_recall") is not None
                        ]
                    ),
                    4,
                )
                if any(
                    r["retrieval"].get("retrieval_recall") is not None
                    for r in recall_rows
                )
                else None
            ),
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
    # Close the incremental-persistence handle on the success path.
    if _partial_fh is not None:
        _partial_fh.close()
    if traced:
        shutdown_observability()  # flush buffered OTel spans before process exit
    return result


if __name__ == "__main__":
    main()
