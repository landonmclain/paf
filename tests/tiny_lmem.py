"""Tiny LongMemEval — a <1 minute repro of the full LME pipeline.

The whole point of this script: validate fast that the PAF memory pipeline
WRITES a real turn index AND RETRIEVES a fact that fell out of context —
without a 10-minute benchmark.

It uses one small LongMemEval question (default `89527b6b`, single-session
type: 2 seed turns) whose answer sits in a long assistant turn, so by design
it falls out of context and must be retrieved via tools.

Crucially, [this runner wires the REAL `TurnSummarizer` into TurnIndexCapability],
which `tests/run_locomo.py::_build_agent` does NOT (it hits the deterministic
stub). That missing wire is why prior LME/LoCoMo runs indexed raw truncated
turn text instead of real `sentence + key_terms + outcome`. This script is
both the fast smoke check AND the reference for the correct build.

Run (from repo root, .env loaded):
    set -a && . ./.env && set +a && PYTHONPATH= .venv/bin/python -m tests.tiny_lmem
    # pick a different question id:
    ... python -m tests.tiny_lmem 89527b6b
    # summarizer OFF (reproduces the stub bug — index rows will lack key_terms/outcome):
    TINY_SUMMARIZER=0 ... python -m tests.tiny_lmem

Exit 0 if the judge says the answer is correct AND retrieval surfaced the
evidence; non-zero otherwise. Never 10 minutes.
"""

from __future__ import annotations

import json
import os
import sys
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

from paf.capabilities.memory import Memory
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

# ---------------------------------------------------------------- config ----
SUBJECT_MODEL = os.environ.get(
    "BEAM_LITE_SUBJECT_MODEL", "openrouter:deepseek/deepseek-v4-flash-0731:nitro"
)
JUDGE_MODEL = os.environ.get("LME_JUDGE_MODEL", "openrouter:openai/gpt-4o-mini")

# The single dial that decides THIS script's whole value: whether the real
# LLM summarizer is wired into TurnIndexCapability (1) or the stub (0).
WIRE_SUMMARIZER = os.environ.get("TINY_SUMMARIZER", "1") != "0"

MAX_LINES = int(os.environ.get("LOCOMO_MAX_LINES", "6"))
LOCOMO_TARGET_TOKENS = int(os.environ.get("LOCOMO_TARGET_TOKENS", "5000"))
LOCOMO_SW_MAX_MESSAGES = int(os.environ.get("LOCOMO_SW_MAX_MESSAGES", "4"))
LOCOMO_SW_KEEP = int(os.environ.get("LOCOMO_SW_KEEP", "2"))

DATA = Path(__file__).parent.parent / "data" / "benchmarks" / "lme_oracle.json"
RESULTS_DIR = Path(__file__).parent / "results"

# ------------------------------------------------ pipeline (run_locomo-faithful) ----
def build_run_locomo_agent(store: FileStepStore, index_dir: Path, memory_dir: Path) -> Agent:
    """EXACT mirror of `tests/run_locomo.py::_build_agent` — INCLUDING its bug.

    TurnIndexCapability is constructed WITHOUT `summarizer=`, so it falls to the
    deterministic stub. Reproduces the "raw truncated text in the index" artifact.
    """
    source = SnapshotHistorySource(store)
    caps = [
        StepPersistence(store=store),
        ConversationSearch(source=source, scope="conversation", add_instructions=False),
        TurnIndexCapability(
            source=source,
            index_store=TurnIndexStore(index_dir),
            max_lines=MAX_LINES,
            add_instructions=False,
            # <-- no summarizer (the bug)
        ),
        Memory.from_spec(
            backend="markdown", directory=str(memory_dir),
            agent_name="locomo-subject", inject_memory=True,
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


def build_agent(store: FileStepStore, index_dir: Path, memory_dir: Path) -> Agent:
    """The CORRECT pipeline: `_build_agent` + the real `TurnSummarizer` wired in.

    This is the reference for what the benchmark SHOULD do. When
    `TINY_SUMMARIZER=0`, falls back to `_build_agent`'s stub for comparison.
    """
    if not WIRE_SUMMARIZER:
        return build_run_locomo_agent(store, index_dir, memory_dir)
    source = SnapshotHistorySource(store)
    caps = [
        StepPersistence(store=store),
        ConversationSearch(source=source, scope="conversation", add_instructions=False),
        # Real LLM summarizer: sentence + key_terms + outcome (160 chars, 6 terms).
        TurnIndexCapability(
            source=source,
            index_store=TurnIndexStore(index_dir),
            max_lines=MAX_LINES,
            add_instructions=False,
            summarizer=TurnSummarizer(model=SUBJECT_MODEL, budget_chars=160, max_key_terms=6),
        ),
        Memory.from_spec(
            backend="markdown", directory=str(memory_dir),
            agent_name="locomo-subject", inject_memory=True,
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


# ------------------------------------------------ turn runner (run_locomo-faithful) ----
def run_turn(agent: Agent, prompt: str, cid: str, name: str, session: str) -> dict:
    tools: list[str] = []
    tool_outputs: list[str] = []
    text = ""
    async def _run():
        nonlocal text
        with root_span(name, **{
            "langfuse.observation.input": _json_dumps({"prompt": prompt, "conversation_id": cid}),
            "langfuse.session.id": session,
        }):
            async with agent.run_stream_events(prompt, conversation_id=cid) as es:
                async for e in es:
                    if isinstance(e, FunctionToolCallEvent):
                        tools.append(str(e.part.tool_name or ""))
                    elif isinstance(e, FunctionToolResultEvent):
                        c = getattr(getattr(e, "part", None), "content", None)
                        if c is None:
                            c = getattr(e, "content", None)
                        if isinstance(c, str) and c.strip():
                            tool_outputs.append(c)
                    elif isinstance(e, AgentRunResultEvent):
                        text = e.result.output
    import asyncio
    asyncio.run(_run())
    return {"text": text, "tools": tools, "tool_outputs": tool_outputs}


# ------------------------------------------------ LME helpers (run_lmem-faithful) ----
def _parse_dt(s: str):
    date_part, _, time_part = s.partition("(")
    y, m, d = (int(x) for x in date_part.strip().split("/"))
    hh = mm = 0
    if time_part.strip():
        try:
            hh, mm = (int(x) for x in time_part.strip().split(":"))
        except ValueError:
            pass
    return (y, m, d, hh, mm)


def seed_turns(q: dict) -> list[dict]:
    sessions, dates, sids = q["haystack_sessions"], q["haystack_dates"], q["haystack_session_ids"]
    tagged = sorted(
        ((_parse_dt(d), d, sid, s) for d, sid, s in zip(dates, sids, sessions, strict=True)),
        key=lambda x: x[0],
    )
    turns = []
    for _dt, dstr, sid, sess in tagged:
        for i, turn in enumerate(sess):
            prefix = f"[date: {dstr}] " if i == 0 else ""
            turns.append({"text": f"{prefix}{turn.get('role','user')}: {turn.get('content','')}",
                          "dia_id": f"{sid}_t{i}"})
    return turns


def gold_probe(q: dict) -> str:
    for sess in q["haystack_sessions"]:
        for turn in sess:
            if turn.get("has_answer") and turn.get("content"):
                return str(turn["content"])
    for sess in q["haystack_sessions"]:
        if sess and sess[0].get("content"):
            return str(sess[0]["content"])
    return ""


def f1(a: str, b: str) -> float:
    as_, bs = set(a.lower().split()), set(b.lower().split())
    if not as_ or not bs:
        return 0.0
    inter = len(as_ & bs)
    return 2 * inter / (len(as_) + len(bs))


def judge(q: dict, response: str) -> dict:
    qtype, gold = q["question_type"], str(q.get("answer", ""))
    prompt = (
        f"Answer the following question. Read the question and the reference answer. "
        f"Determine whether the model's answer is correct (contains the key facts). "
        f"Provide a single token, 'yes' or 'no'.\n\n"
        f"Question: {q['question']}\nReference answer: {gold}\nModel's answer: {response}\n\n"
        f"Is the model's answer correct? (yes/no)"
    )
    judge_agent = Agent(JUDGE_MODEL, name="lme-judge", instructions=(
        "You are the official LongMemEval answer judge. Answer exactly with a single token 'yes' or 'no'."
    ))
    raw = (judge_agent.run_sync(prompt).output or "").strip()
    label = "yes" in raw.lower()
    return {"label": label, "correct": label, "raw": raw, "question_type": qtype}


# ---------------------------------------------------------------- main ----
def main() -> int:
    qid = sys.argv[1] if len(sys.argv) > 1 else "89527b6b"
    data = json.loads(DATA.read_text())
    q = next((x for x in data if str(x.get("question_id")) == qid), None)
    if q is None:
        print(f"no question {qid!r} in {DATA}")
        return 2
    n_turns = sum(len(s) for s in q["haystack_sessions"])
    print(f"question_id={qid} type={q['question_type']} seed_turns={n_turns} recall='{q['question'][:60]}…'")
    print(f"wire_summarizer={WIRE_SUMMARIZER} model={SUBJECT_MODEL}")

    traced = setup_observability()
    if not traced:
        print("[tiny_lmem] WARNING: Langfuse tracing NOT configured — run will "
              "produce JSON/artifacts but NO Langfuse traces.")

    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = time.strftime("%Y%m%d-%H%M%S")
    session = f"tiny-lmem-{run_ts}"
    # Persist the full evidence trail like run_lmem: step persistence (full
    # turn history), the turn index (one-liners + key_terms), and the memory
    # markdown, under a run-scoped artifact dir alongside the report. Without
    # the underlying data a PASS/FAIL is unfalsifiable — and this run is
    # exactly about auditing *what the summarizer put in the index*.
    artifact_dir = RESULTS_DIR / session
    artifact_dir.mkdir(parents=True, exist_ok=True)
    store = FileStepStore(directory=artifact_dir / "store")
    index_dir = artifact_dir / "index"
    memory_dir = artifact_dir / "memory"
    agent = build_agent(store, index_dir, memory_dir)
    (artifact_dir / "meta.json").write_text(json.dumps({
        "subject": SUBJECT_MODEL,
        "judge": JUDGE_MODEL,
        "data": str(DATA),
        "wire_summarizer": WIRE_SUMMARIZER,
        "eviction": {
            "target_tokens": LOCOMO_TARGET_TOKENS,
            "sw_max_messages": LOCOMO_SW_MAX_MESSAGES,
            "sw_keep": LOCOMO_SW_KEEP,
        },
        "max_lines_index_tail": MAX_LINES,
        "traced": traced,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    cid = f"lmem-{uuid.uuid4().hex[:8]}"

        # ---- seed the haystack (2 turns for the default question) ----
    for i, turn in enumerate(seed_turns(q), start=1):
        run_turn(agent, turn["text"], cid, f"tiny.seed.{i}", session)

    # ---- recall: cold-start the question (answer fell out of context) ----
    qctx = f"The date now is {q['question_date']}. " if q.get("question_date") else ""
    r = run_turn(agent, qctx + q["question"], cid, "tiny.recall", session)

    # ---- assert the real summarizer wrote content into the index ----
    index = TurnIndexStore(index_dir)
    entries = index.read_all(conversation_id=cid)
    n_with_kw = sum(1 for e in entries if e.key_terms)
    n_with_out = sum(1 for e in entries if e.outcome)
    print(f"\nindex rows: {len(entries)} | rows with key_terms={n_with_kw} rows with outcome={n_with_out}")
    for e in entries[-4:]:
        print(f"  - [{e.turn_id}] sentence={e.summary[:70]!r} kw={e.key_terms or []} out={e.outcome}")

    probe = gold_probe(q)
    tool_surfaced = any(f1(probe, c) >= 0.30 for c in r["tool_outputs"] if c.strip())
    ans_surfaced = f1(probe, r["text"]) >= 0.35
    if not WIRE_SUMMARIZER:
        n_with_kw = n_with_out = 0  # the stub never fills these

    with root_span(
        "tiny.judge",
        **{
            "langfuse.observation.input": _json_dumps(
                {"question": q["question"], "gold": str(q.get("answer", ""))}
            ),
            "langfuse.session.id": session,
        },
    ):
        verdict = judge(q, r["text"])
    elapsed = time.time() - t0
    retrieved = bool(tool_surfaced or ans_surfaced)
    ok = bool(verdict["correct"]) and retrieved

    # ---- persist the summary report (alongside the artifact dir) ----
    result = {
        "ts": run_ts,
        "question_id": qid,
        "question_type": q["question_type"],
        "seed_turns": n_turns,
        "recall": q["question"],
        "wire_summarizer": WIRE_SUMMARIZER,
        "model": SUBJECT_MODEL,
        "index": {"rows": len(entries), "with_key_terms": n_with_kw, "with_outcome": n_with_out},
        "tools_fired": r["tools"],
        "retrieval": {"tool_surfaced": tool_surfaced, "answer_surfaced": ans_surfaced, "probe_chars": len(probe)},
        "judge": verdict,
        "answer": r["text"],
        "elapsed_s": round(elapsed, 2),
        "pass": ok,
        "retrieved": retrieved,
        "traced": traced,
    }
    out_path = RESULTS_DIR / f"tiny-lmem-{run_ts}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\npersisted: {out_path}")

    print(f"\ntools fired: {r['tools'] or '(none)'}")
    print(f"retrieval: survived tool={tool_surfaced} answer={ans_surfaced} (probe={len(probe)} chars)")
    print(f"model answer: {r['text'][:200]!r}")
    print(f"judge: {'CORRECT' if verdict['correct'] else 'WRONG'} (raw={verdict['raw']!r}, type={q['question_type']})")
    print(f"elapsed: {elapsed:.1f}s")

    print(f"\n=> {'PASS' if ok else 'FAIL'} (correct={verdict['correct']}, retrieved={'yes' if retrieved else 'no'})\n")
    if traced:
        shutdown_observability()  # flush buffered OTel spans before process exit
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
