"""Beam Lite Test 1 — graded live runner (C1-C4 escalation ladder + ablation).

Tier 3 of the recall evidence spine (Turn-Index-Testing.md): a real chat model must
*recognize* the need for a retrieval tool and invoke the right rung, its answer is
scored against ground-truth atoms by an LLM judge, and we ablate (with vs without
the recall machinery) to get the first "helps/hurts" delta.

Judge != answerer (M2 decision, 2026-08-10):
  * subject  = the agent under test  -> deepseek-v4-flash via openrouter (profile default)
  * judge    = gpt-5.6-luna-pro      -> grades each ground-truth atom 0 / 0.5 / 1

Two-part scoring (the doc's non-negotiable):
  1. TOOL INVOCATION gate: did the subject call the expected tool for this Challenge
     (or abstain-tool-for-C1)? From `FunctionToolCallEvent` during the run.
  2. ANSWER rubric: the judge scores each ground-truth atom 0 (absent) / 0.5 (partial)
     / 1 (correct), plus a GROUNDING tag (cited/invoked evidence vs guessed phrasing)
     -- guards BEAM's "judge over-states retrieval grounding" failure mode.

Ablation arms (same Challenge seed, different capability set):
  * with   = StepPersistence + ConversationSearch + TurnIndex          (full recall machinery)
  * without= StepPersistence only                                      (history persists, no
                                                                         index/search tools)

Per-arm isolated tmp store so arms never collide; cold-start (fresh agent over the
same store+cid) each Challenge.

Run (loads .env for OPENROUTER keys, writes JSON):
    PYTHONPATH= .venv/bin/python -m tests.test_beam_lite
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from pydantic_ai import AgentRunResultEvent
from pydantic_ai.messages import FunctionToolCallEvent, TextPart
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence

from paf.capabilities.memory import Memory  # noqa: F401  (see _build_agent)
from paf.capabilities.search import ConversationSearch
from paf.capabilities.turn_index import TurnIndexCapability, TurnIndexStore
from paf.observability.instrumentation import (
    _json_dumps,
    root_span,
    setup_observability,
    shutdown_observability,
)

from .recall_corpus import CHALLENGES, Challenge

# Subject model under test (the agent being measured). Judge is separate (below).
SUBJECT_MODEL = os.environ.get("BEAM_LITE_SUBJECT_MODEL", "openrouter:deepseek/deepseek-v4-flash-0731:nitro")
# Judge model (fixed, stable across modifications per M2 decision).
JUDGE_MODEL = os.environ.get("BEAM_LITE_JUDGE_MODEL", "openrouter:openai/gpt-5.6-luna-pro")

RESULTS_DIR = Path(__file__).parent / "results"
MAX_LINES = 4  # eviction is cheap: fact scrolls past the injected tail quickly

# Ablation arms (same Challenge seed, different capability set), in report order.
ARMS = ("without", "turn_index", "memory")


# ---------------------------------------------------------------------------
# Agent arms
# ---------------------------------------------------------------------------


def _build_agent(
    *,
    mode: str,
    store: FileStepStore,
    index_dir: Path,
    memory_dir: Path,
    max_lines: int = MAX_LINES,
):
    """Wire the subject agent over a shared store.

    mode='without':     StepPersistence only -- history persists and seeds, but the
                        index/search/memory tools and injections are absent.
    mode='turn_index':  StepPersistence + ConversationSearch + TurnIndex (the ladder):
                        search_conversation_history + injected one-liner index + read_turn.
    mode='memory':      StepPersistence + Memory (markdown backend, injected notebook):
                        write_memory/read_memory/search_memory + injected MEMORY.md.
                        Isolated temp memory_dir so the arm never touches repo data.
    """
    from pydantic_ai import Agent

    caps = [StepPersistence(store=store)]
    if mode == "turn_index":
        source = SnapshotHistorySource(store)
        caps.append(ConversationSearch(source=source, scope="conversation", add_instructions=False))
        caps.append(
            TurnIndexCapability(
                source=source,
                index_store=TurnIndexStore(index_dir),
                max_lines=max_lines,
            )
        )
    elif mode == "memory":
        from paf.capabilities.memory import Memory

        caps.append(
            Memory.from_spec(
                backend="markdown",
                directory=str(memory_dir),
                agent_name="beam-lite",
                inject_memory=True,
            )
        )
    return Agent(
        SUBJECT_MODEL,
        name="beam-lite-subject",
        capabilities=caps,
    )


def _run_turn(agent, prompt: str, cid: str, name: str, session: str) -> tuple[list[str], str, dict]:
    """Run one turn via run_stream_events; return (tools_invoked, text, usage).

    The turn is wrapped in a root span named ``name`` (a per-turn Langfuse trace
    -- subject turns are ``beam-lite.<challenge>.<arm>.<phase>``) so each seed /
    evict / recall run is an individually identifiable, fully-named trace with
    clean I/O. Every turn in one benchmark execution shares ``session`` (set as
    ``langfuse.session.id``), so Langfuse groups a whole run into ONE expandable
    session instead of a wall of 75 ungrouped traces. This is the graded-evidence
    spine: traces are only trustworthy if you can tell which challenge+arm+phase
    each one is.

    usage is built from the authoritative AgentRunResultEvent (fires once at run
    end, carrying the whole-run RunUsage):
      {'input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens',
       'total_tokens', 'requests'}
    On a provider that omits part of usage the missing fields default to 0.
    """
    tool_calls: list[str] = []
    text_parts: list[str] = []
    usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0,
                             "cache_read_tokens": 0, "cache_write_tokens": 0,
                             "total_tokens": 0, "requests": 0}

    async def run():
        with root_span(
            name,
            **{
                "langfuse.observation.input": _json_dumps({"prompt": prompt, "conversation_id": cid}),
                "langfuse.session.id": session,
            },
        ) as _span:
            async with agent.run_stream_events(prompt, conversation_id=cid) as es:
                async for event in es:
                    if isinstance(event, FunctionToolCallEvent):
                        tool_calls.append(str(event.part.tool_name or ""))
                    elif isinstance(event, AgentRunResultEvent):
                        u = event.result.usage  # RunUsage (property, not a method)
                        usage["input_tokens"] = getattr(u, "input_tokens", 0) or 0
                        usage["output_tokens"] = getattr(u, "output_tokens", 0) or 0
                        usage["cache_read_tokens"] = getattr(u, "cache_read_tokens", 0) or 0
                        usage["cache_write_tokens"] = getattr(u, "cache_write_tokens", 0) or 0
                        usage["total_tokens"] = getattr(u, "total_tokens", 0) or 0
                        usage["requests"] = getattr(u, "requests", 0) or 0
                        # Trace output on the root span while it's still open.
                        _span.set_attribute(
                            "langfuse.observation.output",
                            _json_dumps({"output": event.result.output}),
                        )
                    elif hasattr(event, "part") and event.part is not None:
                        if isinstance(event.part, TextPart):
                            text_parts.append(event.part.content)

    asyncio.run(run())
    return tool_calls, "".join(text_parts), usage


# ---------------------------------------------------------------------------
# One Challenge through one arm
# ---------------------------------------------------------------------------


def _sum_usage(usages: list[dict]) -> dict:
    """Sum a list of per-turn usage dicts into one aggregate."""
    keys = ("input_tokens", "output_tokens", "cache_read_tokens",
            "cache_write_tokens", "total_tokens", "requests")
    return {k: sum(u.get(k, 0) or 0 for u in usages) for k in keys}


def _run_challenge(challenge: Challenge, *, mode: str, cid: str, session: str) -> dict:
    """Seed, evict, cold-start recall for ONE Challenge on ONE arm.

    mode="memory" additionally:
      * appends challenge.memory_seed_instruction to the SEED prompt (invited
        capture), so the arm measures retrieval-via-memory rather than a
        capture-judgment confound. An empty instruction (C4) leaves it as-is.
      * records whether the model actually called write_memory during SEED
        (captured = it chose to persist; for C4 that means it violated the
        user's "don't make a note of it", worth surfacing explicitly).

    Usage is tracked in three buckets so a recall turn's cost is separable from
    the seed/eviction setup cost: seed / evict (sum of fillers) / recall.
    """
    memory = mode == "memory"
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        store = FileStepStore(directory=base / "steps")
        index_dir = base / "index"
        memory_dir = base / "memory"

        seed_prompt = challenge.seed_prompt
        if memory and challenge.memory_seed_instruction:
            seed_prompt = f"{seed_prompt} {challenge.memory_seed_instruction}"

        # Per-turn trace names (the graded-evidence spine): every turn is its own
        # named Langfuse trace so you can tell seed/evict/recall and challenge+arm
        # apart at a glance in the trace list.
        def name(phase: str) -> str:
            return f"beam-lite.{challenge.id}.{mode}.{phase}"

        # SEED: a real turn plants the fact in transcript + (turn_index) index
        #       or (memory) MEMORY.md notebook.
        seed_agent = _build_agent(mode=mode, store=store, index_dir=index_dir,
                                  memory_dir=memory_dir)
        seed_tools, _, seed_usage = _run_turn(seed_agent, seed_prompt, cid, name("seed"), session)
        captured = "write_memory" in seed_tools

        # EVICT: real-but-bland turns push the fact past the max_lines tail.
        chit = _build_agent(mode=mode, store=store, index_dir=index_dir,
                            memory_dir=memory_dir)
        evict_usage = []
        for i, line in enumerate(challenge.fillers):
            _, _, u = _run_turn(chit, line, cid, name(f"evict.{i+1}"), session)
            evict_usage.append(u)

        # COLD-START recall in a FRESH agent over the same store + cid.
        recall_agent = _build_agent(mode=mode, store=store, index_dir=index_dir,
                                    memory_dir=memory_dir)
        tools, text, recall_usage = _run_turn(recall_agent, challenge.query, cid, name("recall"), session)

    invoked = dict.fromkeys(tools)  # dedupe, preserve order
    usage = {
        "seed": seed_usage,
        "evict": _sum_usage(evict_usage),
        "recall": recall_usage,
        "total": _sum_usage([seed_usage] + evict_usage + [recall_usage]),
    }
    return {
        "challenge": challenge.id,
        "arm": mode,
        "tools_invoked": list(invoked),
        "captured": captured,
        "usage": usage,
        "answer": text.strip(),
        "expected_tool_invoked": _tool_gate(challenge, tools),
    }


def _tool_gate(challenge: Challenge, tools: list[str]) -> bool:
    """Two-part assertion part 1: did the subject invoke the expected tool?
    For C1 the expected mechanism is 'none' (in-context) -- pass iff NO tool called.
    Verbatim-later: normalize names (search_conversation_history vs search_index)."""
    if challenge.expected_tool == "none":
        return len(tools) == 0
    return challenge.expected_tool in tools


# ---------------------------------------------------------------------------
# Judge (gpt-5.6-luna-pro): score ground-truth atoms + grounding tag
# ---------------------------------------------------------------------------


def _judge(challenge: Challenge, datum: dict, session: str) -> dict:
    """Score one answer: per-atom 0/0.5/1 + grounding tag, via the judge model."""
    from pydantic_ai import Agent

    atoms = " | ".join(challenge.ground_truth_atoms)
    judge = Agent(
        JUDGE_MODEL,
        name="beam-lite-judge",
        instructions=(
            "You are a strict rubric judge for a memory-retrieval benchmark. "
            "Score the SUBJECT'S ANSWER against each ground-truth atom: 1 = correct/fully "
            "present, 0.5 = partially/indistinctly present, 0 = absent or wrong. "
            "Also tag grounding: 'tool' if the answer cites/derives from a retrieved tool result "
            "or specific in-context line, 'guess' if it is asserted without evidence. "
            "Be honest and strict -- do NOT award credit for plausible phrasing if the atom "
            "value is not actually present. "
            "Respond ONLY as JSON: {\"atoms\": [{\"atom\": <str>, \"score\": <0|0.5|1>}], "
            "\"grounding\": <\"tool\"|\"guess\">}."
        ),
    )

    prompt = (
        f"Challenge {challenge.id} (mechanism: {challenge.mechanism}).\n"
        f"Ground-truth atoms: {atoms}\n"
        f"Question: {challenge.query}\n"
        f"Tools invoked by subject: {datum['tools_invoked']}\n"
        f"Subject's answer: {datum['answer']!r}\n"
        "Score each atom and tag grounding."
    )

    result = None
    with root_span(
        f"beam-lite.judge.{challenge.id}.{datum.get('arm', '?')}",
        **{
            "langfuse.observation.input": _json_dumps({"challenge": challenge.id, "query": challenge.query}),
            "langfuse.session.id": session,
        },
    ) as _jspan:
        result = judge.run_sync(prompt)
        text = result.output
        # Trace output on the judge span while it's still open (raw judge JSON).
        _jspan.set_attribute(
            "langfuse.observation.output",
            _json_dumps({"answer": text}),
        )
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        blob = json.loads(text[start:end]) if start != -1 and end > start else {}
    except json.JSONDecodeError:
        blob = {}

    atom_scores = {a.get("atom", ""): float(a.get("score", 0.0)) for a in blob.get("atoms", [])}
    # Default sensible: an atom not returned by the judge scores 0.
    out = []
    for atom in challenge.ground_truth_atoms:
        out.append({"atom": atom, "score": atom_scores.get(atom, 0.0)})
    return {
        "atoms": out,
        "grounding": blob.get("grounding", "guess"),
        "raw": text,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _suite(session: str) -> list[dict]:
    rows: list[dict] = []
    for challenge in CHALLENGES:
        for mode in ARMS:
            cid = f"beam-lite-{challenge.id.lower()}-{mode}-{uuid.uuid4().hex[:8]}"
            datum = _run_challenge(challenge, mode=mode, cid=cid, session=session)
            datum["judged"] = _judge(challenge, datum, session)
            datum["atom_score"] = sum(a["score"] for a in datum["judged"]["atoms"]) / len(
                datum["judged"]["atoms"]
            )
            rows.append(datum)
    return rows


def _aggregate(rows: list[dict]) -> dict:
    by_arm = {arm: [] for arm in ARMS}
    for r in rows:
        by_arm[r["arm"]].append(r)

    def arm_stats(arm_rows):
        tool_ok = sum(1 for r in arm_rows if r["expected_tool_invoked"])
        atom_mean = sum(r["atom_score"] for r in arm_rows) / len(arm_rows)
        captured = sum(1 for r in arm_rows if r.get("captured"))
        total_input = sum(r["usage"]["total"]["input_tokens"] for r in arm_rows)
        total_output = sum(r["usage"]["total"]["output_tokens"] for r in arm_rows)
        total_tokens = sum(r["usage"]["total"]["total_tokens"] for r in arm_rows)
        recall_input = sum(r["usage"]["recall"]["input_tokens"] for r in arm_rows)
        recall_tokens = sum(r["usage"]["recall"]["total_tokens"] for r in arm_rows)
        return {
            "n": len(arm_rows),
            "tool_gate_pass": tool_ok,
            "tool_gate_rate": round(tool_ok / len(arm_rows), 4),
            "atom_mean": round(atom_mean, 4),
            "capture_count": captured,
            "capture_rate": round(captured / len(arm_rows), 4) if arm_rows else 0.0,
            "tokens": {
                "input": total_input,
                "output": total_output,
                "total": total_tokens,
                "recall_total": recall_tokens,
                "recall_input": recall_input,
            },
            "per_challenge": {
                r["challenge"]: {
                    "atom": r["atom_score"],
                    "tools": r["tools_invoked"],
                    "grounding": r["judged"]["grounding"],
                    "captured": r.get("captured"),
                    "recall_tokens": r["usage"]["recall"]["total_tokens"],
                    "total_tokens": r["usage"]["total"]["total_tokens"],
                }
                for r in arm_rows
            },
        }

    base = arm_stats(by_arm["without"])
    arms = {arm: arm_stats(by_arm[arm]) for arm in ARMS}
    return {
        "subject": SUBJECT_MODEL,
        "judge": JUDGE_MODEL,
        "max_lines": MAX_LINES,
        "arms": arms,
        "delta_vs_baseline": {
            arm: {
                "tool_gate_rate": round(arms[arm]["tool_gate_rate"] - base["tool_gate_rate"], 4),
                "atom_mean": round(arms[arm]["atom_mean"] - base["atom_mean"], 4),
                "recall_tokens": arms[arm]["tokens"]["recall_total"] - base["tokens"]["recall_total"],
                "total_tokens": arms[arm]["tokens"]["total"] - base["tokens"]["total"],
            }
            for arm in ARMS[1:]
        },
    }


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()  # OPENROUTER_API_KEY (+ Langfuse keys/endpoint for tracing)

    traced = setup_observability()
    if not traced:
        print("[beam-lite] WARNING: Langfuse tracing NOT configured (.env keys/endpoint "
              "missing) -- runs will produce JSON but NO Langfuse traces.")

    t0 = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # One Langfuse session per run: all traces in this execution share this id so
    # the whole benchmark collapses into ONE expandable session in Langfuse.
    session = f"beam-lite-{stamp}"
    rows = _suite(session)
    agg = _aggregate(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"beam-lite-{stamp}.json"
    payload = {"meta": {"created": stamp, "elapsed_s": round(time.time() - t0, 1), "langfuse_session": session}, **agg, "rows": rows}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")
    print(json.dumps(agg, indent=2, ensure_ascii=False))

    if traced:
        shutdown_observability()  # flush buffered OTel spans before process exit


if __name__ == "__main__":
    main()
