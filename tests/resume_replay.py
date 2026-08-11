"""Resume-replay: re-run the recall turn of a FINISHED lmem run with a modified
reasoning prompt, without re-seeding the haystack.

WHY: the full LME benchmark re-seeds 20-72 haystack turns (~20 min / ~200k tokens)
every run. For reasoning/calibration iteration we only care about the RECALL turn -
the haystack backend is a fixed fixture we should burn ZERO tokens re-building.

This rehydrates a finished run's persisted store/index/memory and re-asks the last
recall question, with a NEW reasoning system prompt. Framework-native:
  - resume = `Agent.run(..., message_history=<persisted messages>)` + the same
    `conversation_id` (inherited + explicitly passed), so search_conversation_history
    (conversation-scoped) and read_turn('{run_id}:{turn_id}') still resolve.
  - new prompt = `ReinjectSystemPrompt(replace_existing=True)` (pydantic_ai 2.22.0):
    swaps the agent's configured system_prompt into the head of the resumed run.

CRITICAL (verified 2026-08-11 on lmem-20260811-125906): the persisted user prompts
carry the injected <memory>/<turn-index> blocks baked in. On resume those must be
STRIPPED, or the model sees stale + fresh copies. The strip keys off TextContent.metadata
via paf.capabilities.turn_index._markers.injected_artifact_markers() (same convention
the BM25 corpus uses), NOT string regex on tags.

Run (from repo root, .env loaded):
    set -a && . ./.env && set +a && PYTHONPATH= .venv/bin/python -m tests.resume_replay \
        --from tests/results/lmem-20260811-125906 --dry-run
    # live single question:
    ... -m tests.resume_replay --from <dir> --guidance "When a recalled value looks
        wrong, prefer the freshly retrieved source over memory." --question-id <qid>

Exit 0 on dry-run success; on live run exit reflects judge correctness *only if a
question_id is given* (otherwise informational, exit 0). Never re-seeds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    UserPromptPart,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence  # noqa: E402

from paf.capabilities.memory import Memory  # noqa: E402
from paf.capabilities.search import ConversationSearch, SnapshotHistorySource  # noqa: E402
from paf.capabilities.turn_index import (  # noqa: E402
    TurnIndexCapability,
    TurnIndexStore,
    TurnSummarizer,
)
from paf.capabilities.turn_index._markers import injected_artifact_markers  # noqa: E402

SUBJECT_MODEL = os.environ.get(
    "BEAM_LITE_SUBJECT_MODEL", "openrouter:deepseek/deepseek-v4-flash-0731:nitro"
)
JUDGE_MODEL = os.environ.get("LME_JUDGE_MODEL", "openrouter:openai/gpt-4o-mini")
RESULTS_DIR = ROOT / "tests" / "results"
DATA = ROOT / "data" / "benchmarks" / "lme_oracle.json"

# Match the LME harness eviction defaults so a resumed turn still *must* retrieve.
MAX_LINES = int(os.environ.get("LOCOMO_MAX_LINES", "10"))
LOCOMO_TARGET_TOKENS = int(os.environ.get("LOCOMO_TARGET_TOKENS", "5000"))
LOCOMO_SW_MAX_MESSAGES = int(os.environ.get("LOCOMO_SW_MAX_MESSAGES", "6"))
LOCOMO_SW_KEEP = int(os.environ.get("LOCOMO_SW_KEEP", "3"))


# ---------------------------------------------------------------------------
# STRIP: rebuild ModelMessages dropping injected <memory>/<turn-index> blocks.
# ---------------------------------------------------------------------------
def _is_injected_block(text: TextContent) -> bool:
    meta = getattr(text, "metadata", None)
    return meta is not None and any(
        str(meta).startswith(marker) for marker in injected_artifact_markers()
    )


def strip_injected_blocks(messages: list[ModelMessage]) -> tuple[list[ModelMessage], int]:
    """Return the history with marker-tagged injected TextContent dropped from
    UserPromptPart blocks. System / assistant / tool / plain-text parts pass through.
    Returns (cleaned_history, stripped_block_count)."""
    cleaned: list[ModelMessage] = []
    stripped_count = 0
    for msg in messages:
        if isinstance(msg, ModelRequest):
            new_parts = []
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content = part.content
                    if not isinstance(content, list):
                        new_parts.append(part)
                        continue
                    kept: list = []
                    for block in content:
                        if isinstance(block, TextContent) and _is_injected_block(block):
                            stripped_count += 1
                            continue
                        kept.append(block)
                    # only rebuild if something was dropped
                    if len(kept) == len(content):
                        new_parts.append(part)
                    else:
                        new_parts.append(
                            UserPromptPart(
                                content=kept,
                                timestamp=part.timestamp,
                                part_kind=part.part_kind,
                            )
                        )
                else:
                    new_parts.append(part)
            if new_parts is msg.parts:
                cleaned.append(msg)
            else:
                cleaned.append(
                    ModelRequest(
                        parts=new_parts,
                        timestamp=msg.timestamp,
                        run_id=msg.run_id,
                        conversation_id=msg.conversation_id,
                        metadata=msg.metadata,
                    )
                )
        else:
            cleaned.append(msg)
    return cleaned, stripped_count


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------
def _user_prompt_text(part: UserPromptPart) -> str:
    """Durable text of a (already-stripped) user prompt part."""
    content = part.content
    if isinstance(content, str):
        return content
    texts = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, TextContent) and not _is_injected_block(block):
            texts.append(block.content)
    return " ".join(texts)


# ---------------------------------------------------------------------------
# History acquisition
# ---------------------------------------------------------------------------
async def load_conv_history(source: SnapshotHistorySource, conv_id: str) -> list[ModelMessage]:
    runs = [r for r in await source.list_runs() if r.conversation_id == conv_id]
    runs.sort(key=lambda r: (r.started_at or ""))
    full: list[ModelMessage] = []
    for r in runs:
        full.extend(await source.run_history(run_id=r.run_id))
    return full


def find_last_user_prompt(history: list[ModelMessage]) -> tuple[int, str]:
    """Locate the final user prompt (the recall question) in a cleaned history.
    Returns (index, stripped_text)."""
    last_i, last_text = -1, ""
    for i, msg in enumerate(history):
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                t = _user_prompt_text(part).strip()
                if t:
                    last_i, last_text = i, t
    return last_i, last_text


# ---------------------------------------------------------------------------
# Agent construction (mirrors tiny_lmem.build_agent, correct real summarizer)
# ---------------------------------------------------------------------------
def build_resume_agent(store: FileStepStore, index_dir: Path, memory_dir: Path,
                       guidance: str, scratch_store: FileStepStore | None = None,
                       run_id: str | None = None) -> Agent:
    """Build the resume agent.

    `store` = the PRISTINE run dir's store, used ONLY for the READ side
    (SnapshotHistorySource → ConversationSearch / TurnIndexCapability). The
    write side (StepPersistence) is pointed at a separate `scratch_store` so a
    resume run never mutates the baseline fixture. If scratch_store is None it
    falls back to `store` (original behaviour); pass a dedicated scratch store
    to keep the backend pristine across repeated guidance variants.

    NOTE on the index/memory write seams: `index_dir` / `memory_dir` are the
    dirs this agent WRITES to (per-turn summarizer index rows; memory updates).
    To keep the baseline fixture pristine, the caller should pass COPIES of
    the baseline index/memory living in the scratch dir, seeded from originals.
    """
    source = SnapshotHistorySource(store)
    caps = [
        StepPersistence(store=scratch_store or store, run_id=run_id),
        ConversationSearch(source=source, scope="conversation", add_instructions=False),
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
            guidance=guidance,
        ),
    ]
    # Eviction tier mirrors the harness so the resumed turn still *must* retrieve.
    from pydantic_ai_harness.compaction import (
        ClampOversizedMessages, ClearToolResults, SlidingWindowCompaction, TieredCompaction,
    )
    caps.append(TieredCompaction(
        target_tokens=LOCOMO_TARGET_TOKENS,
        tiers=[
            ClampOversizedMessages(max_part_tokens=50_000, keep_head_chars=2_000, keep_tail_chars=2_000),
            ClearToolResults(max_tokens=1, keep_pairs=4),
            SlidingWindowCompaction(max_messages=LOCOMO_SW_MAX_MESSAGES, keep_messages=LOCOMO_SW_KEEP),
        ],
    ))
    agent = Agent(
        SUBJECT_MODEL,
        name="locomo-subject",
        system_prompt=guidance,  # the reasoning variant
        capabilities=[
            ReinjectSystemPrompt(replace_existing=True),
            *caps,
        ],
    )
    return agent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run_replay(args) -> int | dict:
    run_dir = Path(args.from_dir)
    store = FileStepStore(directory=run_dir / "store")
    source = SnapshotHistorySource(store)
    runs = await source.list_runs()

    if args.list:
        by_conv = Counter(r.conversation_id for r in runs)
        for cid, n in sorted(by_conv.items(), key=lambda kv: -kv[1]):
            cv = [r for r in runs if r.conversation_id == cid]
            first, last = cv[0].started_at, cv[-1].started_at
            print(f"{cid:>16}  {n:3d} runs   from {first} to {last}")
        return 0

    if not runs:
        print(f"no runs in {run_dir / 'store'}")
        return 2

    # Pick conversation: explicit --conv wins (prefix match, forgiving), else most-runs.
    all_cids = sorted({r.conversation_id for r in runs})
    if args.conv:
        matches = [c for c in all_cids if c.startswith(args.conv)]
        if len(matches) == 1:
            conv_id = matches[0]
        elif len(matches) > 1:
            print(f"ERROR: --conv {args.conv!r} is ambiguous, matches {matches}")
            return 2
        else:
            print(f"ERROR: --conv {args.conv!r} matches no conversation. Valid ids:")
            for c, n in sorted(((c, sum(1 for r in runs if r.conversation_id == c))
                                for c in all_cids), key=lambda x: -x[1]):
                print(f"  {c}  ({n} runs)")
            return 2
    else:
        by_conv = Counter(r.conversation_id for r in runs)
        conv_id = max(by_conv, key=by_conv.get)
    conv_runs = [r for r in runs if r.conversation_id == conv_id]
    print(f"conversation: {conv_id}  ({len(conv_runs)} runs)")

    history = await load_conv_history(source, conv_id)
    cleaned, stripped = strip_injected_blocks(history)
    print(f"history: {len(history)} messages -> {len(cleaned)} after strip ({stripped} injected blocks removed)")

    # The final cleaned user prompt is the recall question to re-issue.
    last_i, prompt = find_last_user_prompt(cleaned)
    if last_i < 0 or not prompt:
        print(f"ERROR: no user prompt found in conversation {conv_id}")
        return 2
    messages_for_replay = cleaned[:last_i]
    print(f"replay: {len(messages_for_replay)} prior messages + re-issue recall prompt "
          f"{prompt[:70]!r}...")

    if args.dry_run:
        print("\n[DRY-RUN] no LLM call made. Would resume:")
        print(f"  - message_history: {len(messages_for_replay)} messages "
              f"(first {type(messages_for_replay[0]).__name__}, last {type(messages_for_replay[-1]).__name__})")
        print(f"  - guidance: {args.guidance!r}")
        print(f"  - ReinjectSystemPrompt(replace_existing=True) + conversation_id={conv_id}")
        # sanity: no injected blocks remain in the replay window
        left = sum(
            1 for m in messages_for_replay
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, UserPromptPart) and isinstance(p.content, list)
            for b in p.content
            if isinstance(b, TextContent) and _is_injected_block(b)
        )
        print(f"  - residual injected blocks in replay window: {left} (must be 0)")
        return 0 if left == 0 else 3

    # ---- live: build agent pointed at the EXISTING index/memory + a scratch
    # ---- store for StepPersistence, so the baseline fixture stays PRISTINE ----
    guidance = args.guidance or "Answer using the retrieved facts; when memory and a freshly "\
        "retrieved source disagree, prefer the source."

    # Scratch store lives next to the origin, reusing its timestamp (same
    # date-run name + '.resume' suffix). It is wiped fresh each invocation so
    # repeated guidance variants never contaminate the baseline run dir.
    scratch_dir = RESULTS_DIR / f"{run_dir.name}.resume"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)
    scratch_store = FileStepStore(directory=scratch_dir)

    # Seed the scratch dir with COPIES of the baseline index/memory. The resume
    # agent WRITES new index rows (summarizer) and may update memory; pointing
    # those at copies keeps the baseline index/memory files pristine.
    scratch_index = scratch_dir / "index"
    scratch_memory = scratch_dir / "memory"
    for src, dst in ((run_dir / "index", scratch_index), (run_dir / "memory", scratch_memory)):
        if src.exists():
            shutil.copytree(src, dst)

    agent = build_resume_agent(store, scratch_index, scratch_memory, guidance,
                               scratch_store=scratch_store)

    # observability (optional, mirrors the harness)
    traced = False
    try:
        from paf.observability.instrumentation import setup_observability, shutdown_observability
        traced = setup_observability()
    except Exception:
        traced = False

    session = f"resume-{time.strftime('%Y%m%d-%H%M%S')}"
    tools: list[str] = []
    tool_outputs: list[str] = []
    answer = ""
    from pydantic_ai import AgentRunResultEvent
    from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent

    async def _run():
        nonlocal answer
        async with agent.run_stream_events(
            prompt, message_history=messages_for_replay, conversation_id=conv_id
        ) as es:
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
                    answer = e.result.output
    await _run()

    print(f"\ntools fired: {tools or '(none)'}")
    print(f"model answer: {answer[:300]!r}")

    # ---- question id / gold for judging (judge call happens in sync main) ----
    q = None
    if args.question_id and DATA.exists():
        data = json.loads(DATA.read_text())
        q = next((x for x in data if str(x.get("question_id")) == args.question_id), None)
        if q is None:
            print(f"WARNING: --question-id {args.question_id!r} not found in {DATA}; skipping judge.")

    if traced:
        try:
            shutdown_observability()
        except Exception:
            pass
    return {"answer": answer, "tools": tools, "session": session, "q": q, "qid": args.question_id}


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume-replay a finished lmem run's recall turn.")
    parser.add_argument("--from-dir", required=True, help="run dir with store/ index/ memory/")
    parser.add_argument("--list", action="store_true", help="list conversations and exit")
    parser.add_argument("--conv", help="conversation_id to resume (default: most-runs conv)")
    parser.add_argument("--guidance", default="", help="guidance = reasoning variant system prompt")
    parser.add_argument("--dry-run", action="store_true", help="rehydrate+strip only, no LLM")
    parser.add_argument("--question-id", help="lme question_id for judging (optional)")
    args = parser.parse_args()

    result = asyncio.run(run_replay(args))

    # judge runs OUTSIDE the async loop (the harness _lme_judge uses run_sync).
    if result and isinstance(result, dict) and result.get("q"):
        from tests.run_lmem import _lme_judge
        q, answer, session = result["q"], result["answer"], result["session"]
        verdict = _lme_judge(
            q["question_type"], q["question"], str(q.get("answer", "")),
            answer, result["qid"], span_name="resume.judge", session=session,
        )
        print(f"\njudge: {'CORRECT' if verdict['correct'] else 'WRONG'} "
              f"(type={q['question_type']}, raw={verdict['raw']!r})")
        print(f"  reasoning: {verdict.get('reasoning', '')}")
        print(f"prompt: {q['question'][:80]!r}")
        return 0 if bool(verdict["correct"]) else 1

    # dry-run / no question-id: no judge performed.
    if result is not None and isinstance(result, int):
        return result
    print("done (no judge: pass --question-id to grade the response).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())