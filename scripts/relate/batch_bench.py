"""Benchmark judges on PRODUCTION-shaped batches: one new fact vs its top-12.

``judge_bench.py`` scores single pairs; the real reviewer (memory/_pairwise.py)
judges one new fact B against up to 12 older candidates in ONE call. This
rebuilds those batches from the agent's real memory journal and scores judges
on them:

- Batches: for each new fact B of the judge-bench consensus pairs, the BM25
  top-12 among older index-winner facts in B's scope (own project + global),
  as ``review_new_facts`` picks them. The labelled partner A is forced in when
  BM25 didn't rank it, so every batch carries a known answer. Read-only: uses
  the journal + index as ``mine_pairs.py`` does (``current_facts`` would
  update the index on disk).
- deepseek: the production judge (``deepseek_judge``). Its batched labels are
  the reference for every candidate; its answers on the labelled pairs show
  whether batching moves DeepSeek itself.
- chat:<model>[@effort]: the same production prompt + PairReview via Ollama.
- decision:<model>: ONE /v1/systemone call per batch (new fact + 12 older facts
  as state, a relation + keep question per candidate).

Usage (from scripts/relate, .env sourced):
    PYTHONPATH= ../../.venv/bin/python batch_bench.py deepseek chat:qwen3.8:27b decision:nimble:latest
    PYTHONPATH= ../../.venv/bin/python batch_bench.py --report
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from judge_bench import PAIR_TIMEOUT_SECONDS, _slug, build_set
from reviewer_labels import (
    TEV1_CRITERIA,
    TEV1_KEEP,
    _append,
    _dir,
    _load,
    _obsolete,
    _pct,
    _post,
)

from paf.capabilities.memory._index import MemoryIndex
from paf.capabilities.memory._journal import JournalStore
from paf.capabilities.memory._pairwise import (
    DEFAULT_TOP_K,
    Fact,
    PairReview,
    _Bm25,
    _judge_spec,
    _same_scope,
    deepseek_judge,
    pair_prompt,
)
from paf.config import PROJECT_ROOT
from paf.degeneration_guard import run_guarded

BATCH_TIMEOUT_SECONDS = 3 * PAIR_TIMEOUT_SECONDS


# --- batches ----------------------------------------------------------------------


def build_batches(agent: str) -> list[dict]:
    """``[{b, older: [lines], labelled: {a_line: pair_id}}]``, cached on disk."""
    d = _dir(agent)
    path = d / "batch-bench" / "batches.jsonl"
    if path.exists():
        return _load(path)
    mem = PROJECT_ROOT / "data" / "memory" / agent
    records = dict(JournalStore(mem).read_all())
    pool = {
        int(e["current_line"])
        for e in MemoryIndex.load(mem).entries
        if int(e["current_line"]) in records
        and not records[int(e["current_line"])].source_ref.startswith("merge(")
    }
    by_b: dict[int, dict[int, str]] = {}
    for c, _ref in build_set(d, contested=False):
        by_b.setdefault(c["b"]["line"], {})[c["a"]["line"]] = c["id"]
    bm = _Bm25({ln: records[ln] for ln in pool})
    batches = []
    for b, labelled in sorted(by_b.items()):
        B = records[b]
        eligible = [ln for ln in pool if ln < b and _same_scope(B, records[ln])]
        older = bm.top(B.fact, eligible, DEFAULT_TOP_K)
        for a in labelled:  # force the known answer in, replacing the weakest
            if a not in older:
                older = (
                    older[: DEFAULT_TOP_K - 1] if len(older) >= DEFAULT_TOP_K else older
                ) + [a]
        batches.append(
            {
                "b": b,
                "older": older,
                "labelled": {str(a): pid for a, pid in labelled.items()},
            }
        )
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in batches))
    return batches


def _facts(agent: str, batch: dict) -> tuple[Fact, list[Fact]]:
    records = _records(agent)
    return Fact(batch["b"], records[batch["b"]]), [
        Fact(ln, records[ln]) for ln in batch["older"]
    ]


_RECORDS: dict[str, dict] = {}


def _records(agent: str) -> dict:
    if agent not in _RECORDS:
        _RECORDS[agent] = dict(
            JournalStore(PROJECT_ROOT / "data" / "memory" / agent).read_all()
        )
    return _RECORDS[agent]


# --- judges -------------------------------------------------------------------------


def _row(batch: dict, older: list[Fact], judgements, seconds: float) -> dict:
    labels: dict[str, dict] = {}
    for j in judgements:
        if 1 <= j.index <= len(older):
            labels[str(older[j.index - 1].line)] = {
                "label": j.label,
                "keep": j.keep if j.label == "same" else None,
            }
    return {"b": batch["b"], "labels": labels, "seconds": round(seconds, 2)}


async def run_chat(
    agent: str, judge: str, batches: list[dict], out: Path, workers: int
) -> None:
    from pydantic_ai import Agent

    if judge == "deepseek":
        call = deepseek_judge
    else:
        _kind, _, rest = judge.partition(":")
        model, _, effort = rest.partition("@")
        _m, _s, instructions, _sugg = _judge_spec()
        settings: dict = {"temperature": 0}
        if effort:
            settings["openai_reasoning_effort"] = effort
        agent_ = Agent(
            f"ollama:{model}",
            name="batch-bench",
            instructions=instructions,
            output_type=PairReview,
            model_settings=settings,  # type: ignore[arg-type]
            retries=2,
        )

        async def call(new: Fact, older: list[Fact]):
            result = await run_guarded(agent_, pair_prompt(new, older))
            return result.output.judgements

    sem = asyncio.Semaphore(workers)

    async def one(batch: dict) -> None:
        new, older = _facts(agent, batch)
        async with sem:
            t = time.time()
            try:
                js = await asyncio.wait_for(call(new, older), BATCH_TIMEOUT_SECONDS)
                row = _row(batch, older, js, time.time() - t)
            except Exception as e:  # noqa: BLE001 - an invalid output is a result too
                row = {
                    "b": batch["b"],
                    "labels": {},
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "seconds": round(time.time() - t, 2),
                }
        _append(out, row)

    await asyncio.gather(*(one(b) for b in batches))


def run_decision(agent: str, model: str, batches: list[dict], out: Path) -> None:
    def crit(defs: dict, i: int) -> dict:
        # The pair criteria speak of fact_a (older) / fact_b (newer).
        return {
            k: v.replace("fact_b", "new_fact").replace("fact_a", f"older_{i}")
            for k, v in defs.items()
        }

    for batch in batches:
        new, older = _facts(agent, batch)
        state = {"new_fact": new.record.fact} | {
            f"older_{i}": f.record.fact for i, f in enumerate(older, 1)
        }
        questions: dict = {}
        for i in range(1, len(older) + 1):
            questions[f"relation_{i}"] = {
                "type": "choice",
                "instructions": f"How does new_fact relate to older_{i}?",
                "criteria": crit(TEV1_CRITERIA, i),
            }
            questions[f"keep_{i}"] = {
                "type": "choice",
                "instructions": f"If new_fact and older_{i} say the same thing, which should be kept?",
                "criteria": {
                    "new": crit(TEV1_KEEP, i)["new"],
                    "old": crit(TEV1_KEEP, i)["old"],
                },
            }
        t = time.time()
        try:
            ans = _post(
                "http://127.0.0.1:11434/v1/systemone",
                {"model": model, "state": state, "questions": questions},
                timeout=BATCH_TIMEOUT_SECONDS,
            )["answers"]
            labels = {}
            for i, f in enumerate(older, 1):
                rel = ans[f"relation_{i}"]
                labels[str(f.line)] = {
                    "label": rel["choice"],
                    "keep": ans[f"keep_{i}"]["choice"]
                    if rel["choice"] == "same"
                    else None,
                    "probabilities": {
                        k: round(v, 4) for k, v in rel["probabilities"].items()
                    },
                }
            row = {
                "b": batch["b"],
                "labels": labels,
                "seconds": round(time.time() - t, 2),
            }
        except Exception as e:  # noqa: BLE001
            row = {
                "b": batch["b"],
                "labels": {},
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "seconds": round(time.time() - t, 2),
            }
        _append(out, row)


# --- report -----------------------------------------------------------------------


def report(agent: str, batches: list[dict]) -> str:
    d = _dir(agent)
    out_dir = d / "batch-bench"
    pair_ref = {r["id"]: r for r in _load(d / "reviewer-reference.jsonl")}
    ds = {r["b"]: r for r in _load(out_dir / "deepseek.jsonl")}
    lines = [
        "# Batched (1 new fact vs top-12) judges",
        "",
        f"{len(batches)} batches, {sum(len(b['older']) for b in batches)} candidate judgements.",
        "`vs DeepSeek batched` scores every candidate against the production judge's",
        "own batched labels; `known pairs` scores only the labelled (A, B) pairs",
        "against the pairwise reference (DeepSeek consensus).",
        "",
        "| judge | valid batches | vs DeepSeek: label | obsolete | precision | recall | keep | known pairs: label | obsolete | s/batch |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in sorted(out_dir.glob("*.jsonl")):
        if f.name == "batches.jsonl":
            continue
        rows = {r["b"]: r for r in _load(f)}
        valid = [b for b in batches if rows.get(b["b"], {}).get("labels")]
        # vs DeepSeek batched (skip for deepseek itself)
        pairs = [
            (rows[b["b"]]["labels"].get(str(ln)), ds[b["b"]]["labels"].get(str(ln)))
            for b in batches
            if b["b"] in rows and b["b"] in ds and ds[b["b"]].get("labels")
            for ln in b["older"]
        ]
        pairs = [(j, r) for j, r in pairs if r is not None]
        if f.stem == "deepseek":
            vs = ["—"] * 5
        else:
            ok = sum(j is not None and j["label"] == r["label"] for j, r in pairs)
            tp = sum(
                j is not None and _obsolete(j["label"]) and _obsolete(r["label"])
                for j, r in pairs
            )
            fp = sum(
                j is not None and _obsolete(j["label"]) and not _obsolete(r["label"])
                for j, r in pairs
            )
            fn = sum(
                (j is None or not _obsolete(j["label"])) and _obsolete(r["label"])
                for j, r in pairs
            )
            obs = sum(
                j is not None and _obsolete(j["label"]) == _obsolete(r["label"])
                for j, r in pairs
            )
            keep_p = [
                (j, r)
                for j, r in pairs
                if j and j["label"] == "same" and r["label"] == "same" and r.get("keep")
            ]
            vs = [
                _pct(ok, len(pairs)),
                _pct(obs, len(pairs)),
                f"{tp / (tp + fp):.0%}" if tp + fp else "n/a",
                f"{tp / (tp + fn):.0%}" if tp + fn else "n/a",
                _pct(sum(j["keep"] == r["keep"] for j, r in keep_p), len(keep_p)),
            ]
        known = [
            (rows[b["b"]]["labels"].get(a), pair_ref[pid])
            for b in batches
            if b["b"] in rows
            for a, pid in b["labelled"].items()
            if pid in pair_ref
        ]
        k_label = sum(j is not None and j["label"] == r["label"] for j, r in known)
        k_obs = sum(
            j is not None and _obsolete(j["label"]) == _obsolete(r["label"])
            for j, r in known
        )
        secs = [rows[b["b"]]["seconds"] for b in batches if b["b"] in rows]
        lines.append(
            f"| {f.stem} | {_pct(len(valid), len(batches))} | "
            + " | ".join(vs)
            + f" | {_pct(k_label, len(known))} | {_pct(k_obs, len(known))} | "
            f"{statistics.median(secs):.1f} |"
            if secs
            else f"| {f.stem} | no rows |"
        )
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark judges on production-shaped batches."
    )
    ap.add_argument(
        "judges", nargs="*", help="deepseek, chat:<model>[@effort], decision:<model>"
    )
    ap.add_argument("--agent", default="demo_assistant")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent DeepSeek calls.")
    ap.add_argument(
        "--limit", type=int, default=None, help="Only the first N batches (smoke test)."
    )
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    batches = build_batches(args.agent)[: args.limit]
    out_dir = _dir(args.agent) / "batch-bench"
    print(f"{len(batches)} batches", flush=True)
    if not args.report:
        for judge in args.judges:
            out = out_dir / f"{_slug(judge)}.jsonl"
            done = {r["b"] for r in _load(out) if r.get("labels")}
            todo = [b for b in batches if b["b"] not in done]
            print(f"{judge}: {len(done)} done, {len(todo)} to judge", flush=True)
            t = time.time()
            if judge.startswith("decision:"):
                run_decision(args.agent, judge.partition(":")[2], todo, out)
            else:
                workers = args.workers if judge == "deepseek" else 1
                asyncio.run(run_chat(args.agent, judge, todo, out, workers))
            print(f"{judge}: finished in {time.time() - t:.0f}s", flush=True)
    print(report(args.agent, batches))
    return 0


if __name__ == "__main__":
    sys.exit(main())
