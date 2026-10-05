"""Two-stage judge, all on OpenRouter: Jev screens, DeepSeek confirms.

Same process as ``two_stage.py`` with no local GPU:

Stage 1 -- Jev (``typesafe/jev-1.13``, a System One decision model) labels every
           pair via OpenRouter's ``/api/v1/systemone`` (same request shape as
           the local Ollama ``/v1/systemone``). Output tokens are free; each
           response carries ``usage.cost``.
Stage 2 -- ONLY the pairs Jev called obsolete (same / update) are escalated to
           the production reviewer (the ``memory_pair_reviewer`` spec: DeepSeek
           + PairReview), one pair per call.
Final  -- a pair is obsolete only if BOTH stages say so; DeepSeek's label (and
           keep side) wins. Anything Jev passes over is ``different``.

Scored against the same DeepSeek reference as the local runs. NB the confirmer
IS the reference model, so stage-2 agreement is optimistic by construction; the
number that carries signal is stage 1's recall (what Jev lets through unseen).

Usage (from scripts/relate, .env sourced):
    PYTHONPATH= ../../.venv/bin/python two_stage_cloud.py --n 40
    PYTHONPATH= ../../.venv/bin/python two_stage_cloud.py --n 200 --screen typesafe/jev-latest
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from judge_bench import build_set
from reviewer_labels import (
    TEV1_CRITERIA,
    TEV1_KEEP,
    _append,
    _dir,
    _load,
    _obsolete,
    _pct,
    pair_prompt,
)
from two_stage import sample

from paf.capabilities.memory._pairwise import PairReview, _agent, _judge_spec
from paf.degeneration_guard import run_guarded

JEV_URL = "https://openrouter.ai/api/v1/systemone"
JEV_MODEL = "typesafe/jev-1.13"
WORKERS = 8
PAIR_TIMEOUT_SECONDS = 300


def jev_one(c: dict, model: str) -> dict:
    body = {
        "model": model,
        "state": {"fact_a": c["a"]["fact"], "fact_b": c["b"]["fact"]},
        "questions": {
            "relation": {
                "type": "choice",
                "instructions": "How does fact_b (newer) relate to fact_a (older)?",
                "criteria": TEV1_CRITERIA,
            },
            "keep": {
                "type": "choice",
                "instructions": "If they say the same thing, which fact should be kept?",
                "criteria": TEV1_KEEP,
            },
        },
    }
    req = urllib.request.Request(
        JEV_URL,
        json.dumps(body).encode(),
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        },
    )
    err = ""
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                r = json.load(resp)
            rel, keep = r["answers"]["relation"], r["answers"]["keep"]
            return {
                "id": c["id"],
                "label": rel["choice"],
                "confidence": rel.get("confidence"),
                "probabilities": rel.get("probabilities"),
                "keep": keep["choice"],
                "cost": (r.get("usage") or {}).get("cost", 0.0),
                "model": r.get("model", model),
            }
        except urllib.error.HTTPError as exc:
            err = f"{exc.code}: {exc.read()[:200].decode(errors='replace')}"
            if exc.code != 429 and exc.code < 500:
                break
        except Exception as exc:  # noqa: BLE001 - network blips retry
            err = f"{type(exc).__name__}: {exc}"
        time.sleep(2**attempt)
    return {"id": c["id"], "label": None, "error": err}


def run_screen(model: str, items: list, out: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    t = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        for row in pool.map(lambda it: jev_one(it[0], model), items):
            _append(out, row)
    print(f"  screen wall time {time.time() - t:.0f}s", flush=True)


async def run_confirm(items: list, out: Path) -> None:
    _m, _s, instructions, _sugg = _judge_spec()
    agent = _agent(PairReview, instructions)
    sem = asyncio.Semaphore(WORKERS)

    async def one(c: dict) -> dict:
        row: dict = {"id": c["id"]}
        async with sem:
            try:
                result = await asyncio.wait_for(
                    run_guarded(agent, pair_prompt(c)), PAIR_TIMEOUT_SECONDS
                )
                js = result.output.judgements
                j = next((x for x in js if x.index == 1), js[0] if js else None)
                if j is None:
                    raise ValueError("no judgement")
                row |= {"label": j.label, "keep": j.keep if j.label == "same" else None}
            except Exception as e:  # noqa: BLE001 - an invalid output is a result too
                row |= {"label": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        return row

    for row in await asyncio.gather(*(one(c) for c, _ref in items)):
        _append(out, row)


def main() -> int:
    ap = argparse.ArgumentParser(description="Jev screen + DeepSeek escalation.")
    ap.add_argument("--agent", default="demo_assistant")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--screen", default=JEV_MODEL, help="OpenRouter decision model id.")
    args = ap.parse_args()

    d = _dir(args.agent)
    items = sample(build_set(d, contested=False, full=True), args.n)
    out = d / "two-stage-cloud"
    out.mkdir(exist_ok=True)
    s_file, c_file = out / "screen.jsonl", out / "confirm.jsonl"
    for f in (s_file, c_file):
        f.unlink(missing_ok=True)

    t0 = time.time()
    print(f"stage 1: {args.screen} on {len(items)} pairs", flush=True)
    run_screen(args.screen, items, s_file)
    screen = {r["id"]: r for r in _load(s_file)}
    failed = [pid for pid, r in screen.items() if not r.get("label")]
    # A failed screen call can't be trusted as "different": escalate it.
    flagged = [
        (c, r)
        for c, r in items
        if not screen[c["id"]].get("label") or _obsolete(screen[c["id"]]["label"])
    ]
    t1 = time.time()
    cost = sum(r.get("cost") or 0 for r in screen.values())
    print(
        f"stage 1: {t1 - t0:.0f}s, ${cost:.5f}, flagged {len(flagged)}/{len(items)} "
        f"({len(failed)} failed calls escalated)",
        flush=True,
    )

    asyncio.run(run_confirm(flagged, c_file))
    confirm = {r["id"]: r for r in _load(c_file)}
    t2 = time.time()
    print(
        f"stage 2 ({_judge_spec()[0]}): {len(flagged)} escalated in {t2 - t1:.0f}s",
        flush=True,
    )

    rows = []
    for c, ref in items:
        pid = c["id"]
        s, k = screen[pid], confirm.get(pid)
        label, keep = "different", None
        if k is not None and k.get("label") and _obsolete(k["label"]):
            label, keep = k["label"], k.get("keep")
        rows.append(
            {
                "id": pid,
                "screen": s.get("label"),
                "confidence": s.get("confidence"),
                "confirm": k.get("label") if k else None,
                "final": label,
                "keep": keep,
                "ref": ref["label"],
                "ref_keep": ref.get("keep"),
            }
        )
    (out / "result.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def stats(label_of) -> tuple[int, int, int, int]:
        tp = sum(_obsolete(label_of(r)) and _obsolete(r["ref"]) for r in rows)
        fp = sum(_obsolete(label_of(r)) and not _obsolete(r["ref"]) for r in rows)
        fn = sum(not _obsolete(label_of(r)) and _obsolete(r["ref"]) for r in rows)
        ok = sum(label_of(r) == r["ref"] for r in rows)
        return tp, fp, fn, ok

    print(
        f"\n{len(rows)} pairs ({sum(_obsolete(r['ref']) for r in rows)} truly obsolete per the DeepSeek reference)"
    )
    for name, fn_ in (
        ("Jev alone", lambda r: r["screen"]),
        ("Jev+DeepSeek", lambda r: r["final"]),
    ):
        tp, fp, fn, ok = stats(fn_)
        prec = f"{tp / (tp + fp):.0%}" if tp + fp else "n/a"
        rec = f"{tp / (tp + fn):.0%}" if tp + fn else "n/a"
        print(
            f"  {name:13} exact {_pct(ok, len(rows))} | flagged {tp + fp}: {tp} right, {fp} wrong, {fn} missed | precision {prec} recall {rec}"
        )
    keep_rows = [
        r for r in rows if r["final"] == "same" and r["ref"] == "same" and r["ref_keep"]
    ]
    print(
        f"  keep side (confirmed same): {_pct(sum(r['keep'] == r['ref_keep'] for r in keep_rows), len(keep_rows))}"
    )
    print(
        f"  escalation rate {len(flagged) / len(rows):.0%}; total {t2 - t0:.0f}s; rows in {out}/result.jsonl"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
