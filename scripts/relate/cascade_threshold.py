"""Per-pair Jev vs DeepSeek on saved journals: find the escalation rule.

For every (new fact, older candidate) the reviewer would compare, record DeepSeek's
label and Jev's label/probabilities, then sweep escalation rules. A rule
"escalates" a pair; DeepSeek's same/update labels on pairs a rule does NOT
escalate are the facts the cascade would hide.

Usage (repo root, .env sourced):
    PYTHONPATH= .venv/bin/python scripts/relate/cascade_threshold.py \\
        data/benchmarks/memory-v2-lme_oracle data/benchmarks/memory-lme_oracle_tier --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path

from paf.capabilities.memory import _cascade
from paf.capabilities.memory._pairwise import deepseek_judge, review_new_facts

ROWS: list[dict] = []
_CALLS = itertools.count()


async def recording_judge(new, older):
    call = next(_CALLS)
    model, _ = _cascade._screen_config()
    ds_task = asyncio.create_task(deepseek_judge(new, older))
    try:
        ans = (await _cascade._post_screen(_cascade.screen_request(model, new, older)))[
            "answers"
        ]
    except Exception:  # noqa: BLE001
        ans = None
    ds = await ds_task
    labels = {j.index: j.label for j in ds}
    for i in range(1, len(older) + 1):
        rel = ans[f"relation_{i}"] if ans else None
        ROWS.append(
            {
                "call": call,
                "ds": labels.get(i, "different"),
                "jev": rel["choice"] if rel else None,
                "p": rel["probabilities"] if rel else None,
                "conf": rel.get("confidence") if rel else None,
            }
        )
    return ds


async def one(qdir: Path, sem: asyncio.Semaphore) -> None:
    if not (qdir / "memory-journal.jsonl").exists():
        return
    async with sem:
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(qdir / "memory-journal.jsonl", tmp)
            try:
                await review_new_facts(
                    tmp, dry_run=True, start_after=0, judge=recording_judge
                )
            except KeyError:
                pass


def report() -> None:
    n = len(ROWS)
    pos = [r for r in ROWS if r["ds"] in ("same", "update")]
    print(f"\n{n} candidate pairs; DeepSeek same/update on {len(pos)}")
    print(f"Jev label on those {len(pos)}: ", end="")
    for lab in ("same", "update", "different"):
        print(f"{lab}={sum(r['jev'] == lab for r in pos)} ", end="")
    print()
    print("\nrule                         escalated   hidden DS-positives")
    rules = {
        "label != different": lambda r: r["jev"] != "different",
    }
    for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999):
        rules[f"+ P(different) < {t}"] = lambda r, t=t: (
            r["jev"] != "different" or r["p"] is None or r["p"].get("different", 0) < t
        )
        rules[f"+ conf < {t}"] = lambda r, t=t: (
            r["jev"] != "different" or float(r["conf"] or 0) < t
        )
    calls: dict[int, list[dict]] = {}
    for r in ROWS:
        calls.setdefault(r["call"], []).append(r)
    print(
        f"\n{len(calls)} reviewer calls; a call goes to DeepSeek if ANY candidate escalates"
    )
    print(
        "rule                         pairs esc.  calls to DeepSeek  hidden DS-positives"
    )
    for name, esc in rules.items():
        e = sum(esc(r) for r in ROWS)
        hidden = sum(not esc(r) for r in pos)
        c = sum(any(esc(r) for r in rs) for rs in calls.values())
        print(
            f"{name:28} {e / n:6.0%}   {c:5} ({c / len(calls):4.0%})   {hidden}/{len(pos)}"
        )
    print("\nJev P(different) on the DeepSeek positives it called different:")
    for r in pos:
        if r["jev"] == "different":
            print(f"  ds={r['ds']:6} conf={r['conf']} p={json.dumps(r['p'])}")


async def main_async(args) -> None:
    sem = asyncio.Semaphore(args.workers)
    qdirs = [
        q for root in args.roots for q in sorted(Path(root).iterdir())[: args.limit]
    ]
    await asyncio.gather(*(one(q, sem) for q in qdirs))
    report()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    asyncio.run(main_async(ap.parse_args()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
