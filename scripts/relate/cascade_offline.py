"""Offline fidelity check: cascade (Jev -> DeepSeek) vs DeepSeek-only reviewer.

Runs ``review_new_facts(dry_run=True, start_after=0)`` over saved memory
journals with BOTH judges and diffs the proposed actions. Nothing is written to
the benchmark snapshots (journals are copied to a scratch dir) and no extractor
tokens are spent -- only reviewer calls.

Metrics, with DeepSeek-only as the reference:
  agree     actions both proposed (same op + target)
  extra     cascade proposed, DeepSeek did not (cascade over-archives)
  missed    DeepSeek proposed, cascade did not (a fact the screen hid)
and the cascade's escalation rate + Jev cost.

Usage (repo root, .env sourced):
    PYTHONPATH= .venv/bin/python scripts/relate/cascade_offline.py \\
        data/benchmarks/memory-lme_oracle_journal data/benchmarks/memory-longmemeval_s_cleaned
    ... --min-confidence 0.8      # override the spec's screen_min_confidence
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

from paf.capabilities.memory import _cascade
from paf.capabilities.memory._pairwise import review_new_facts


async def actions(directory: str, judge) -> set[tuple[str, int]]:
    report = await review_new_facts(directory, dry_run=True, start_after=0, judge=judge)
    return {(a.op, a.target) for a in report.actions}


async def one_question(qdir: Path, cache: Path) -> dict | None:
    journal = qdir / "memory-journal.jsonl"
    if not journal.exists():
        return None
    key = cache / f"{qdir.parent.name}__{qdir.name}.json"
    ds: set[tuple[str, int]] | None = None
    if key.exists():
        ds = {(op, ln) for op, ln in json.loads(key.read_text())}
    if ds is None:
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(journal, tmp)
            try:
                ds = await actions(tmp, None)
            except KeyError:  # an older journal schema; not reviewable
                print(f"{qdir.name}: skipped (old journal schema)", flush=True)
                return None
        key.write_text(json.dumps(sorted(ds)))
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(journal, tmp)
        cs = await actions(tmp, _cascade.cascade_judge)
    n = sum(1 for _ in journal.open())
    print(
        f"{qdir.name}: {n} facts, deepseek {len(ds)} actions, cascade {len(cs)}, "
        f"missed {sorted(ds - cs)}, extra {sorted(cs - ds)}",
        flush=True,
    )
    return {
        "facts": n,
        "ds": len(ds),
        "agree": len(ds & cs),
        "extra": len(cs - ds),
        "missed": len(ds - cs),
    }


async def main_async(args) -> int:
    if args.min_confidence is not None:
        orig = _cascade._screen_config
        _cascade._screen_config = lambda: (orig()[0], args.min_confidence)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.workers)

    async def bounded(qdir: Path):
        async with sem:
            return await one_question(qdir, cache)

    qdirs = [
        q for root in args.roots for q in sorted(Path(root).iterdir())[: args.limit]
    ]
    results = [r for r in await asyncio.gather(*(bounded(q) for q in qdirs)) if r]
    keys = ("facts", "ds", "agree", "extra", "missed")
    tot = {k: sum(r[k] for r in results) for k in keys}
    print(
        f"\n{len(results)} questions, {tot['facts']} facts | deepseek actions "
        f"{tot['ds']} | agree {tot['agree']} | cascade extra {tot['extra']} | "
        f"missed {tot['missed']}"
    )
    print(_cascade.STATS.summary())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", help="dirs of <qid>/memory-journal.jsonl")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--cache",
        default="/tmp/claude-1000/-home-landon-projects-Personal-pydantic-agent-framework/12d0c514-98e4-40e7-a908-3b9f01e03a2b/scratchpad/ds-baseline",
        help="Dir caching the DeepSeek-only baseline per question.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Questions per root.")
    ap.add_argument("--min-confidence", type=float, default=None)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
