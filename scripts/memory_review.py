"""Run the pairwise memory reviewer by hand — dry run by default.

Proposes (or, with --apply, writes) the reviewer's actions for an agent's
memory and prints each one with both facts, for a spot-check before the
reviewer runs on its own.

Usage (from the repo root):
    set -a && . ./.env && set +a
    .venv/bin/python scripts/memory_review.py --last 30          # dry run
    .venv/bin/python scripts/memory_review.py --apply           # from watermark
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import textwrap

from paf.capabilities.memory._journal import JournalStore
from paf.capabilities.memory._pairwise import review_new_facts
from paf.capabilities.memory._window import memory_dir_for


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent", default="demo_assistant")
    ap.add_argument("--last", type=int, help="review the last N journal lines")
    ap.add_argument("--apply", action="store_true", help="write the actions")
    ap.add_argument("--top-k", type=int, default=12)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    directory = str(memory_dir_for(args.agent))
    journal = JournalStore(directory)
    records = dict(journal.read_all())
    start = journal.count() - args.last if args.last else None
    report = asyncio.run(
        review_new_facts(
            directory,
            dry_run=not args.apply,
            start_after=start,
            top_k=args.top_k,
            agent_name=args.agent,
        )
    )

    def fact(line: int | None) -> str:
        if line is None or line not in records:
            return ""
        r = records[line]
        return textwrap.shorten(f"#{line} ({r.born_on[:10]}) {r.fact}", 160)

    for a in report.actions:
        print(f"\n{a.op.upper():9s} #{a.target}  ({a.label}) — {a.reason}")
        print(f"  target: {fact(a.target)}")
        if a.other is not None:
            print(f"  other:  {fact(a.other)}")
    print(f"\n{report.summary()}; watermark -> {report.watermark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
