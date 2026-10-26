"""Re-file legacy (v1) memory facts under the v2 schema — dry run by default.

Labels the oldest not-yet-re-filed legacy facts (keep/drop, type, scope,
project) and prints each decision; with --apply, writes them to the override
sidecar (archive / reclassify — reversible, the journal is untouched).

Usage (from the repo root):
    set -a && . ./.env && set +a
    PYTHONPATH= .venv/bin/python scripts/memory_backfill.py --agent demo_assistant --limit 10
    PYTHONPATH= .venv/bin/python scripts/memory_backfill.py --agent demo_assistant --apply

--refile FROM:TO splits a project instead: each fact filed under FROM is asked
"is this specifically about TO?" and moved if so (TO needs a description).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import textwrap

from paf.capabilities.memory._backfill import backfill, refile
from paf.capabilities.memory._window import memory_dir_for


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent", default="demo_assistant")
    ap.add_argument("--limit", type=int, help="label at most N legacy facts")
    ap.add_argument("--ids", help="only these journal lines, e.g. 12,40,977")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--apply", action="store_true", help="write the overrides")
    ap.add_argument("--refile", metavar="FROM:TO", help="split a project instead")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    only = {int(x) for x in args.ids.split(",")} if args.ids else None

    if args.refile:
        source, _, target = args.refile.partition(":")
        rep = asyncio.run(
            refile(
                str(memory_dir_for(args.agent)),
                agent_name=args.agent,
                source=source,
                target=target,
                limit=args.limit,
                only=only,
                batch_size=args.batch,
                concurrency=args.concurrency,
                dry_run=not args.apply,
            )
        )
        for ln, r, why in rep.moved:
            print(f"\n#{ln}  MOVE -> {target}  — {why}")
            print(textwrap.indent(textwrap.shorten(r.fact, 220), "  "))
        if rep.unlabeled:
            print(f"\nunlabeled: {rep.unlabeled}")
        print(f"\n{rep.summary(source, target)}")
        return 0

    report = asyncio.run(
        backfill(
            str(memory_dir_for(args.agent)),
            agent_name=args.agent,
            limit=args.limit,
            only=only,
            batch_size=args.batch,
            concurrency=args.concurrency,
            dry_run=not args.apply,
        )
    )
    for d in report.decisions:
        r = d.fact.record
        old = f"{r.type}/{r.tier}, from {d.fact.hint or 'none'}"
        if d.op == "archive":
            new = "ARCHIVE"
        else:
            f = d.fields or {}
            where = f["scope"] if f["scope"] == "global" else f"project:{f['project']}"
            new = f"{f['type']}/{where}"
        print(f"\n#{d.fact.line}  [{old}] -> {new}  — {d.label.reason}")
        print(textwrap.indent(textwrap.shorten(r.fact, 220), "  "))
    if report.unlabeled:
        print(f"\nunlabeled: {report.unlabeled}")
    print(f"\n{report.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
