"""Clean two-phase interactive recall test for the Turn Index capability.

Runs against a REAL live model (OPENAI_API_KEY / OPENROUTER_API_KEY must be set),
on the clean conversation_id `clean-test`:
  phase  seed   -> plant distinct facts (writes turn-index rows via the live summarizer)
  phase  recall -> fresh process, same cid, ask only answerable from the seeded
                   index; then exercise the `search_index` tool explicitly.

The point is a single, readable, datapoint: run seed once, run recall once, and the
on-disk rows under data/.turn-index are exactly that conversation, no cross-talk.

Usage:
    PYTHONPATH="" .venv/bin/python -u tools/recall_test.py seed
    PYTHONPATH="" .venv/bin/python -u tools/recall_test.py recall
"""
from __future__ import annotations

import asyncio
import sys

from paf.runtime import Session

CID = "clean-test"


def main() -> int:
    phase = sys.argv[1]
    with Session("assistant", conversation_id=CID) as session:
        restored = asyncio.run(session.restore())
        print(f"[{phase}] conversation={CID} restored={restored}")
        if phase == "seed":
            out = session.run_sync(
                "Remember: my favorite color is teal and my dog is named Widget. "
                "Deploy target is the staging server blue-server-7.",
                conversation_id=CID,
            )
            print(f"[seed]  >> {out}")
            out = session.run_sync(
                "Also remember: my capybara is named Mango and the backup "
                "database is on rack C4.",
                conversation_id=CID,
            )
            print(f"[seed2] >> {out}")
        else:
            out = session.run_sync(
                "From memory (not from notes): what is my favorite color, my "
                "dog's name, and the deploy target? One concise sentence.",
                conversation_id=CID,
            )
            print(f"[recall] >> {out}")
            out = session.run_sync(
                "Use the search_index tool for 'teal' and report the "
                "matches verbatim.",
                conversation_id=CID,
            )
            print(f"[recall-tool] >> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())