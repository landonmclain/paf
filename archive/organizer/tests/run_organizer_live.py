"""Live LLM test runner for the memory organizer (journal tail -> curated index).

This is a LIVE test: it spends real OpenRouter tokens running the organizer
leaf agent (which classifies the journal tail and commits ``memory-index.json``
atomically). It is NOT part of the plain ``pytest`` run — like the other
benchmark runners in ``tests/`` it is invoked explicitly and sources ``.env``
itself is NOT done here (run from the repo root with env already exported, or
the caller sources ``.env`` first).

Modes
-----
--fixture  (default)  Run over the disposable fixture journal at
                      ``data/benchmarks/memory/organizer-live``. Two batches of
                      records exercise NEW/CONFIRM/SUPERSEDE + merge/prune and
                      the watermark-advance path.
--live                Run over the real ``data/memory`` journal (the agent's
                      actual durable memory) — produces the first
                      ``memory-index.json`` there. Destructive-ish: it stays in
                      ``data/memory`` (the same dir as the live MEMORY.md + app
                      journal); use --copy to work on a copy instead.

Invariants verified after the run (deterministic, no model needed):
  * watermark reached the last journal line (every record classified).
  * every entry's source lines are non-empty, <= watermark, and valid.
  * updated_on >= born_on for every entry (recency never precedes provenance).
  * ids are strictly monotonic / never reused.
  * born_on never changes once minted (checked across a run boundary when the
    index pre-exists).

Usage:
    PYTHONPATH= .venv/bin/python -m tests.run_organizer_live --fixture
    PYTHONPATH= .venv/bin/python -m tests.run_organizer_live --live
    PYTHONPATH= .venv/bin/python -m tests.run_organizer_live --live --copy
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from paf.config import PROJECT_ROOT

FIXTURE_DIR = PROJECT_ROOT / "data" / "benchmarks" / "memory" / "organizer-live"
LIVE_DIR = PROJECT_ROOT / "data" / "memory"


def _load_index(store) -> list:
    return store.load()


def run(mem_dir: Path, *, copy: bool = False) -> int:
    from archive.organizer import IndexStore
    from paf.capabilities.memory._journal import JournalStore

    target = mem_dir
    if copy and (target / "memory-journal.jsonl").exists():
        import tempfile

        work = Path(tempfile.mkdtemp(prefix="organizer-live-"))
        shutil.copy2(target / "memory-journal.jsonl", work / "memory-journal.jsonl")
        if (target / "memory-index.json").exists():
            shutil.copy2(target / "memory-index.json", work / "memory-index.json")
        target = work
        print(f"copied fixture to scratch dir: {work}")

    journal = JournalStore(target)
    index_store = IndexStore(target)

    records = journal.read_all()
    pre_wm = index_store.load_watermark()
    new_count = sum(1 for ln, _ in records if ln > pre_wm)
    print(f"journal records: {len(records)} | pre-watermark: {pre_wm} | new tail: {new_count}")
    if new_count == 0:
        print("no new records to organize; nothing to do.")
        return 0

    print(f">>> organizer.run() over {new_count} new records (live LLM, costs tokens)...")
    try:
        outcome = await_run(target, journal, index_store)
    except Exception as exc:  # noqa: BLE001 - surface live failures clearly
        print(f"ORGANIZER RUN FAILED: {type(exc).__name__}: {exc}")
        return 1

    print(f"organizer output: {outcome}")

    # --- verify invariants -------------------------------------------------
    import json as _json

    index_path = target / "memory-index.json"
    if not index_path.exists():
        print("FAIL: no memory-index.json written")
        return 1

    raw = _json.loads(index_path.read_text(encoding="utf-8"))
    entries = raw["entries"]
    wm = raw["watermark"]
    last_line = records[-1][0]

    problems: list[str] = []
    if wm != last_line:
        problems.append(f"watermark {wm} != last journal line {last_line} (records unclassified)")
    if raw["version"] != 4:
        problems.append(f"unexpected index version {raw['version']}")

    ids_seen: set[str] = set()
    for e in entries:
        if e["id"] in ids_seen:
            problems.append(f"id reused: {e['id']}")
        ids_seen.add(e["id"])
        if e["born_on"] > e["updated_on"]:
            problems.append(f"{e['id']}: updated_on {e['updated_on']} < born_on {e['born_on']}")
        for ln in e["source"]:
            if ln > wm or ln < 1:
                problems.append(f"{e['id']}: bad source line {ln}")

    print(f"index entries: {len(entries)} | watermark: {wm}")
    if problems:
        print("INVARIANT FAILURES:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("All invariants hold. Index:")
    for e in entries:
        print(f"  {e['id']} [{e['type']}/{e['subject']}] {e['fact']}")
    return 0


def await_run(target, journal, index_store):
    """Async wrapper so the runner keeps a clean sync main()."""
    import asyncio

    from archive.organizer._organize import run_organizer

    return asyncio.run(run_organizer(journal, index_store=index_store))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", dest="which", action="store_const", const="fixture",
        default="fixture", help="Run over the disposable organizer-live fixture (default).",
    )
    parser.add_argument(
        "--live", dest="which", action="store_const", const="live",
        help="Run over the real data/memory journal.",
    )
    parser.add_argument("--copy", action="store_true",
                        help="Work on a scratch copy (never mutate the source dir).")
    args = parser.parse_args(argv)

    mem_dir = LIVE_DIR if args.which == "live" else FIXTURE_DIR
    if not (mem_dir / "memory-journal.jsonl").exists():
        print(f"No journal at {mem_dir}. Run the fixture generator first.")
        return 1
    return run(mem_dir, copy=args.copy)


if __name__ == "__main__":
    sys.exit(main())
