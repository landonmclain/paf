"""Snapshot the durable seed summaries from a finished run into a COMMITTED,
question-keyed store: data/benchmarks/generated/<question_id>.jsonl

The seed summaries are expensive to generate (real TurnSummarizer LLM calls on
every LongMemEval haystack turn) and STABLE (deterministic given the same seed
data + model). They currently live only under `tests/results/<run>/*`, which is
gitignored and cleaned between runs — so every run regenerates them from
scratch (~45 summarizer calls for a 4-Q run). This script lifts them out into a
durable, diffable, reviewable location so re-runs / resume work can reuse them.

WHAT it captures: ONLY the `seed-*` rows (run_id.startswith('seed-')) — the
per-turn one-liner index the summarizer produced. It does NOT capture the
recall-run rows (the agent's own per-turn self-summaries), which are run-
ephemeral and not stable.

Mapping conversation -> question: the runner processes questions in order and
gives each a fresh `lmem-<8hex>` conversation id. The script takes the explicit
question order (--qids) that the run used, and assigns seed conversations to
questions by first-appearance order. Each row is rewritten with a STABLE
signature: {question_id, turn_id, summary, key_terms, outcome} — the ephemeral
conversation_id / run_id / ts / hash are dropped (they're run-local noise).

Run (from repo root):
    PYTHONPATH= .venv/bin/python tests/snapshot_seed_index.py \
        --from tests/results/lmem-20260812-083703 \
        --qids 6a1eabeb 0a995998 7161e7e2 8a2466db
Exit 0 on success. Idempotent: overwrites the per-question JSONL in place.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "benchmarks" / "generated"


def _find_index_jsonl(run_dir: Path) -> Path:
    hits = list(run_dir.rglob("turn_index.jsonl"))
    if not hits:
        raise SystemExit(f"no turn_index.jsonl under {run_dir}")
    return hits[0]


def _seed_rows_by_conversation(index_path: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    """Group seed rows by conversation, preserving first-appearance order."""
    by_conv: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not str(row.get("run_id", "")).startswith("seed-"):
            continue  # recall-run self-summaries are not stable
        conv = row["conversation_id"]
        by_conv.setdefault(conv, []).append(row)
    return by_conv


def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot durable seed summaries to a committed store.")
    ap.add_argument("--from", dest="run_dir", required=True, help="run dir containing index/turn_index.jsonl")
    ap.add_argument("--qids", nargs="+", required=True, help="question_ids in run order (matches conv order)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    index_path = _find_index_jsonl(run_dir)
    by_conv = _seed_rows_by_conversation(index_path)

    # The runner assigns conversation ids in the order questions are processed,
    # so conv order == qid order. Guard on count.
    if len(by_conv) != len(args.qids):
        print(
            f"WARNING: {len(by_conv)} seed conversations but {len(args.qids)} qids "
            f"-- mapping may be wrong; check --qids order.",
            file=sys.stderr,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for (_, rows), qid in zip(by_conv.items(), args.qids, strict=True):
        assert isinstance(rows, list), "seed row group must be a list"
        # order rows by turn_id so the file reads as the transcript does
        rows.sort(key=lambda r: int(r.get("turn_id", 0)))
        out = OUT_DIR / f"{qid}.jsonl"
        with out.open("w", encoding="utf-8") as fp:
            for r in rows:
                sig = {
                    "question_id": qid,
                    "turn_id": r.get("turn_id"),
                    "summary": r.get("summary", ""),
                    "key_terms": r.get("key_terms"),
                    "outcome": r.get("outcome"),
                }
                fp.write(json.dumps(sig, ensure_ascii=False) + "\n")
        written += len(rows)
        print(f"  wrote {out.name} ({len(rows)} seed summaries)")

    print(f"\nOK: {written} durable seed summaries -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
