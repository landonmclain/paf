"""Code-search eval runner — benchmark retrieval against a gold set (Phase 0 shell).

Measures `RetrievalBackend.search` against the hand-curated gold set in
`queries.jsonl`. Each query has a gold `answer_files` / `answer_symbols` list; we score
the returned `SearchHit`s with **recall@k, precision@k, MRR@k, nDCG@k** — the same
ID-membership spirit as the repo's existing `tests/retrieval_scorer.py` recall@k.

The **gate** (from Codebase-Map-Capability-Spec §3.7.5): any future retrieval method
(including a semantic backend) must beat the BM25 baseline on this set by a decided
margin on the headline metric, or it is not adopted. No metrics, no adoption.

Not yet wired to a real backend in Phase 0 (search is Phase 3); the scorer is the
"fixed target" the map will be judged against. Run with `--queries` to point at a
`queries.jsonl`; the default emits an empty table until Phase 3 plugs in a backend.

Usage:
    PYTHONPATH= .venv/bin/python -m tests.code_search_eval.eval --queries tests/code_search_eval/queries.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Pluggable backend — Phase 3 substitutes a real RetrievalBackend here.
_BACKEND = None  # set by main() when a backend is injected (Phase 3)


def _parse_hits(hits: list) -> tuple[list[str], list[str]]:
    """Flatten a hit list into (files, symbols) for scoring. Hit = SearchHit-like."""
    files, symbols = [], []
    for h in hits:
        files.append(getattr(h, "file", h.get("file") if isinstance(h, dict) else None))
        sym = getattr(h, "symbol", h.get("symbol") if isinstance(h, dict) else None)
        if sym:
            symbols.append(sym)
    return files, symbols


def _hits_at_k(hits: list, k: int) -> list:
    return hits[:k]


def recall_at_k(gold_files: set[str], hits: list, k: int) -> float:
    shown = {f for f, _ in _parse_hits(_hits_at_k(hits, k)) if f}
    if not gold_files:
        return 1.0
    return len(shown & gold_files) / len(gold_files)


def precision_at_k(gold_files: set[str], hits: list, k: int) -> float:
    shown = {f for f, _ in _parse_hits(_hits_at_k(hits, k)) if f}
    return len(shown & gold_files) / max(len(shown), 1)


def mrr(gold_files: set[str], hits: list) -> float:
    for i, (f, _) in enumerate(_parse_hits(hits)):
        if f in gold_files:
            return 1.0 / (i + 1)
    return 0.0


def ndcg(gold_files: set[str], hits: list, k: int) -> float:
    """nDCG@k — graded relevance over gold membership (binary grade)."""
    dcg = 0.0
    for i, (f, _) in enumerate(_parse_hits(_hits_at_k(hits, k))):
        if f in gold_files:
            dcg += 1.0 / math.log2(i + 2)
    if not dcg:
        return 0.0
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_files), k)))
    return dcg / ideal if ideal else 0.0


def score_query(query: dict, hits: list, k: int = 10) -> dict:
    gold_files = set(query.get("answer_files", []))
    return {
        "recall@k": recall_at_k(gold_files, hits, k),
        "precision@k": precision_at_k(gold_files, hits, k),
        "mrr": mrr(gold_files, hits),
        "ndcg@k": ndcg(gold_files, hits, k),
    }


def load_queries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(queries_path: Path, backend, k: int = 10) -> list[dict]:
    queries = load_queries(queries_path)
    rows = []
    for q in queries:
        hits = backend.search(q["query"], mode=q.get("mode", "keywords"), limit=q.get("k", k))
        score = score_query(q, hits, k)
        rows.append({"query": q["query"], **score, "n_hits": len(hits)})
    return rows


def _fmt_table(rows: list[dict]) -> str:
    if not rows:
        return "(no queries scored — plug a RetrievalBackend in Phase 3)"
    header = f"{'query':<34} {'r@10':>6} {'p@10':>6} {'MRR':>6} {'nDCG':>6}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['query'][:33]:<34} {r['recall@k']:>6.2f} {r['precision@k']:>6.2f} "
            f"{r['mrr']:>6.2f} {r['ndcg@k']:>6.2f}"
        )
    # averages row
    if rows:
        keys = ["recall@k", "precision@k", "mrr", "ndcg@k"]
        avg = {c: sum(r[c] for r in rows) / len(rows) for c in keys}
        lines.append("-" * len(header))
        lines.append(
            f"{'AVERAGE':<34} {avg['recall@k']:>6.2f} {avg['precision@k']:>6.2f} "
            f"{avg['mrr']:>6.2f} {avg['ndcg@k']:>6.2f}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Code-search eval: score retrieval against a gold set.")
    parser.add_argument("--queries", default="tests/code_search_eval/queries.jsonl")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    if _BACKEND is None:
        print(_fmt_table([]))
        print("\nNOTE: no RetrievalBackend wired (search is Phase 3). Queries file exists at:")
        print(f"  {args.queries}")
        return 0

    rows = evaluate(Path(args.queries), _BACKEND, k=args.k)
    print(_fmt_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
