"""Candidate-finding recall: does a method surface a known duplicate in its top-k?

Target pairs are the reviewer's "duplicates" (archive reasons citing another
line, and merge(#a,#b) source lists) — found by READING, not word matching, so
they test lexical and semantic finders fairly. Caveat: the reviewer over-merged,
so some target pairs are related-not-duplicate facts; absolute numbers are a
floor, comparisons between methods are what this is for.

A pair counts as found at k if either fact appears in the other's top-k.
Methods: BM25 (always), each --embed model alone (Ollama /api/embed, cosine),
and each model fused with BM25 by reciprocal rank fusion (RRF, k=60).

Usage (from the repo root):
    PYTHONPATH= .venv/bin/python scripts/relate/recall_bench.py
    PYTHONPATH= .venv/bin/python scripts/relate/recall_bench.py \\
        --embed nomic-embed-text-v2-moe:latest embeddinggemma:latest
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

from mine_pairs import Bm25Index, load_corpus, reviewer_pairs

from paf.config import PROJECT_ROOT

KS = (1, 5, 10, 20)

# Same-prefix-both-sides conventions for symmetric similarity, per model card.
# (Qwen recommends the instruction on the query side only; applying it to both
# sides likely handicapped qwen3-embedding:0.6b in the 2026-09-30 run.)
PREFIXES = {
    "nomic-embed-text:latest": "clustering: ",
    "nomic-embed-text-v2-moe:latest": "search_query: ",
    "embeddinggemma:latest": "task: sentence similarity | query: ",
    "snowflake-arctic-embed2:latest": "",
    "qwen3-embedding:0.6b": "Instruct: Find memory facts that state the same claim\nQuery: ",
    "qwen3-embedding:4b": "Instruct: Find memory facts that state the same claim\nQuery: ",
}


def embed(model: str, texts: list[str]):
    import numpy as np

    prefix = PREFIXES.get(model, "")
    vecs: list[list[float]] = []
    for s in range(0, len(texts), 64):
        body = {"model": model, "input": [prefix + t for t in texts[s : s + 64]]}
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/embed",
            json.dumps(body).encode(),
            {"content-type": "application/json"},
        )
        vecs += json.load(urllib.request.urlopen(req, timeout=900))["embeddings"]
    v = np.array(vecs, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Duplicate-candidate recall benchmark.")
    ap.add_argument("--agent", default="demo_assistant")
    ap.add_argument("--embed", nargs="*", default=[], help="Ollama embedding models")
    args = ap.parse_args()

    d = PROJECT_ROOT / "data" / "memory" / args.agent
    records, lines = load_corpus(d, 400)
    pos = {ln: i for i, ln in enumerate(lines)}
    texts = [records[ln].fact for ln in lines]
    pairs = list(
        {(min(a, b), max(a, b)) for a, b, _ in reviewer_pairs(d, records, set(lines))}
    )
    qs = sorted({pos[x] for p in pairs for x in p})

    def recall(ranked: dict[int, list[int]]) -> dict[int, float]:
        return {
            k: sum(
                (pos[b] in ranked[pos[a]][:k]) or (pos[a] in ranked[pos[b]][:k])
                for a, b in pairs
            )
            / len(pairs)
            for k in KS
        }

    def fmt(r: dict[int, float]) -> str:
        return "  ".join(f"@{k}:{r[k]:.1%}" for k in KS)

    bm = Bm25Index(texts)
    bm_rank = {i: [j for j, _ in bm.neighbors(i, 50)] for i in qs}
    print(f"{len(pairs)} reviewer pairs, {len(texts)} facts")
    print(f"{'bm25':34s} {fmt(recall(bm_rank))}", flush=True)

    for model in args.embed:
        t = time.time()
        V = embed(model, texts)
        secs = time.time() - t
        sims = V[qs] @ V.T
        emb_rank: dict[int, list[int]] = {}
        for row, i in enumerate(qs):
            s = sims[row]
            s[i] = -2.0
            emb_rank[i] = [int(j) for j in s.argsort()[::-1][:50]]
        fused: dict[int, list[int]] = {}
        for i in qs:
            sc: dict[int, float] = {}
            for ranking in (bm_rank[i], emb_rank[i]):
                for r, j in enumerate(ranking):
                    sc[j] = sc.get(j, 0.0) + 1 / (60 + r)
            fused[i] = sorted(sc, key=lambda j: -sc[j])
        per_fact = secs / len(texts) * 1000
        print(f"{model:34s} {fmt(recall(emb_rank))}   ({per_fact:.0f}ms/fact)")
        print(f"{'  + bm25 (RRF)':34s} {fmt(recall(fused))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
