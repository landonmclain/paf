"""Mine candidate fact pairs for the same/update/related/unrelated dataset.

Picks REAL pairs from an agent's memory journal (no model involved) so a
labeler (DeepSeek, then a human pass) can label them, and a decision model
(tev1, Laya) can be scored against the labels. Labels are NOT assigned here —
the BM25 ratio band and the source are sampling aids only.

Sources (pairs keyed by (min line, max line); A is always the older fact):
  bm25     — BM25 nearest neighbors over all index winners, sampled by
             ratio band (score(A→B) / score(A→A)): high >0.6, mid 0.25–0.6,
             low <0.25.
  update   — high-overlap neighbor pairs whose texts differ mainly in numbers
             or one or two swapped words (the rare `update` label).
  reviewer — reviewer archive reasons that cite another line ("Duplicate of
             kept #81") and merge(#a,#b) source lists.
  random   — uniform random pairs (easy `unrelated`).

The corpus is every index winner, INCLUDING reviewer-archived ones (the
archived duplicates are exactly the same/update cases we want), minus
reviewer merge rows and facts over --max-chars.

Usage (from the repo root):
    PYTHONPATH= .venv/bin/python scripts/relate/mine_pairs.py [--agent demo_assistant]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from paf.capabilities.memory._index import MemoryIndex
from paf.capabilities.memory._journal import JournalStore
from paf.capabilities.search._ranking import Bm25Params, bm25_score, tokenize
from paf.config import PROJECT_ROOT

HIGH, LOW = 0.6, 0.25


def load_corpus(directory: Path, max_chars: int):
    """All journal records by line, plus the index-winner lines eligible as pairs."""
    records = dict(JournalStore(directory).read_all())
    index = MemoryIndex.load(directory)
    lines = [
        int(e["current_line"])
        for e in index.entries
        if int(e["current_line"]) in records
        and not records[int(e["current_line"])].source_ref.startswith("merge(")
        and len(records[int(e["current_line"])].fact) <= max_chars
    ]
    return records, sorted(lines)


class Bm25Index:
    """BM25 over a fixed corpus: idf computed once, postings to skip zero docs."""

    def __init__(self, texts: list[str]) -> None:
        self.docs = [tokenize(t) for t in texts]
        n = len(self.docs)
        self.avgdl = sum(map(len, self.docs)) / n
        self.postings: dict[str, set[int]] = defaultdict(set)
        for i, toks in enumerate(self.docs):
            for t in set(toks):
                self.postings[t].add(i)
        self.idf = {
            t: math.log((n - len(ds) + 0.5) / (len(ds) + 0.5) + 1)
            for t, ds in self.postings.items()
        }
        self.params = Bm25Params()

    def neighbors(self, i: int, k: int) -> list[tuple[int, float]]:
        """Top-k other docs by score(i→j) / score(i→i)."""
        q = list(dict.fromkeys(self.docs[i]))
        self_score = bm25_score(q, self.docs[i], self.idf, self.avgdl, self.params)
        if self_score <= 0:
            return []
        cands = set().union(*(self.postings[t] for t in q)) - {i}
        scored = [
            (
                j,
                bm25_score(q, self.docs[j], self.idf, self.avgdl, self.params)
                / self_score,
            )
            for j in cands
        ]
        return sorted(scored, key=lambda x: -x[1])[:k]


def band(ratio: float) -> str:
    return "high" if ratio > HIGH else "mid" if ratio >= LOW else "low"


_NUM = re.compile(r"\d")


def looks_like_update(a: list[str], b: list[str]) -> bool:
    """Nearly the same words, but different numbers or one or two swapped words."""
    wa = {t for t in a if not _NUM.search(t)}
    wb = {t for t in b if not _NUM.search(t)}
    if not wa or not wb:
        return False
    if len(wa & wb) / len(wa | wb) < 0.7:
        return False
    na = {t for t in a if _NUM.search(t)}
    nb = {t for t in b if _NUM.search(t)}
    if na and nb and na != nb:
        return True
    return 1 <= len(wa - wb) <= 2 and 1 <= len(wb - wa) <= 2


def reviewer_pairs(
    directory: Path, records, eligible: set[int]
) -> list[tuple[int, int, str]]:
    """(line, line, reason) pairs the reviewer called duplicates or merged together."""
    out: list[tuple[int, int, str]] = []
    path = directory / "memory-overrides.jsonl"
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            o = json.loads(raw)
            if o.get("op") != "archive" or o.get("by") != "reviewer":
                continue
            reason = o.get("reason") or ""
            target = int(o["target_line"])
            refs = [int(x) for x in re.findall(r"#(\d+)", reason) if int(x) != target]
            if refs and target in records and refs[0] in records:
                out.append((target, refs[0], reason[:200]))
    for line, rec in records.items():
        if rec.source_ref.startswith("merge("):
            ids = [int(x) for x in re.findall(r"#(\d+)", rec.source_ref)]
            for a, b in zip(ids, ids[1:], strict=False):
                if a in records and b in records:
                    out.append((a, b, f"merged into #{line}"))
    # Only pairs of plain (non-merge, length-capped) facts.
    return [(a, b, r) for a, b, r in out if a in eligible and b in eligible]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent", default="demo_assistant")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-chars", type=int, default=400)
    ap.add_argument("--k", type=int, default=5, help="neighbors per query fact")
    ap.add_argument("--bm25-high", type=int, default=120)
    ap.add_argument("--bm25-mid", type=int, default=120)
    ap.add_argument("--bm25-low", type=int, default=60)
    ap.add_argument("--update", type=int, default=50)
    ap.add_argument("--reviewer", type=int, default=120)
    ap.add_argument("--random", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    directory = PROJECT_ROOT / "data" / "memory" / args.agent
    out_dir = args.out or directory / "relate-dataset"
    records, lines = load_corpus(directory, args.max_chars)
    eligible = set(lines)
    bm = Bm25Index([records[ln].fact for ln in lines])
    print(f"corpus: {len(lines)} index facts (of {len(records)} journal lines)")

    # BM25 neighbor pool: every corpus fact as a query, top-k neighbors each.
    pool: dict[tuple[int, int], dict] = {}
    for i, line in enumerate(lines):
        for j, ratio in bm.neighbors(i, args.k):
            key = (min(line, lines[j]), max(line, lines[j]))
            if key not in pool or ratio > pool[key]["ratio"]:
                pool[key] = {"ratio": ratio, "query": line, "ti": i, "tj": j}
    print(
        f"bm25 pool: {len(pool)} neighbor pairs  "
        f"{dict(Counter(band(p['ratio']) for p in pool.values()))}"
    )

    chosen: dict[tuple[int, int], dict] = {}
    per_query: Counter[int] = Counter()

    def take(key, source, **extra) -> bool:
        if key in chosen or key[0] == key[1]:
            return False
        chosen[key] = {"source": source, **extra}
        return True

    # update scan over the whole pool (before band sampling claims the pairs).
    upd = [
        (key, p)
        for key, p in pool.items()
        if p["ratio"] > HIGH and looks_like_update(bm.docs[p["ti"]], bm.docs[p["tj"]])
    ]
    rng.shuffle(upd)
    for key, p in upd[: args.update]:
        take(key, "update", ratio=round(p["ratio"], 3))

    # Band-stratified BM25 sample, at most 2 pairs per query fact.
    items = list(pool.items())
    rng.shuffle(items)
    want = {"high": args.bm25_high, "mid": args.bm25_mid, "low": args.bm25_low}
    got: Counter[str] = Counter()
    for key, p in items:
        b = band(p["ratio"])
        if got[b] >= want[b] or per_query[p["query"]] >= 2:
            continue
        if take(key, "bm25", ratio=round(p["ratio"], 3), band=b):
            got[b] += 1
            per_query[p["query"]] += 1

    rev = reviewer_pairs(directory, records, eligible)
    rng.shuffle(rev)
    n_rev = 0
    for a, b, reason in rev:
        if n_rev >= args.reviewer:
            break
        if take((min(a, b), max(a, b)), "reviewer", reviewer_reason=reason):
            n_rev += 1

    n_rand = 0
    while n_rand < args.random:
        a, b = rng.sample(lines, 2)
        if take((min(a, b), max(a, b)), "random"):
            n_rand += 1

    # Write: A is the older fact (born_on, then line), ids p0001…
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "candidates.jsonl"
    rows = []
    for n, (key, meta) in enumerate(sorted(chosen.items()), start=1):
        x, y = (records[key[0]], key[0]), (records[key[1]], key[1])
        older, newer = sorted([x, y], key=lambda r: (r[0].born_on or "", r[1]))

        def side(rec_line):
            rec, line = rec_line
            return {
                "line": line,
                "date": (rec.born_on or "")[:10],
                "type": rec.type,
                "tier": rec.tier,
                "fact": rec.fact,
            }

        rows.append({"id": f"p{n:04d}", "a": side(older), "b": side(newer), **meta})
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(rows)} candidates -> {path}")
    print("by source:", dict(Counter(r["source"] for r in rows)))
    print(
        "bm25 bands:",
        dict(Counter(r.get("band") for r in rows if r["source"] == "bm25")),
    )
    print(
        f"(pools: {len(upd)} update-shaped pairs, {len(rev)} reviewer pairs available)"
    )
    for src in ("update", "bm25", "reviewer", "random"):
        sample = [r for r in rows if r["source"] == src][:2]
        for r in sample:
            tag = r.get("band") or r.get("reviewer_reason", "")[:50]
            print(f"\n[{src}] {r['id']} {tag}")
            print(f"  A {r['a']['date']} #{r['a']['line']}: {r['a']['fact'][:110]}")
            print(f"  B {r['b']['date']} #{r['b']['line']}: {r['b']['fact'][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
