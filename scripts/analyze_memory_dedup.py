"""Read-only analysis of paf memory: candidate importance-score inputs + how
much more we could dedup from the journal via bounded fuzzy clustering.

Run from repo root with PYTHONPATH=. Never writes to data/memory.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict

BASE = "data/memory"


def _tokens(fact: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", fact.lower())
    STOP = set(
        "a an the and or but of to in for on with is are was were be been "
        "has have had do does did it its this that these those about into from "
        "using uses use used user via by as at not no yes i you we they he she".split()
    )
    return {t for t in raw if t not in STOP and len(t) > 2}


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    ij = len(a & b)
    return max(ij / len(a | b), ij / min(len(a), len(b)))


def load(agent: str):
    d = f"{BASE}/{agent}"
    with open(f"{d}/memory-journal.jsonl") as fh:
        journal_lines = [json.loads(l) for l in fh if l.strip()]
    index = json.load(open(f"{d}/memory-index.json"))
    entries = index.get("entries", [])
    # fold overrides last-wins per target_line
    ov = {}
    for l in open(f"{d}/memory-overrides.jsonl"):
        s = l.strip()
        if not s:
            continue
        o = json.loads(s)
        if o.get("target_line") is not None:
            ov[o["target_line"]] = o
    return journal_lines, entries, ov


def report(agent: str):
    journal, entries, ov = load(agent)
    print(f"\n################ {agent} ################")
    print(f"journal records: {len(journal)} | index entries: {len(entries)}")

    # --- candidate 1: dedup mention count (len deduped_lines) ---
    dedup = Counter(len(e.get("deduped_lines", [])) for e in entries)
    print("\n[candidate] dedup mention-count (len deduped_lines) per entry:")
    for k in sorted(dedup):
        print(f"   {k:>3} mentions -> {dedup[k]:>5} entries")

    # --- candidate 2: merge count — the REAL signal lives on each journal record's
    # source_ref "merge(#a,#b,...)", NOT in override reasons. For each index winner,
    # look up its current_line record; if it's a merge record, count folded items.
    line_to_src = {}
    for ln, r in enumerate(journal, 1):
        m = re.search(r"merge\((#[0-9]+(?:,#[0-9]+)*)\)", r.get("source_ref") or "")
        if m:
            line_to_src[ln] = len(m.group(1).split(","))
    merge_lines = set(line_to_src)
    per_entry = Counter()
    n_with_merge = 0
    live_merge_winners = 0
    for e in entries:
        ln = int(e["current_line"])
        if ln in line_to_src:
            per_entry[line_to_src[ln]] += 1
            live_merge_winners += 1
    print(f"\n[candidate] memory_merge count per INDEX WINNER (from journal source_ref):")
    print(f"   distinct merge-records: {len(merge_lines)} | live winners that are merges: {live_merge_winners}")
    print(f"   items-merged-per-winner distribution (winner:count): "
          f"{dict(sorted(per_entry.items()))}")

    # --- candidate 3: recency (age in days vs newest) ---
    born = [e.get("born_on", "")[:10] for e in entries if e.get("born_on")]
    if born:
        from datetime import date
        def parse(s):
            try:
                return date.fromisoformat(s[:10])
            except ValueError:
                return None
        dates = [parse(b) for b in born]
        dates = [d for d in dates if d]
        maxd = max(dates)
        ages = [(maxd - d).days for d in dates]
        from collections import Counter as C
        print(f"\n[candidate] recency: newest {maxd}; age (days) distribution:")
        for k in sorted(set(ages)):
            print(f"   {k:>3} days old -> {ages.count(k):>5} entries")

    # --- dedup potential: fuzzy cluster the journal and measure collapse ---
    # bounded: cluster by token-overlap >= 0.60 across ALL facts (full pass here
    # for measurement; a live impl would be bounded).
    toks = {i: _tokens(r.get("fact", "")) for i, r in enumerate(journal)}
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n = len(journal)
    for i in range(n):
        ti = toks[i]
        if not ti:
            continue
        for j in range(max(0, i - 400), i):
            if _overlap(ti, toks[j]) >= 0.60:
                union(i, j)
    roots = Counter(find(i) for i in range(n))
    clusters = [c for c in roots.values() if c > 1]
    print(f"\n[dedup potential] NAIVE max(Jaccard,containment)>=0.60 SEGMENTS — "
          f"this OVER-MERGES (artifact):")
    print(f"   journal records      : {n}")
    print(f"   would-be distinct    : {len(roots)}")
    print(f"   facts that would merge: {n - len(roots)} records folded")
    print(f"   largest cluster sizes: {sorted(roots.values(), reverse=True)[:5]}")

    # --- CORRECTED: pure-Jaccard, non-transitive fuzzy dedup potential ---
    def jacc(a, b):
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    for TH in (0.70, 0.80, 0.85):
        edges = 0
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if jacc(toks[i], toks[j]) >= TH:
                    edges += 1
                    pairs.append((i, j))
        parent2 = {}

        def find2(x, p=parent2):
            p.setdefault(x, x)
            while p[x] != x:
                p[x] = p[p[x]]
                x = p[x]
            return x

        def union2(a, b):
            ra, rb = find2(a), find2(b)
            if ra != rb:
                parent2[ra] = rb

        for i, j in pairs:
            union2(i, j)
        roots2 = Counter(find2(i) for i in range(n))
        cl = [c for c in roots2.values() if c > 1]
        print(f"   Jaccard>= {TH}: direct-sim pairs={edges} | distinct={len(roots2)} "
              f"(fold {n - len(roots2)} into {len(cl)} clusters) | "
              f"largest={max(roots2.values())}")


if __name__ == "__main__":
    agents = sys.argv[1:] or ["assistant", "demo_assistant"]
    for a in agents:
        report(a)
