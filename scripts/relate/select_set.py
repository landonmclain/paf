"""Build the balanced labeled set from model-labeled candidates.

Selects ~200 pairs balanced by DeepSeek label (50 per label; a short label —
usually `update` — contributes what it has and the rest fill the gap), splits
them into dev / test halves stratified by label, and flags the pairs a human
must decide:

- the two judges disagree (DeepSeek vs tev1), or
- DeepSeek marked the pair ``ambiguous``, or
- a random sample of agreements (``--audit``), to catch mistakes the two
  judges share.

Writes ``dataset.jsonl`` beside the candidates. Human labels live separately in
``human-labels.jsonl`` (written by ``review_pairs.py``), so re-running this
never loses them. A row's gold label = the human label if present, else the
judges' agreed label (only for unflagged agreements).

Usage (from the repo root):
    PYTHONPATH= .venv/bin/python scripts/relate/select_set.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from pairs import LABELS

from paf.config import PROJECT_ROOT


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def main() -> int:
    ap = argparse.ArgumentParser(description="Select the balanced labeled set.")
    ap.add_argument("--agent", default="demo_assistant")
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--audit", type=int, default=30, help="random agreements to review")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    d = PROJECT_ROOT / "data" / "memory" / args.agent / "relate-dataset"
    cands = {c["id"]: c for c in _load(d / "candidates.jsonl")}
    ds = {r["id"]: r for r in _load(d / "labels-deepseek.jsonl")}
    tv = {r["id"]: r for r in _load(d / "labels-tev1.jsonl")}
    ids = [i for i in cands if i in ds and i in tv]
    print(f"{len(ids)} pairs labeled by both judges (of {len(cands)} candidates)")

    by_label: dict[str, list[str]] = {lbl: [] for lbl in LABELS}
    for i in ids:
        by_label[ds[i]["label"]].append(i)
    for v in by_label.values():
        rng.shuffle(v)

    # Balanced: up to size/4 per label; leftover slots go round-robin to
    # labels that still have pairs.
    per = args.size // len(LABELS)
    picked = {lbl: v[:per] for lbl, v in by_label.items()}
    rest = {lbl: v[per:] for lbl, v in by_label.items()}
    while sum(map(len, picked.values())) < args.size and any(rest.values()):
        for lbl in LABELS:
            if rest[lbl] and sum(map(len, picked.values())) < args.size:
                picked[lbl].append(rest[lbl].pop())

    rows: list[dict] = []
    for chosen in picked.values():
        for n, i in enumerate(chosen):  # alternate -> stratified dev/test split
            c, a, b = cands[i], ds[i], tv[i]
            rows.append(
                {
                    "id": i,
                    "split": "dev" if n % 2 == 0 else "test",
                    "source": c["source"],
                    "a": c["a"],
                    "b": c["b"],
                    "deepseek": {k: a[k] for k in ("label", "reason", "ambiguous")},
                    "tev1": {k: b[k] for k in ("label", "confidence", "probabilities")},
                }
            )

    agree = [r for r in rows if r["deepseek"]["label"] == r["tev1"]["label"]]
    audit = set(rng.sample([r["id"] for r in agree], min(args.audit, len(agree))))
    for r in rows:
        why = []
        if r["tev1"]["label"] is None:
            why.append("tev1-error")
        elif r["deepseek"]["label"] != r["tev1"]["label"]:
            why.append("disagree")
        if r["deepseek"]["ambiguous"]:
            why.append("ambiguous")
        if r["id"] in audit and not why:
            why.append("audit")
        r["needs_review"] = bool(why)
        r["review_reason"] = why

    rng.shuffle(rows)  # review order: mixed, not grouped by label
    out = d / "dataset.jsonl"
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )

    print(f"wrote {len(rows)} -> {out}")
    print("by DeepSeek label:", dict(Counter(r["deepseek"]["label"] for r in rows)))
    print("by split:", dict(Counter(r["split"] for r in rows)))
    print(
        f"judges agree on {len(agree)}/{len(rows)}; "
        f"needs review: {sum(r['needs_review'] for r in rows)} "
        f"{dict(Counter(w for r in rows for w in r['review_reason']))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
