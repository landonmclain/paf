"""Probe: prove gpt-4o-mini judge variance at temp 0.

Re-judges the IDENTICAL stored (question, gold, model_answer) inputs K times
via the harness's own _lme_judge (gpt-4o-mini, temperature 0, seeded, provider
pinned). If a single input yields both 'yes' and 'no' across runs, that is
direct evidence the judge is non-deterministic even at temp 0.

Also prints gold-token-overlap with the model answer as an independent,
judge-free measure of "was the answer content actually present."
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())

from tests.run_lmem import _lme_judge  # noqa: E402


def gold_overlap(gold, ans) -> float:
    gt = set(re.findall(r"[a-z0-9']+", str(gold).lower()))
    at = set(re.findall(r"[a-z0-9']+", str(ans).lower()))
    if not gt:
        return 0.0
    return len(gt & at) / len(gt)


def main() -> None:
    K = int(os.environ.get("K", "5"))
    data = json.load(open("tests/results/lmem-20260929-014329.json", encoding="utf-8"))[
        "recall"
    ]
    by = {q["question_id"]: q for q in data["questions"]}

    flipped_wrong = [
        "gpt4_70e84552_abs",
        "57f827a0",
        "352ab8bd",
        "gpt4_ab202e7f",
        "gpt4_0a05b494",
        "a2f3aa27",
        "51c32626",
        "7405e8b1",
        "gpt4_7f6b06db",
        "gpt4_61e13b3c",
        "6e984302",
        "gpt4_8279ba03",
    ]
    # controls: first 3 questions the stored run judged CORRECT (stable baseline)
    controls = [
        q["question_id"]
        for q in data["questions"]
        if (q.get("judged") or {}).get("correct")
    ][:3]

    qids = flipped_wrong + controls
    print(
        f"K={K} re-judgments per question | {len(qids)} questions "
        f"({len(flipped_wrong)} flipped-wrong + {len(controls)} correct-controls)"
    )
    print("=" * 100)

    n_unstable = 0
    for qid in qids:
        q = by[qid]
        gold, ans = q["gold_answer"], q["answer"]
        ov = gold_overlap(gold, ans)
        dist: dict[str, int] = {}
        reasons: list[str] = []
        for _ in range(K):
            j = _lme_judge(q["question_type"], q["question"], gold, ans, qid)
            dist[j["label"]] = dist.get(j["label"], 0) + 1
            reasons.append(j["reasoning"])
        stable = len(dist) == 1
        if not stable:
            n_unstable += 1
        stored = "correct" if (q.get("judged") or {}).get("correct") else "wrong"
        print(
            f"{qid} | type={q['question_type']:<20} | gold-tok-overlap={ov:.2f} "
            f"| stored={stored} | {K}x verdicts={dist} | stable={'YES' if stable else 'NO ***'}"
        )
        if not stable:
            for i, r in enumerate(reasons):
                print(f"      run{i + 1}: {r.strip()[:180]!r}")
    print("=" * 100)
    print(
        f"UNSTABLE (verdict flipped across identical inputs): {n_unstable}/{len(qids)}"
    )
    print("=> judge variance at temp 0 is PROVEN if any of these flipped.")


if __name__ == "__main__":
    main()
