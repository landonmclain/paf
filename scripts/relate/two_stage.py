"""Two-stage judge, small end-to-end check: screen fast, confirm slow.

Stage 1 -- a fast decision model (default ``nimble``) labels every pair.
Stage 2 -- the screen is unloaded from Ollama, then a thinking chat model
           (default ``qwen3.8:27b``) re-judges ONLY the pairs the screen
           called obsolete (same / update), one pair per call, with the
           production reviewer prompt. The two models never need to share
           the GPU: they run in sequence.
Final  -- a pair is obsolete only if BOTH stages say so; the confirmer's
           label (and keep side) wins. Anything the screen passes over is
           ``different``, as in production.

Scored against the DeepSeek reference on a small mixed sample. This is a
does-the-process-work check, not a benchmark.

Usage (from scripts/relate, .env sourced):
    PYTHONPATH= ../../.venv/bin/python two_stage.py            # 40 pairs
    PYTHONPATH= ../../.venv/bin/python two_stage.py --n 40 --screen decision:clef-flash:9b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request

from judge_bench import _slug, build_set, parse_judge, run_chat, run_decision
from reviewer_labels import _dir, _load, _obsolete, _pct


def unload(model: str) -> None:
    """Free a model's GPU memory (Ollama keep_alive=0)."""
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        body,
        {"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=60).read()


def sample(items: list, n: int) -> list:
    """A small mixed set: roughly 1/4 update, 3/8 same, 3/8 different."""
    want = {"update": n // 4, "same": 3 * n // 8, "different": n - n // 4 - 3 * n // 8}
    out = []
    for label, k in want.items():
        out += [it for it in items if it[1]["label"] == label][:k]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Two-stage judge, end-to-end check.")
    ap.add_argument("--agent", default="demo_assistant")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--screen", default="decision:nimble:latest")
    ap.add_argument("--confirm", default="chat:qwen3.8:27b")
    args = ap.parse_args()

    d = _dir(args.agent)
    items = sample(build_set(d, contested=False, full=True), args.n)
    out = d / "two-stage"
    out.mkdir(exist_ok=True)
    s_kind, s_model, _ = parse_judge(args.screen)
    c_kind, c_model, c_effort = parse_judge(args.confirm)
    assert s_kind == "decision" and c_kind == "chat", (
        "screen=decision:..., confirm=chat:..."
    )
    s_file, c_file = (
        out / f"screen-{_slug(args.screen)}.jsonl",
        out / f"confirm-{_slug(args.confirm)}.jsonl",
    )
    for f in (s_file, c_file):
        f.unlink(missing_ok=True)

    t0 = time.time()
    run_decision(s_model, items, s_file)
    screen = {r["id"]: r for r in _load(s_file)}
    flagged = [(c, r) for c, r in items if _obsolete(screen[c["id"]].get("label"))]
    t1 = time.time()
    print(
        f"stage 1 ({s_model}): {len(items)} pairs in {t1 - t0:.0f}s, flagged {len(flagged)} as obsolete",
        flush=True,
    )

    unload(s_model)
    asyncio.run(run_chat(c_model, c_effort, flagged, c_file))
    confirm = {r["id"]: r for r in _load(c_file)}
    t2 = time.time()
    print(
        f"stage 2 ({c_model}): re-judged {len(flagged)} flagged pairs in {t2 - t1:.0f}s",
        flush=True,
    )
    unload(c_model)

    rows = []
    for c, ref in items:
        pid = c["id"]
        s, k = screen[pid], confirm.get(pid)
        label, keep = "different", None
        if _obsolete(s.get("label")) and k and k.get("label") and _obsolete(k["label"]):
            label, keep = k["label"], k.get("keep")
        rows.append(
            {
                "id": pid,
                "screen": s.get("label"),
                "confirm": k.get("label") if k else None,
                "final": label,
                "keep": keep,
                "ref": ref["label"],
                "ref_keep": ref.get("keep"),
            }
        )
    (out / "result.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def stats(label_of) -> tuple[int, int, int, int]:
        tp = sum(_obsolete(label_of(r)) and _obsolete(r["ref"]) for r in rows)
        fp = sum(_obsolete(label_of(r)) and not _obsolete(r["ref"]) for r in rows)
        fn = sum(not _obsolete(label_of(r)) and _obsolete(r["ref"]) for r in rows)
        ok = sum(label_of(r) == r["ref"] for r in rows)
        return tp, fp, fn, ok

    print(
        f"\n{len(rows)} pairs ({sum(_obsolete(r['ref']) for r in rows)} truly obsolete per the DeepSeek reference)"
    )
    for name, fn_ in (
        ("screen alone", lambda r: r["screen"]),
        ("two-stage", lambda r: r["final"]),
    ):
        tp, fp, fn, ok = stats(fn_)
        prec = f"{tp / (tp + fp):.0%}" if tp + fp else "n/a"
        rec = f"{tp / (tp + fn):.0%}" if tp + fn else "n/a"
        print(
            f"  {name:13} exact {_pct(ok, len(rows))} | flagged {tp + fp}: {tp} right, {fp} wrong | precision {prec} recall {rec}"
        )
    keep_rows = [
        r for r in rows if r["final"] == "same" and r["ref"] == "same" and r["ref_keep"]
    ]
    print(
        f"  keep side (confirmed same): {_pct(sum(r['keep'] == r['ref_keep'] for r in keep_rows), len(keep_rows))}"
    )
    print(f"  total {t2 - t0:.0f}s; rows in {out}/result.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
