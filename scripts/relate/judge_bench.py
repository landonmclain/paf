"""Benchmark local judges against the DeepSeek reference on a small fixed set.

Reference: ``reviewer-reference.jsonl`` from ``reviewer_labels.py`` (DeepSeek
where its two runs agree; a stronger referee where they don't). The fixed set
is every reference ``update`` plus 40 ``same`` and 40 ``different`` pairs drawn
(deterministically) from the consensus pairs, so the labels being matched are
the clear ones. ``--contested`` also scores the refereed pairs, separately:
those are genuinely ambiguous (the referee split 50/50 with DeepSeek's runs).

Judges are named ``kind:model[@effort]``:

  chat:gpt-oss:20b@medium   the PRODUCTION reviewer path -- the
                            memory_pair_reviewer prompt + PairReview structured
                            output, via pydantic-ai ``ollama:<model>``. The
                            optional ``@effort`` sets reasoning_effort (``none``
                            turns a thinking model's thinking off).
  decision:nimble:latest    decision models via Ollama's /v1/systemone (the
                            same 3-way criteria + keep question tev1 got).

Each judge's labels go to ``judge-bench/<judge>.jsonl`` (``judge-bench-full/`` with
``--full``; resumable); every
run rewrites ``judge-bench/summary.md`` with all judges scored so far.

Usage (from scripts/relate, with OLLAMA_BASE_URL set for chat judges):
    PYTHONPATH= ../../.venv/bin/python judge_bench.py chat:gpt-oss:20b@medium decision:nimble:latest
    PYTHONPATH= ../../.venv/bin/python judge_bench.py --report
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import statistics
import sys
import time
from pathlib import Path

from reviewer_labels import (
    _append,
    _dir,
    _load,
    _obsolete,
    _pct,
    pair_prompt,
    tev1_one,
)

from paf.capabilities.memory._pairwise import PairReview, _judge_spec
from paf.degeneration_guard import run_guarded

PER_LABEL = {"same": 40, "different": 40}  # plus every `update`
PAIR_TIMEOUT_SECONDS = 300


def _rank(pid: str) -> str:
    return hashlib.sha256(f"judge-bench:{pid}".encode()).hexdigest()


def build_set(d: Path, contested: bool, full: bool = False) -> list[tuple[dict, dict]]:
    """``(candidate, reference)`` pairs: the fixed consensus set (+ contested)."""
    cands = {c["id"]: c for c in _load(d / "candidates.jsonl")}
    ref = [r for r in _load(d / "reviewer-reference.jsonl") if r["id"] in cands]
    picked: list[dict] = []
    for label in ("update", "same", "different"):
        pool = sorted(
            (r for r in ref if r["src"] == "consensus" and r["label"] == label),
            key=lambda r: _rank(r["id"]),
        )
        picked += pool if full or label == "update" else pool[: PER_LABEL[label]]
    if contested:
        picked += [r for r in ref if r["src"] == "referee"]
    return [(cands[r["id"]], r) for r in picked]


def _slug(judge: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", judge)


def parse_judge(judge: str) -> tuple[str, str, str | None]:
    kind, _, rest = judge.partition(":")
    if kind not in ("chat", "decision") or not rest:
        raise SystemExit(
            f"bad judge {judge!r}: use chat:<model>[@effort] or decision:<model>"
        )
    model, _, effort = rest.partition("@")
    return kind, model, effort or None


async def run_chat(model: str, effort: str | None, items, out: Path) -> None:
    from pydantic_ai import Agent

    _m, _s, instructions, _sugg = _judge_spec()
    settings: dict = {"temperature": 0}
    if effort:
        settings["openai_reasoning_effort"] = effort
    agent = Agent(
        f"ollama:{model}",
        name="judge-bench",
        instructions=instructions,
        output_type=PairReview,
        model_settings=settings,  # type: ignore[arg-type]
        retries=2,
    )
    for c, _ref in items:
        t = time.time()
        row: dict = {"id": c["id"]}
        try:
            result = await asyncio.wait_for(
                run_guarded(agent, pair_prompt(c)), PAIR_TIMEOUT_SECONDS
            )
            js = result.output.judgements
            j = next((x for x in js if x.index == 1), js[0] if js else None)
            if j is None:
                raise ValueError("no judgement")
            row |= {"label": j.label, "keep": j.keep if j.label == "same" else None}
        except Exception as e:  # noqa: BLE001 - an invalid output is a result too
            row |= {"label": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        row["seconds"] = round(time.time() - t, 2)
        _append(out, row)


def run_decision(model: str, items, out: Path) -> None:
    for c, _ref in items:
        t = time.time()
        r = tev1_one(c, model)
        row = {"id": c["id"], "label": r.get("label"), "keep": r.get("keep")}
        if r.get("probabilities"):
            row["probabilities"] = r["probabilities"]
        if r.get("error"):
            row["error"] = r["error"]
        row["seconds"] = round(time.time() - t, 2)
        _append(out, row)


def score(items, rows: dict[str, dict]) -> dict:
    scored = [(c, ref) for c, ref in items if c["id"] in rows]
    if not scored:
        return {}
    valid = [(c, ref) for c, ref in scored if rows[c["id"]].get("label")]

    def lab(c):
        return rows[c["id"]]["label"]

    tp = sum(_obsolete(lab(c)) and _obsolete(ref["label"]) for c, ref in valid)
    fp = sum(_obsolete(lab(c)) and not _obsolete(ref["label"]) for c, ref in valid)
    fn = sum(not _obsolete(lab(c)) and _obsolete(ref["label"]) for c, ref in valid)
    keep_items = [
        (c, ref)
        for c, ref in valid
        if ref["label"] == "same" and lab(c) == "same" and ref.get("keep")
    ]
    secs = [rows[c["id"]]["seconds"] for c, _ in scored if "seconds" in rows[c["id"]]]
    return {
        "n": len(scored),
        "valid": _pct(len(valid), len(scored)),
        "label": _pct(sum(lab(c) == ref["label"] for c, ref in valid), len(scored)),
        "obsolete": _pct(
            sum(_obsolete(lab(c)) == _obsolete(ref["label"]) for c, ref in valid),
            len(scored),
        ),
        "precision": f"{tp / (tp + fp):.0%}" if tp + fp else "n/a",
        "recall": f"{tp / (tp + fn):.0%}" if tp + fn else "n/a",
        "keep": _pct(
            sum(rows[c["id"]].get("keep") == ref["keep"] for c, ref in keep_items),
            len(keep_items),
        ),
        "sec": f"{statistics.median(secs):.1f}" if secs else "n/a",
    }


def summary(d: Path, items, folder: str = "judge-bench") -> str:
    out_dir = d / folder
    consensus = [(c, r) for c, r in items if r["src"] == "consensus"]
    contested = [(c, r) for c, r in items if r["src"] == "referee"]
    head = (
        "| judge | n | valid | label | obsolete | precision | recall | keep | s/pair |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    lines = [
        "# Local judges vs the DeepSeek reference",
        "",
        "Consensus set: every `update` + 40 `same` + 40 `different` where DeepSeek's",
        "two runs agreed. DeepSeek's own run-to-run agreement on all 537 pairs is 91%",
        "(label) — a judge can't meaningfully beat that ceiling here.",
        "`obsolete` = same|update vs different (the decision that hides a fact);",
        "`precision` matters most there: a wrong obsolete hides a fact that's true.",
        "",
        "## Consensus pairs",
        "",
        head,
    ]
    files = sorted(out_dir.glob("*.jsonl")) if out_dir.exists() else []
    for f in files:
        rows = {r["id"]: r for r in _load(f)}
        s = score(consensus, rows)
        if s:
            lines.append(
                f"| {f.stem} | {s['n']} | {s['valid']} | {s['label']} | {s['obsolete']} | "
                f"{s['precision']} | {s['recall']} | {s['keep']} | {s['sec']} |"
            )
    if contested:
        lines += ["", "## Contested pairs (agreement with the referee)", "", head]
        for f in files:
            rows = {r["id"]: r for r in _load(f)}
            s = score(contested, rows)
            if s:
                lines.append(
                    f"| {f.stem} | {s['n']} | {s['valid']} | {s['label']} | {s['obsolete']} | "
                    f"{s['precision']} | {s['recall']} | {s['keep']} | {s['sec']} |"
                )
    text = "\n".join(lines) + "\n"
    (out_dir).mkdir(exist_ok=True)
    (out_dir / "summary.md").write_text(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark local judges vs the DeepSeek reference."
    )
    ap.add_argument(
        "judges", nargs="*", help="chat:<model>[@effort] or decision:<model>"
    )
    ap.add_argument("--agent", default="demo_assistant")
    ap.add_argument(
        "--contested", action="store_true", help="Also run/score the refereed pairs."
    )
    ap.add_argument("--report", action="store_true", help="Only print the summary.")
    ap.add_argument(
        "--full",
        action="store_true",
        help="Use EVERY consensus pair (not 40+40+all update); results go to judge-bench-full/.",
    )
    args = ap.parse_args()

    d = _dir(args.agent)
    items = build_set(d, args.contested, args.full)
    folder = "judge-bench-full" if args.full else "judge-bench"
    out_dir = d / folder
    out_dir.mkdir(exist_ok=True)
    if not args.report:
        for judge in args.judges:
            kind, model, effort = parse_judge(judge)
            out = out_dir / f"{_slug(judge)}.jsonl"
            done = {r["id"] for r in _load(out)}
            todo = [(c, r) for c, r in items if c["id"] not in done]
            print(f"{judge}: {len(done)} done, {len(todo)} to judge", flush=True)
            t = time.time()
            if kind == "chat":
                asyncio.run(run_chat(model, effort, todo, out))
            else:
                run_decision(model, todo, out)
            print(f"{judge}: finished in {time.time() - t:.0f}s", flush=True)
    print(summary(d, items, folder))
    return 0


if __name__ == "__main__":
    sys.exit(main())
