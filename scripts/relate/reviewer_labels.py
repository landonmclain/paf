"""Label the mined pairs the way the PRODUCTION pairwise reviewer would, and
measure how far each judge can be trusted -- with DeepSeek as the reference.

The relate dataset's earlier labels (``label_pairs.py``, ``review_pairs.py``)
used a 4-way rubric (same / update / related / unrelated) whose definitions
differ from what ``memory/_pairwise.py`` actually asks: e.g. "B repeats A and
adds a claim" is ``related`` there but ``same, keep new`` in production. This
script labels against the production task instead:

  r1, r2   DeepSeek with the production judge (spec, settings, prompt, schema
           from ``memory_pair_reviewer``), run twice: pairs where it disagrees
           with itself are ambiguous, and the reviewer should leave them alone.
  referee  A stronger model rules on the pairs where r1 and r2 disagree.
  tev1     The local decision model, asked the same 3-way question
           (Ollama's /v1/systemone, with probabilities) plus which to keep.
  report   Agreement + the reference label set; tev1 scored against it, with
           its "obsolete" cut-off tuned on a dev half and tested on the other.

Each pair is judged alone (NEW = fact B, OLDER = [fact A]); production batches
up to 12 older candidates per call, which this deliberately leaves out.

Every step is resumable (ids already in a step's file are skipped).

Usage (from the repo root):
    set -a && . ./.env && set +a
    PYTHONPATH= .venv/bin/python scripts/relate/reviewer_labels.py all
    PYTHONPATH= .venv/bin/python scripts/relate/reviewer_labels.py report
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from paf.capabilities.memory._pairwise import PairReview, _agent, _judge_spec
from paf.config import PROJECT_ROOT
from paf.degeneration_guard import run_guarded

REFEREE_MODEL = "openrouter:openai/gpt-5.6-terra"
TEV1_MODEL = "tev1:4b"
LABELS = ("same", "update", "different")

# The production prompt's label definitions, condensed for tev1's criteria.
TEV1_CRITERIA = {
    "same": (
        "one fact is fully covered by the other: they say the same thing, or one "
        "says everything the other does plus more"
    ),
    "update": (
        "the same specific attribute of the same specific thing, but the value "
        "changed (new place, number, date, version, status, or a reversed "
        "decision), and fact_b replaces fact_a's entire claim"
    ),
    "different": (
        "anything else: same topic but each says something the other doesn't, or "
        "a different topic. The thing a fact is about is specific (a project, "
        "tool, machine, place, person) -- never just 'the user'"
    ),
}
TEV1_KEEP = {
    "new": "fact_b (newer) says everything fact_a does, or they are equal",
    "old": "fact_a (older) says everything fact_b does, plus more",
}


def _dir(agent: str) -> Path:
    return PROJECT_ROOT / "data" / "memory" / agent / "relate-dataset"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def _append(path: Path, row: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def _render(f: dict) -> str:
    """Same shape as ``_pairwise.Fact.render``: ``(#line, date) fact``."""
    return f"(#{f['line']}, {(f.get('date') or '')[:10] or 'undated'}) {f['fact']}"


def pair_prompt(c: dict) -> str:
    """Same shape as ``_pairwise.pair_prompt`` with one older fact."""
    return "\n".join([f"NEW {_render(c['b'])}", "", "OLDER:", f"1. {_render(c['a'])}"])


# --- DeepSeek (production judge) and the referee ----------------------------------


async def _judge(agent, c: dict, model: str) -> dict:
    try:
        result = await run_guarded(agent, pair_prompt(c))
        j = next(
            (x for x in result.output.judgements if x.index == 1),
            result.output.judgements[0] if result.output.judgements else None,
        )
        if j is None:
            raise ValueError("no judgement returned")
        return {
            "id": c["id"],
            "label": j.label,
            "keep": j.keep if j.label == "same" else None,
            "reason": j.reason,
            "model": model,
        }
    except Exception as e:  # noqa: BLE001 - record and move on; rerun retries
        return {"id": c["id"], "label": None, "error": f"{type(e).__name__}: {e}"}


async def label_step(
    cands: list[dict], out: Path, *, referee: bool, workers: int
) -> None:
    from pydantic_ai import Agent

    model, _settings, instructions, _sugg = _judge_spec()
    if referee:
        model = REFEREE_MODEL
        agent = Agent(
            model,
            name="pair-referee",
            instructions=instructions,
            output_type=PairReview,
            retries=2,
        )
    else:
        agent = _agent(PairReview, instructions)

    done = {r["id"] for r in _load(out) if r.get("label")}
    todo = [c for c in cands if c["id"] not in done]
    print(f"{out.name}: {len(done)} done, {len(todo)} to label with {model}")
    sem = asyncio.Semaphore(workers)
    finished = 0

    async def one(c: dict) -> None:
        nonlocal finished
        async with sem:
            row = await _judge(agent, c, model)
        if row.get("label"):
            _append(out, row)
        else:
            print(f"  {c['id']} failed: {row['error'][:160]}")
        finished += 1
        if finished % 25 == 0:
            print(f"  {finished}/{len(todo)}")

    await asyncio.gather(*(one(c) for c in todo))


# --- tev1 (local decision model) ----------------------------------------------------


def _post(url: str, body: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def tev1_one(c: dict, model: str = TEV1_MODEL) -> dict:
    body = {
        "model": model,
        "state": {"fact_a": c["a"]["fact"], "fact_b": c["b"]["fact"]},
        "questions": {
            "relation": {
                "type": "choice",
                "instructions": "How does fact_b (newer) relate to fact_a (older)?",
                "criteria": TEV1_CRITERIA,
            },
            "keep": {
                "type": "choice",
                "instructions": "If they say the same thing, which fact should be kept?",
                "criteria": TEV1_KEEP,
            },
        },
    }
    detail = ""
    for _attempt in range(2):
        try:
            ans = _post("http://127.0.0.1:11434/v1/systemone", body)["answers"]
            rel, keep = ans["relation"], ans["keep"]
            return {
                "id": c["id"],
                "label": rel["choice"],
                "probabilities": {
                    k: round(v, 4) for k, v in rel["probabilities"].items()
                },
                "keep": keep["choice"],
                "model": model,
            }
        except urllib.error.HTTPError as exc:
            # Some inputs break the loaded model ("invalid candidate
            # probability"); unload so the retry gets a fresh copy.
            detail = exc.read()[:200].decode(errors="replace")
            try:
                _post(
                    "http://127.0.0.1:11434/api/generate",
                    {"model": model, "keep_alive": 0},
                    timeout=60,
                )
            except Exception:  # noqa: BLE001
                pass
    return {"id": c["id"], "label": None, "error": detail}


def tev1_step(cands: list[dict], out: Path) -> None:
    done = {r["id"] for r in _load(out) if r.get("label")}
    todo = [c for c in cands if c["id"] not in done]
    print(f"{out.name}: {len(done)} done, {len(todo)} to label with {TEV1_MODEL}")
    for n, c in enumerate(todo, 1):
        row = tev1_one(c)
        if row.get("label"):
            _append(out, row)
        else:
            print(f"  {c['id']} failed: {row['error']}")
        if n % 50 == 0:
            print(f"  {n}/{len(todo)}")


# --- report -------------------------------------------------------------------------


def _obsolete(label: str | None) -> bool:
    """The decision that hides a fact: same (archive) or update (supersede)."""
    return label in ("same", "update")


def _split(pid: str) -> str:
    return "dev" if int(hashlib.sha256(pid.encode()).hexdigest(), 16) % 2 else "test"


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} = {n / d:.0%}" if d else "n/a"


def report(d: Path) -> dict:
    cands = {c["id"]: c for c in _load(d / "candidates.jsonl")}
    r1 = {r["id"]: r for r in _load(d / "reviewer-r1.jsonl")}
    r2 = {r["id"]: r for r in _load(d / "reviewer-r2.jsonl")}
    ref = {r["id"]: r for r in _load(d / "reviewer-referee.jsonl")}
    tev = {r["id"]: r for r in _load(d / "reviewer-tev1.jsonl")}
    lines: list[str] = [f"# Reviewer-task labels ({len(cands)} mined pairs)", ""]

    both = [i for i in cands if i in r1 and i in r2]
    agree = [i for i in both if r1[i]["label"] == r2[i]["label"]]
    agree_obs = [
        i for i in both if _obsolete(r1[i]["label"]) == _obsolete(r2[i]["label"])
    ]
    keep_pairs = [i for i in agree if r1[i]["label"] == "same"]
    keep_agree = [i for i in keep_pairs if r1[i].get("keep") == r2[i].get("keep")]
    lines += [
        "## DeepSeek self-consistency (r1 vs r2)",
        f"- label: {_pct(len(agree), len(both))}",
        f"- obsolete-or-not: {_pct(len(agree_obs), len(both))}",
        f"- keep side (both said same): {_pct(len(keep_agree), len(keep_pairs))}",
        f"- r1 labels: {dict(Counter(r1[i]['label'] for i in both))}",
        f"- flips (r1→r2): {dict(Counter((r1[i]['label'], r2[i]['label']) for i in both if i not in agree))}",
        "",
    ]

    # Reference: DeepSeek where it agrees with itself; the referee otherwise.
    reference: dict[str, dict] = {}
    contested = [i for i in both if i not in agree]
    for i in agree:
        reference[i] = {
            "label": r1[i]["label"],
            "keep": r1[i].get("keep"),
            "src": "consensus",
        }
    for i in contested:
        if i in ref:
            reference[i] = {
                "label": ref[i]["label"],
                "keep": ref[i].get("keep"),
                "src": "referee",
            }
    ref_vs_r1 = [i for i in contested if i in ref and ref[i]["label"] == r1[i]["label"]]
    lines += [
        "## Reference labels",
        f"- consensus: {len(agree)}, refereed: {sum(1 for i in contested if i in ref)}"
        f" of {len(contested)} contested",
        f"- referee sided with r1: {_pct(len(ref_vs_r1), sum(1 for i in contested if i in ref))}",
        f"- distribution: {dict(Counter(v['label'] for v in reference.values()))}",
        "",
    ]

    scored = [i for i in reference if i in tev]
    if scored:
        three = [i for i in scored if tev[i]["label"] == reference[i]["label"]]
        obs = [
            i
            for i in scored
            if _obsolete(tev[i]["label"]) == _obsolete(reference[i]["label"])
        ]
        lines += [
            f"## tev1 vs reference ({len(scored)} pairs)",
            f"- 3-way label: {_pct(len(three), len(scored))}",
            f"- obsolete-or-not (argmax): {_pct(len(obs), len(scored))}",
            f"- confusion (reference→tev1): {dict(Counter((reference[i]['label'], tev[i]['label']) for i in scored if tev[i]['label'] != reference[i]['label']))}",
        ]

        # Tune an "obsolete" cut-off on P(same)+P(update) on dev; score on test.
        # Precision matters most: a wrong obsolete hides a fact that is true.
        def p_obs(i: str) -> float:
            p = tev[i]["probabilities"]
            return p.get("same", 0) + p.get("update", 0)

        dev = [i for i in scored if _split(i) == "dev"]
        test = [i for i in scored if _split(i) == "test"]

        def stats(ids: list[str], t: float) -> tuple[float, float, float]:
            tp = sum(p_obs(i) >= t and _obsolete(reference[i]["label"]) for i in ids)
            fp = sum(
                p_obs(i) >= t and not _obsolete(reference[i]["label"]) for i in ids
            )
            fn = sum(p_obs(i) < t and _obsolete(reference[i]["label"]) for i in ids)
            acc = sum((p_obs(i) >= t) == _obsolete(reference[i]["label"]) for i in ids)
            prec = tp / (tp + fp) if tp + fp else 1.0
            rec = tp / (tp + fn) if tp + fn else 1.0
            return acc / len(ids), prec, rec

        grid = [x / 100 for x in range(5, 100)]
        best = max(grid, key=lambda t: stats(dev, t)[0]) if dev else 0.5
        safe = [t for t in grid if dev and stats(dev, t)[1] >= 0.95]
        safe_t = min(safe) if safe else None
        acc, prec, rec = stats(test, best) if test else (0, 0, 0)
        lines += [
            f"- cut-off for accuracy (dev {len(dev)}): P(obsolete) >= {best:.2f} → test {len(test)}: "
            f"accuracy {acc:.0%}, precision {prec:.0%}, recall {rec:.0%}",
        ]
        if safe_t is not None:
            acc, prec, rec = stats(test, safe_t)
            lines.append(
                f"- cut-off for ≥95% precision on dev: >= {safe_t:.2f} → test: "
                f"accuracy {acc:.0%}, precision {prec:.0%}, recall {rec:.0%}"
            )
        keep_ids = [
            i
            for i in scored
            if reference[i]["label"] == "same" and reference[i].get("keep")
        ]
        keep_ok = [i for i in keep_ids if tev[i].get("keep") == reference[i]["keep"]]
        lines += [
            f"- keep side on reference `same` pairs: {_pct(len(keep_ok), len(keep_ids))}",
            "",
        ]

    text = "\n".join(lines)
    (d / "reviewer-report.md").write_text(text + "\n")
    (d / "reviewer-reference.jsonl").write_text(
        "".join(json.dumps({"id": i, **v}) + "\n" for i, v in sorted(reference.items()))
    )
    print(text)
    return reference


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Label mined pairs with the production reviewer task."
    )
    ap.add_argument("step", choices=("r1", "r2", "referee", "tev1", "report", "all"))
    ap.add_argument("--agent", default="demo_assistant")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--limit", type=int, default=None, help="Only the first N pairs (smoke test)."
    )
    args = ap.parse_args()

    d = _dir(args.agent)
    cands = _load(d / "candidates.jsonl")[: args.limit]
    print(f"{len(cands)} candidate pairs in {d}")
    steps = (
        ("r1", "r2", "referee", "tev1", "report")
        if args.step == "all"
        else (args.step,)
    )
    for step in steps:
        if step in ("r1", "r2"):
            asyncio.run(
                label_step(
                    cands,
                    d / f"reviewer-{step}.jsonl",
                    referee=False,
                    workers=args.workers,
                )
            )
        elif step == "referee":
            r1 = {r["id"]: r for r in _load(d / "reviewer-r1.jsonl")}
            r2 = {r["id"]: r for r in _load(d / "reviewer-r2.jsonl")}
            contested = [
                c
                for c in cands
                if c["id"] in r1
                and c["id"] in r2
                and r1[c["id"]]["label"] != r2[c["id"]]["label"]
            ]
            asyncio.run(
                label_step(
                    contested,
                    d / "reviewer-referee.jsonl",
                    referee=True,
                    workers=args.workers,
                )
            )
        elif step == "tev1":
            tev1_step(cands, d / "reviewer-tev1.jsonl")
        else:
            report(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
