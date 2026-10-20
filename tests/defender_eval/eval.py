"""Prompt-injection defender calibration eval.

Scores a labeled corpus of tool outputs (``fixtures.jsonl``) with the defender
EXACTLY as production builds it (``build._build_injection_guard``'s scorer), and
reports where
it is wrong: benign outputs it would withhold (false positives, which break the
agent) and injections it would let through (false negatives, which break the
guarantee). No LLM is involved, so a run is free, offline and deterministic.

Why it exists: the classifier takes the max score over sentence chunks, so
ordinary tool output can trip it (a traceback, a listing of `/etc`), and the
model is then told the result was withheld. Whether that is acceptable, and what
to tune, is an empirical question; this is the measurement.

Fixture schema (one JSON object per line)::

    id, label ("benign" | "injection"), category, tool, text, source, note,
    known_issue (optional bool: the verdict is currently WRONG and tracked)

Usage:
    PYTHONPATH= .venv/bin/python -m tests.defender_eval.eval
    PYTHONPATH= .venv/bin/python -m tests.defender_eval.eval --json out.json
    PYTHONPATH= .venv/bin/python -m tests.defender_eval.eval --harvest 24

``--harvest N`` lists tool outputs from the last N hours of Langfuse traces in
which the defender fired, as unlabeled candidates: label them and append to
``fixtures.jsonl`` so real-world false positives become permanent fixtures.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).with_name("fixtures.jsonl")
LABELS = ("benign", "injection")
_SWEEP = (0.5, 0.6, 0.64, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)


@dataclass
class Verdict:
    id: str
    label: str
    category: str
    tool: str
    blocked: bool
    risk_level: str
    score: float
    pattern_blocked: bool
    """Blocked by Tier 1 patterns (the score alone would not have blocked)."""
    trigger: str | None
    """The sentence chunk the classifier scored highest."""
    known_issue: bool = False
    note: str = ""

    @property
    def wrong(self) -> bool:
        return self.blocked != (self.label == "injection")


def load_fixtures(path: Path = FIXTURES) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate fixture ids: {dupes}")
    for r in rows:
        if r["label"] not in LABELS:
            raise ValueError(f"{r['id']}: label must be one of {LABELS}")
    return rows


def build_defense():
    """The production scorer, from the production builder (default policy)."""
    from paf.capabilities.build import _build_injection_guard

    return _build_injection_guard(None).scorer


def block_threshold(scorer) -> float:
    return float(scorer.threshold)


def evaluate(rows: list[dict[str, Any]] | None = None, defense=None) -> list[Verdict]:
    rows = rows if rows is not None else load_fixtures()
    defense = defense or build_defense()
    threshold = block_threshold(defense)
    out: list[Verdict] = []
    for r in rows:
        res = defense.score_sync(r["text"], r["tool"])
        score = float(res.score or 0.0)
        out.append(
            Verdict(
                id=r["id"],
                label=r["label"],
                category=r["category"],
                tool=r["tool"],
                blocked=res.hit,
                risk_level=res.risk_level,
                score=score,
                pattern_blocked=res.hit and score < threshold,
                trigger=res.trigger,
                known_issue=bool(r.get("known_issue")),
                note=r.get("note", ""),
            )
        )
    return out


# ------------------------------------------------------------------ metrics
def rates(verdicts: list[Verdict]) -> dict[str, float | int]:
    benign = [v for v in verdicts if v.label == "benign"]
    inj = [v for v in verdicts if v.label == "injection"]
    fp = sum(v.blocked for v in benign)
    tp = sum(v.blocked for v in inj)
    return {
        "benign": len(benign),
        "injection": len(inj),
        "false_positives": fp,
        "false_positive_rate": fp / len(benign) if benign else 0.0,
        "detected": tp,
        "detection_rate": tp / len(inj) if inj else 0.0,
    }


def by_category(verdicts: list[Verdict]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Verdict]] = defaultdict(list)
    for v in verdicts:
        groups[v.category].append(v)
    return {
        cat: {
            "label": vs[0].label,
            "n": len(vs),
            "blocked": sum(v.blocked for v in vs),
            "wrong": sum(v.wrong for v in vs),
        }
        for cat, vs in sorted(groups.items())
    }


def sweep(verdicts: list[Verdict], thresholds=_SWEEP) -> list[dict[str, float]]:
    """Re-decide every fixture at each block threshold, from its recorded score.

    A fixture stays blocked if Tier 1 patterns blocked it, or its score clears
    the threshold. This is the cost curve: what each cutoff trades.
    """
    rows = []
    for t in thresholds:
        blocked = [v.pattern_blocked or v.score >= t for v in verdicts]
        benign = [
            b for v, b in zip(verdicts, blocked, strict=True) if v.label == "benign"
        ]
        inj = [
            b for v, b in zip(verdicts, blocked, strict=True) if v.label == "injection"
        ]
        rows.append(
            {
                "threshold": t,
                "false_positives": sum(benign),
                "false_positive_rate": sum(benign) / len(benign) if benign else 0.0,
                "detected": sum(inj),
                "detection_rate": sum(inj) / len(inj) if inj else 0.0,
            }
        )
    return rows


def aux_sweep(
    rows: list[dict[str, Any]], aux_thresholds=(0.1, 0.15, 0.2, 0.3, 0.5, 2.0)
) -> list[dict[str, float]]:
    """Does the model's second ("aux") head separate false positives from hits?

    The bundled model is two-headed and the library leaves the operating point to
    the caller: block iff ``main >= main_threshold AND aux < aux_threshold``.
    Collect (main, aux) per fixture once, then sweep the aux cutoff. ``2.0``
    never vetoes, i.e. the production rule.
    """
    from stackone_defender import PromptDefense
    from stackone_defender.types import MultiheadConfig

    from paf.capabilities.injection._scorer import _payload

    main_t = block_threshold(build_defense())
    probe = PromptDefense(
        config={"traversal": {"skip_large_arrays": False}},
        block_high_risk=True,
        enable_tier2=True,
        tier2_config={
            "multihead": MultiheadConfig(main_threshold=main_t, aux_threshold=2.0)
        },
    )
    scored = []
    for r in rows:
        res = probe.defend_tool_result(_payload(r["text"]), r["tool"])
        scored.append((r["label"], res.tier2_raw_score or 0.0, res.tier2_aux_score))
    out = []
    for a in aux_thresholds:
        blocked = [(m >= main_t and (x is None or x < a)) for _, m, x in scored]
        benign = [
            b for (lab, _, _), b in zip(scored, blocked, strict=True) if lab == "benign"
        ]
        inj = [
            b
            for (lab, _, _), b in zip(scored, blocked, strict=True)
            if lab == "injection"
        ]
        out.append(
            {
                "aux_threshold": a,
                "false_positives": sum(benign),
                "detected": sum(inj),
                "injections": len(inj),
            }
        )
    return out


# ------------------------------------------------------------------ report
def render(
    verdicts: list[Verdict], defense=None, rows: list[dict[str, Any]] | None = None
) -> str:
    r = rates(verdicts)
    lines = [
        "Prompt-injection defender calibration",
        f"  fixtures: {len(verdicts)}  ({r['benign']} benign, {r['injection']} injection)",
    ]
    if defense is not None:
        lines.append(f"  block threshold in production: {block_threshold(defense)}")
    lines += [
        f"  false positives (benign withheld): {r['false_positives']}/{r['benign']}"
        f"  = {r['false_positive_rate']:.0%}",
        f"  detection (injections withheld):   {r['detected']}/{r['injection']}"
        f"  = {r['detection_rate']:.0%}",
        "",
        "By category (wrong = verdict disagrees with the label):",
    ]
    for cat, c in by_category(verdicts).items():
        lines.append(
            f"  {cat:18} {c['label']:9} n={c['n']:<3} blocked={c['blocked']:<3} wrong={c['wrong']}"
        )
    wrong = sorted((v for v in verdicts if v.wrong), key=lambda v: (v.label, -v.score))
    lines += ["", f"Wrong verdicts ({len(wrong)}):"]
    for v in wrong:
        kind = "FALSE POSITIVE" if v.label == "benign" else "MISSED INJECTION"
        tag = " [known]" if v.known_issue else ""
        lines.append(
            f"  {kind:16} {v.id:22} {v.tool:14} score={v.score:.2f} {v.risk_level}{tag}"
        )
        if v.trigger:
            lines.append(f"      trigger: {v.trigger[:110]!r}")
    lines += [
        "",
        "Threshold sweep (Tier 1 pattern blocks held fixed; score cutoff varied):",
        "  threshold   false-pos          detection",
    ]
    for s in sweep(verdicts):
        mark = (
            "  <- production"
            if defense and s["threshold"] == block_threshold(defense)
            else ""
        )
        lines.append(
            f"  {s['threshold']:<10}  {s['false_positives']:>2} ({s['false_positive_rate']:>4.0%})"
            f"       {s['detected']:>2}/{r['injection']} ({s['detection_rate']:>4.0%}){mark}"
        )
    lines += [
        "",
        "Second-head check (block iff main >= threshold AND aux < cutoff; 2.0 = never veto):",
        "  aux cutoff   false-pos   detection",
    ]
    for a in aux_sweep(rows or load_fixtures()):
        lines.append(
            f"  {a['aux_threshold']:<10}   {a['false_positives']:>3}        "
            f"{a['detected']}/{a['injections']}"
        )
    scores = sorted(v.score for v in verdicts if v.label == "benign")
    inj_scores = sorted(v.score for v in verdicts if v.label == "injection")
    if scores and inj_scores:
        lines += [
            "",
            f"Score ranges: benign max={scores[-1]:.2f} median={scores[len(scores) // 2]:.2f}"
            f" | injection min={inj_scores[0]:.2f} median={inj_scores[len(inj_scores) // 2]:.2f}",
            "  (overlap between benign max and injection min means no single cutoff separates them)",
        ]
    return "\n".join(lines)


# ------------------------------------------------------------------ harvest
def _clickhouse(sql: str) -> str:
    return subprocess.run(
        [
            "docker",
            "exec",
            "langfuse4-clickhouse-1",
            "clickhouse-client",
            "--query",
            sql,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def harvest(hours: int) -> list[dict[str, Any]]:
    """Unlabeled candidates: tool outputs from traces where the defender fired."""
    traces = _clickhouse(
        "SELECT DISTINCT trace_id FROM events_core WHERE project_id='paf-project' "
        f"AND name='paf.guardrail.prompt_injection' AND start_time > now() - INTERVAL {int(hours)} HOUR "
        "FORMAT TSV"
    ).split()
    rows = []
    for trace in traces:
        raw = _clickhouse(
            "SELECT name, input, output FROM events_full WHERE project_id='paf-project' "
            f"AND trace_id='{trace}' AND type='TOOL' ORDER BY start_time FORMAT JSONEachRow"
        )
        for line in raw.splitlines():
            ev = json.loads(line)
            rows.append(
                {
                    "id": f"harvest-{trace[:8]}-{len(rows)}",
                    "label": None,
                    "category": "harvested",
                    "tool": ev["name"],
                    "text": ev["output"],
                    "source": "trace",
                    "note": f"trace {trace}; label benign|injection, set category, then append",
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fixtures", type=Path, default=FIXTURES)
    ap.add_argument("--json", type=Path, help="write per-fixture verdicts here")
    ap.add_argument("--harvest", type=int, metavar="HOURS")
    args = ap.parse_args()
    if args.harvest is not None:
        for row in harvest(args.harvest):
            print(json.dumps(row, ensure_ascii=False))
        return
    defense = build_defense()
    rows = load_fixtures(args.fixtures)
    verdicts = evaluate(rows, defense)
    print(render(verdicts, defense, rows))
    if args.json:
        args.json.write_text(json.dumps([asdict(v) for v in verdicts], indent=2))


if __name__ == "__main__":
    main()
