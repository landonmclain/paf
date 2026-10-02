"""Model-label mined candidate pairs: DeepSeek (rubric) and tev1 (decision model).

Reads ``candidates.jsonl`` (from ``mine_pairs.py``) and writes one file per judge
beside it — ``labels-deepseek.jsonl`` / ``labels-tev1.jsonl`` — one row per pair
id. Resumable: ids already in a judge's file are skipped, so a crash or Ctrl-C
loses at most the in-flight batch.

These are DRAFT labels. ``select_set.py`` builds the balanced 200 and flags the
pairs a human must decide (judge disagreements, ``ambiguous``, a random sample
of agreements); ``review_pairs.py`` collects those human labels.

Usage (from the repo root):
    set -a && . ./.env && set +a
    PYTHONPATH= .venv/bin/python scripts/relate/label_pairs.py --judge both
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pairs import DEFS_V3, LABELS

from paf.config import PROJECT_ROOT

DEEPSEEK_MODEL = "deepseek/deepseek-v4-flash-0731"

SYSTEM = """\
You label pairs of memory facts for a dataset. Each fact is one statement an
assistant remembered about its user. Fact A is older; fact B is newer.

Decide how B relates to A. Ask these questions IN ORDER and stop at the first yes:

1. same — A and B make the same claim. Differences are only wording, word
   order, or minor detail that does not change the claim.
2. update — A and B are about the same attribute of the same subject, but B
   gives a different value: a new city, number, date, version, status, or a
   reversed decision. B would replace A; both cannot be true now.
3. related — A and B are about the same subject but different attributes.
   Both can be true at the same time.
4. unrelated — anything else.

The SUBJECT is the specific thing a fact is about: a project, tool, machine,
place, person, or area of preference. It is never just "the user" — almost
every fact mentions the user. "The user's paf test suite" and "the user's site
build" have different subjects.

Edge cases:
- B repeats A and adds a separate, unrelated claim → related.
- A says "X is not allowed", B says "X works now" → update.
- Same attribute, but different projects (paf's tests vs the site's tests)
  → unrelated.
- If you cannot decide between two labels, pick the better one and set
  "ambiguous": true. Do not invent context the facts do not state.

Worked examples:
- A: "The user prefers dark roast coffee." B: "The user's favorite coffee is a
  dark roast." → same
- A: "The user's 5K personal best is 27:12." B: "The user's 5K personal best is
  25:50." → update
- A: "The user's site is built with Astro." B: "The user's site has a /map page
  showing notes as a dependency graph." → related
- A: "The paf test suite takes about 50 seconds." B: "The site build takes
  about 10 seconds." → unrelated (different subjects)
- A: "The user writes notes in short sentences." B: "The user writes notes in
  short sentences and wants every note to end with a principles section." →
  related (B adds a separate claim)

Give a short reason (under 15 words) BEFORE each label.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "reason": {"type": "string"},
                    "label": {"type": "string", "enum": list(LABELS)},
                    "ambiguous": {"type": "boolean"},
                },
                "required": ["id", "reason", "label", "ambiguous"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def _post(url: str, body: dict, headers: dict | None = None, timeout: int = 180):
    req = urllib.request.Request(
        url,
        json.dumps(body).encode(),
        {"content-type": "application/json", **(headers or {})},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def _append(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _pair_view(c: dict) -> dict:
    return {
        "id": c["id"],
        "a": {"date": c["a"]["date"], "fact": c["a"]["fact"]},
        "b": {"date": c["b"]["date"], "fact": c["b"]["fact"]},
    }


def deepseek_batch(batch: list[dict], model: str) -> list[dict]:
    """Label one batch; retried once on a malformed / incomplete reply."""
    user = "Label each pair. Return one object per pair, in order.\n\n" + json.dumps(
        [_pair_view(c) for c in batch], ensure_ascii=False, indent=1
    )
    want = {c["id"] for c in batch}
    for attempt in range(3):
        try:
            r = _post(
                "https://openrouter.ai/api/v1/chat/completions",
                {
                    "model": model,
                    "temperature": 0,
                    "reasoning": {"effort": "medium"},
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "labels",
                            "strict": True,
                            "schema": SCHEMA,
                        },
                    },
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                    ],
                },
                {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
            )
            got = json.loads(r["choices"][0]["message"]["content"])["labels"]
            got = [g for g in got if g["id"] in want]
            if {g["id"] for g in got} == want:
                return [
                    {
                        "id": g["id"],
                        "label": g["label"],
                        "reason": g["reason"],
                        "ambiguous": g["ambiguous"],
                        "model": model,
                    }
                    for g in got
                ]
        except Exception as exc:  # noqa: BLE001 - retry, then give up on the batch
            print(f"  batch {batch[0]['id']}… attempt {attempt + 1} failed: {exc}")
        time.sleep(2)
    return []


def _unload(model: str) -> None:
    """Drop the model from Ollama memory so the next call loads a fresh copy."""
    try:
        _post(
            "http://127.0.0.1:11434/api/generate",
            {"model": model, "keep_alive": 0},
            timeout=60,
        )
    except Exception:  # noqa: BLE001 - best effort
        pass


def tev1_one(c: dict, model: str) -> dict:
    q = {
        "relation": {
            "type": "choice",
            "instructions": "How does fact_b (newer) relate to fact_a (older)?",
            "criteria": DEFS_V3,
        }
    }
    body = {
        "model": model,
        "state": {"fact_a": c["a"]["fact"], "fact_b": c["b"]["fact"]},
        "questions": q,
    }
    detail = ""
    for _attempt in range(2):
        try:
            ans = _post("http://127.0.0.1:11434/v1/systemone", body)["answers"][
                "relation"
            ]
            return {
                "id": c["id"],
                "label": ans["choice"],
                "confidence": round(ans["confidence"], 3),
                "probabilities": {
                    k: round(v, 3) for k, v in ans["probabilities"].items()
                },
                "model": model,
            }
        except urllib.error.HTTPError as exc:
            # Ollama 0.32 + tev1: some inputs return "scoring returned an
            # invalid candidate probability" AND leave the loaded model broken
            # (every later call fails). Unload so the next call gets a fresh
            # copy, then retry this pair once.
            detail = exc.read()[:200].decode(errors="replace")
            _unload(model)
    print(f"  tev1 failed on {c['id']}: {detail}")
    return {
        "id": c["id"],
        "label": None,
        "confidence": None,
        "probabilities": {},
        "model": model,
        "error": detail,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Model-label candidate pairs.")
    ap.add_argument("--agent", default="demo_assistant")
    ap.add_argument("--judge", choices=("deepseek", "tev1", "both"), default="both")
    ap.add_argument("--deepseek-model", default=DEEPSEEK_MODEL)
    ap.add_argument("--tev1-model", default="tev1:4b")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    d = PROJECT_ROOT / "data" / "memory" / args.agent / "relate-dataset"
    cands = _load(d / "candidates.jsonl")
    print(f"{len(cands)} candidates")

    if args.judge in ("deepseek", "both"):
        out = d / "labels-deepseek.jsonl"
        done = {r["id"] for r in _load(out)}
        todo = [c for c in cands if c["id"] not in done]
        batches = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]
        print(
            f"deepseek: {len(done)} done, {len(todo)} to label in {len(batches)} calls"
        )
        # Write each batch as it finishes (not in submission order), so one slow
        # or hung batch can't hold finished ones back; reruns resume from here.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(deepseek_batch, b, args.deepseek_model) for b in batches
            ]
            for n, fut in enumerate(as_completed(futures), 1):
                rows = fut.result()
                _append(out, rows)
                print(f"  {n}/{len(batches)} batches ({len(rows)} labeled)", flush=True)

    if args.judge in ("tev1", "both"):
        out = d / "labels-tev1.jsonl"
        done = {r["id"] for r in _load(out)}
        todo = [c for c in cands if c["id"] not in done]
        print(f"tev1: {len(done)} done, {len(todo)} to label")
        for n, c in enumerate(todo, 1):
            _append(out, [tev1_one(c, args.tev1_model)])
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}", flush=True)

    for judge in ("deepseek", "tev1"):
        rows = _load(d / f"labels-{judge}.jsonl")
        if rows:
            counts: dict[str, int] = {}
            for r in rows:
                counts[r["label"]] = counts.get(r["label"], 0) + 1
            print(f"{judge}: {len(rows)} labeled {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
