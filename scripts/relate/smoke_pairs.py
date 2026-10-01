"""Run the six smoke pairs through one model and print label, confidence, timing.

Backends:
  systemone   — Ollama's Jev-style decision endpoint (/v1/systemone), for
                decision models such as tev1:4b / tev1:0.8b. Returns calibrated
                per-option probabilities and a confidence.
  ollama-chat — Ollama /api/chat with a JSON-schema-constrained label and token
                logprobs (gemma3:4b, qwen3.5:4b). Logprob "probabilities" are
                the label token's top alternatives — uncalibrated.
  openrouter  — OpenRouter chat completions with a strict JSON schema (labels
                only). Needs OPENROUTER_API_KEY (source .env with set -a).

Usage (from the repo root):
    PYTHONPATH= .venv/bin/python scripts/relate/smoke_pairs.py --backend systemone --model tev1:4b --defs v2
    PYTHONPATH= .venv/bin/python scripts/relate/smoke_pairs.py --backend ollama-chat --model gemma3:4b
    set -a && . ./.env && set +a && PYTHONPATH= .venv/bin/python scripts/relate/smoke_pairs.py \\
        --backend openrouter --model deepseek/deepseek-v4-flash-0731
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request

from pairs import DEFS, LABELS, PAIRS, system_prompt

OLLAMA = "http://127.0.0.1:11434"
SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": list(LABELS)}},
    "required": ["label"],
    "additionalProperties": False,
}


def _post(url: str, body: dict, headers: dict | None = None) -> dict:
    req = urllib.request.Request(
        url,
        json.dumps(body).encode(),
        {"content-type": "application/json", **(headers or {})},
    )
    return json.load(urllib.request.urlopen(req, timeout=300))


def systemone(model: str, defs: dict, a: str, b: str):
    q = {
        "relation": {
            "type": "choice",
            "instructions": "How does fact_b relate to fact_a?",
            "criteria": defs,
        }
    }
    r = _post(
        f"{OLLAMA}/v1/systemone",
        {"model": model, "state": {"fact_a": a, "fact_b": b}, "questions": q},
    )
    ans = r["answers"]["relation"]
    return ans["choice"], ans["confidence"], ans["probabilities"]


def ollama_chat(model: str, defs: dict, a: str, b: str):
    r = _post(
        f"{OLLAMA}/api/chat",
        {
            "model": model,
            "stream": False,
            "think": False,
            "format": SCHEMA,
            "logprobs": True,
            "top_logprobs": 4,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system_prompt(defs)},
                {"role": "user", "content": f"A: {a}\nB: {b}"},
            ],
        },
    )
    label = json.loads(r["message"]["content"])["label"]
    # Top alternatives at the first token that starts a label (rough, uncalibrated).
    probs = None
    for tok in r.get("logprobs") or []:
        t = tok.get("token", "").strip('"')
        if t and any(lbl.startswith(t[:2]) for lbl in LABELS):
            probs = {
                x["token"]: math.exp(x["logprob"]) for x in tok.get("top_logprobs", [])
            }
            break
    return label, None, probs


def openrouter(model: str, defs: dict, a: str, b: str):
    r = _post(
        "https://openrouter.ai/api/v1/chat/completions",
        {
            "model": model,
            "temperature": 0,
            "reasoning": {"effort": "low"},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "r", "strict": True, "schema": SCHEMA},
            },
            "messages": [
                {"role": "system", "content": system_prompt(defs)},
                {"role": "user", "content": f"A: {a}\nB: {b}"},
            ],
        },
        {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
    )
    label = json.loads(r["choices"][0]["message"]["content"])["label"]
    return label, None, {"tokens": r["usage"]["total_tokens"]}


BACKENDS = {
    "systemone": systemone,
    "ollama-chat": ollama_chat,
    "openrouter": openrouter,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Six-pair relation smoke test.")
    ap.add_argument("--backend", choices=BACKENDS, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--defs", choices=DEFS, default="v1")
    args = ap.parse_args()

    run, defs, ok = BACKENDS[args.backend], DEFS[args.defs], 0
    for a, b, gold in PAIRS:
        t = time.time()
        label, conf, probs = run(args.model, defs, a, b)
        dt = time.time() - t
        ok += label == gold
        conf_s = f" conf={conf:.2f}" if conf is not None else ""
        probs_s = (
            {k: round(v, 2) if isinstance(v, float) else v for k, v in probs.items()}
            if probs
            else ""
        )
        print(f"{dt:5.2f}s gold={gold:9s} got={label:9s}{conf_s} {probs_s}")
    print(f"{ok}/{len(PAIRS)} correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
