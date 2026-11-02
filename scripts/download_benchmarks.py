#!/usr/bin/env python3
"""Download the benchmark datasets the harness needs.

The benchmark data is not in the repo (``data/`` is gitignored), so a fresh
clone cannot run ``tests/run_lmem.py`` until this has been run once.

    uv run scripts/download_benchmarks.py            # fetch everything
    uv run scripts/download_benchmarks.py --list     # what is available
    uv run scripts/download_benchmarks.py --only longmemeval-oracle
    uv run scripts/download_benchmarks.py --force    # re-download

Datasets are verified against a pinned SHA-256 and checked for the schema the
harness reads, so a bad or partial download never lands in ``data/benchmarks/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "data" / "benchmarks"

# name -> (url, destination, pinned sha256). Pinned so a run is reproducible:
# if upstream changes, the download fails loudly instead of silently shifting
# the benchmark's ground truth.
DATASETS: dict[str, dict[str, object]] = {
    "longmemeval-oracle": {
        "url": (
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
            "resolve/main/longmemeval_oracle.json"
        ),
        "dest": BENCHMARKS_DIR / "memory-v3-lme_oracle.json",
        "sha256": "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",
        "question_count": 500,
    },
}

# Keys tests/run_lmem.py indexes into.
REQUIRED_KEYS = {
    "question_id",
    "question_type",
    "question",
    "answer",
    "question_date",
    "haystack_dates",
    "haystack_session_ids",
    "haystack_sessions",
    "answer_session_ids",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _display(path: Path) -> str:
    """Repo-relative when it is inside the repo, absolute otherwise."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def validate(payload: object, spec: dict[str, object]) -> str:
    """Check the schema the harness reads. Returns a human-readable summary."""
    if not isinstance(payload, list):
        raise ValueError("expected a top-level JSON array of questions")

    expected = int(spec["question_count"])
    if len(payload) != expected:
        raise ValueError(f"expected {expected} questions, got {len(payload)}")

    for index, question in enumerate(payload):
        if not isinstance(question, dict):
            raise ValueError(f"question {index} is not an object")
        missing = REQUIRED_KEYS - question.keys()
        if missing:
            raise ValueError(f"question {index} is missing {sorted(missing)}")
        if not question["haystack_sessions"]:
            raise ValueError(f"question {question['question_id']} has no sessions")
        if not set(question["answer_session_ids"]) <= set(
            question["haystack_session_ids"]
        ):
            raise ValueError(
                f"question {question['question_id']} lists answer sessions "
                "that are not in the haystack"
            )

    sessions = sum(len(q["haystack_sessions"]) for q in payload)
    turns = sum(sum(len(s) for s in q["haystack_sessions"]) for q in payload)
    types = sorted({q["question_type"] for q in payload})
    return (
        f"{len(payload)} questions, {sessions / len(payload):.2f} sessions/question, "
        f"{turns / len(payload):.2f} turns/question, {len(types)} types: {', '.join(types)}"
    )


def download(name: str, spec: dict[str, object], force: bool) -> bool:
    dest = Path(spec["dest"])
    pinned = str(spec["sha256"])

    if dest.exists() and not force:
        digest = sha256(dest.read_bytes())
        if digest == pinned:
            print(f"{name}: already present and verified ({_display(dest)})")
            return True
        print(f"{name}: existing file has the wrong checksum, re-downloading")

    url = str(spec["url"])
    print(f"{name}: fetching {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            raw = response.read()
    except (urllib.error.URLError, OSError) as exc:
        print(f"{name}: download failed: {exc}", file=sys.stderr)
        return False

    digest = sha256(raw)
    if digest != pinned:
        print(
            f"{name}: checksum mismatch\n"
            f"  expected {pinned}\n"
            f"  got      {digest}\n"
            "  upstream may have changed; update the pin in this script if intentional.",
            file=sys.stderr,
        )
        return False

    try:
        summary = validate(json.loads(raw), spec)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"{name}: schema validation failed: {exc}", file=sys.stderr)
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    print(f"{name}: wrote {_display(dest)} ({len(raw):,} bytes) — {summary}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        help="download a single dataset by name",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list available datasets and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even when the file is already present and verified",
    )
    args = parser.parse_args()

    if args.list:
        for name, spec in DATASETS.items():
            print(f"{name} -> {_display(Path(spec['dest']))}")
        return 0

    if args.only and args.only not in DATASETS:
        print(
            f"unknown dataset {args.only!r}; known: {', '.join(DATASETS)}",
            file=sys.stderr,
        )
        return 2

    names = [args.only] if args.only else list(DATASETS)

    failed = [name for name in names if not download(name, DATASETS[name], args.force)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
