"""Terminal tool: label the pairs ``select_set.py`` flagged for human review.

Shows one pair at a time — older fact A, newer fact B, their dates — and takes
a single keypress. The judges' labels are hidden until you answer (so they
can't anchor you), then shown briefly so a slip is easy to spot and undo.

Keys:  1 same   2 update   3 different (keep both; 4 also works)
       s skip   u undo last   q quit   ? help

Only same / update drive a reviewer action; related vs unrelated never does,
so both are one bucket: "different". Scoring merges the judges' related and
unrelated into "different" too.

Labels are saved after every keypress to ``human-labels.jsonl`` (one row per
pair id; the latest answer wins), so quitting and resuming loses nothing.

Usage (from the repo root):
    PYTHONPATH= .venv/bin/python scripts/relate/review_pairs.py
    PYTHONPATH= .venv/bin/python scripts/relate/review_pairs.py --all   # every row
    PYTHONPATH= .venv/bin/python scripts/relate/review_pairs.py --redo  # incl. labeled
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
import time
from pathlib import Path

from paf.config import PROJECT_ROOT

# Labels that all mean "keep both" — the reviewer acts only on same / update.
BUCKET_DIFFERENT = {"related", "unrelated", "different"}

KEYS = {"1": "same", "2": "update", "3": "different", "4": "different"}
HELP = """\
  1 same       one fact is fully covered by the other: they say the same thing,
               OR one says everything the other does plus more (the one that
               adds nothing would be archived)
  2 update     same specific attribute, value changed; B replaces A
  3 different  each says something the other doesn't (partial overlap), or a
               different topic entirely. Keep both. (4 also = different)
  s skip   u undo last   q quit
"""
DIM, BOLD, CYAN, YELLOW, GREEN, RESET = (
    ("\x1b[2m", "\x1b[1m", "\x1b[36m", "\x1b[33m", "\x1b[32m", "\x1b[0m")
    if sys.stdout.isatty()
    else ("",) * 6
)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def _getkey() -> str:
    """One keypress without Enter (falls back to a line read off a tty)."""
    if not sys.stdin.isatty():
        return (sys.stdin.readline() or "q").strip()[:1]
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _wrap(text: str, indent: str) -> str:
    width = max(40, shutil.get_terminal_size((100, 24)).columns - len(indent) - 2)
    return textwrap.fill(text, width, initial_indent=indent, subsequent_indent=indent)


def main() -> int:
    ap = argparse.ArgumentParser(description="Human-label flagged pairs.")
    ap.add_argument("--agent", default="demo_assistant")
    ap.add_argument(
        "--all", action="store_true", help="review every row, not just flagged"
    )
    ap.add_argument("--redo", action="store_true", help="include already-labeled rows")
    ap.add_argument(
        "--only",
        metavar="LABEL",
        help="revisit only pairs you already labeled LABEL (e.g. --only update)",
    )
    args = ap.parse_args()

    d = PROJECT_ROOT / "data" / "memory" / args.agent / "relate-dataset"
    rows = _load(d / "dataset.jsonl")
    if not rows:
        print("no dataset.jsonl — run select_set.py first")
        return 2
    path = d / "human-labels.jsonl"
    human = {r["id"]: r for r in _load(path)}

    def save() -> None:
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in human.values()),
            encoding="utf-8",
        )

    if args.only:
        queue = [r for r in rows if human.get(r["id"], {}).get("label") == args.only]
        total = len(queue)
    else:
        queue = [
            r
            for r in rows
            if (args.all or r["needs_review"]) and (args.redo or r["id"] not in human)
        ]
        total = sum(1 for r in rows if args.all or r["needs_review"])
    if args.only:
        print(f"{len(queue)} pairs you labeled {args.only!r} (s keeps the old label)\n")
    else:
        print(f"{len(queue)} to label ({total - len(queue)} of {total} already done)\n")
    print(HELP)

    history: list[str] = []
    i = 0
    while i < len(queue):
        r = queue[i]
        print(f"{DIM}{'─' * 60}{RESET}")
        print(
            f"{BOLD}{total - len(queue) + i + 1}/{total}{RESET}  {DIM}{r['id']} · "
            f"{r['split']} · {r['source']} · flagged: {', '.join(r['review_reason']) or '-'}"
            f"{RESET}"
        )
        prev = human.get(r["id"], {}).get("label")
        if prev:
            print(f"  {DIM}your previous label: {prev}{RESET}")
        print(f"{CYAN}A{RESET} {DIM}{r['a']['date']}{RESET}")
        print(_wrap(r["a"]["fact"], "  "))
        print(f"{YELLOW}B{RESET} {DIM}{r['b']['date']}{RESET}")
        print(_wrap(r["b"]["fact"], "  "))
        print(
            f"{DIM}[1 same · 2 update · 3 different · s · u · q · ?]{RESET} ",
            end="",
            flush=True,
        )
        key = _getkey()
        print(key)
        if key == "q":
            break
        if key == "?":
            print(HELP)
            continue
        if key == "s":
            i += 1
            continue
        if key == "u":
            if history:
                last = history.pop()
                human.pop(last, None)
                save()
                i = max(0, i - 1)
                print(f"{DIM}undid {last}{RESET}")
            continue
        label = KEYS.get(key)
        if label is None:
            continue
        human[r["id"]] = {"id": r["id"], "label": label, "at": time.time()}
        history.append(r["id"])
        save()
        ds, tv = r["deepseek"], r["tev1"]

        def mark(x: str, label: str = label) -> str:
            same = x == label or (label == "different" and x in BUCKET_DIFFERENT)
            return f"{GREEN}{x}{RESET}" if same else x

        print(
            f"  {DIM}you:{RESET} {BOLD}{label}{RESET}   "
            f"{DIM}deepseek:{RESET} {mark(ds['label'])} {DIM}({ds['reason']}){RESET}   "
            f"{DIM}tev1:{RESET} {mark(tv['label'] or '—')} "
            f"{DIM}{tv['confidence'] if tv['confidence'] is None else round(tv['confidence'], 2)}{RESET}\n"
        )
        i += 1

    done = sum(1 for r in rows if (args.all or r["needs_review"]) and r["id"] in human)
    print(f"\nlabeled {done}/{total}; saved to {path}")
    counts = {"same": 0, "update": 0, "different": 0}
    for h in human.values():
        lbl = h["label"]
        counts["different" if lbl in BUCKET_DIFFERENT else lbl] += 1
    print("your labels:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
