"""p1-gate: extract real paf python files via the ship path, report per-file results.

Walks the paf ``src/`` tree, runs ``analyze_file`` on every ``.py``, and reports:
- files skipped (unsupported/oversized/unparseable/no-symbols) with reasons
- per-file symbol counts + a sampled symbol dump for hand-inspection
- any process() exceptions (should be none on our own code)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from paf.capabilities.codebase_map._analyze import analyze_file

ROOT = Path("/home/alice/projects/Personal/pydantic-agent-framework")


def main(top: str = "src/paf", show: int = 0) -> None:
    scan = ROOT / top
    py_files = sorted(scan.rglob("*.py"))
    extracted, skipped_empty, skipped_err, total_symbols = [], [], [], 0

    for f in py_files:
        try:
            fi = analyze_file(f, ROOT)  # scan_root = repo root => repo-relative paths
        except Exception as e:  # noqa: BLE001 — gate must never crash
            skipped_err.append((str(f.relative_to(ROOT)), repr(e)))
            continue
        if fi is None:
            skipped_empty.append(str(f.relative_to(ROOT)))
            continue
        extracted.append(fi)
        total_symbols += len(fi.searchable_symbols)

    print(f"== scan_root={scan} ==")
    print(f"python files: {len(py_files)}")
    print(f"extracted: {len(extracted)}  no-symbols/skip: {len(skipped_empty)}  error: {len(skipped_err)}")
    print(f"total searchable symbols: {total_symbols}")

    if skipped_err:
        print("\n-- ERRORS (should be none on own code) --")
        for p, e in skipped_err[:20]:
            print(f"  {p}: {e}")

    if skipped_empty:
        print(f"\n-- skipped no-symbols ({len(skipped_empty)}) --")
        for p in skipped_empty[:30]:
            print(f"  {p}")

    print("\n== per-file symbol counts (top by count) ==")
    rows = sorted(extracted, key=lambda fi: -len(fi.searchable_symbols))
    for fi in rows[:25]:
        print(f"{len(fi.searchable_symbols):4d}  {fi.path}  [lang={fi.language}]")

    if show:
        print(f"\n== sampled symbol detail ({show} files, top-level only) ==")
        for fi in rows[:show]:
            print(f"\n### {fi.path}")
            if fi.doc:
                print(f"  MODULE DOC: {fi.doc[:120]!r}")
            if fi.imports:
                print(f"  IMPORTS({len(fi.imports)}): {fi.imports[:8]}")
            for s in fi.searchable_symbols:
                if s.parent is None:  # top-level
                    print(
                        f"  {s.kind:<9} {s.qualified}"
                        f"{('  <= ' + s.signature) if s.signature else ''}"
                    )
                    if s.doc:
                        print(f"            doc: {s.doc[:100]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", default="src/paf")
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()
    main(args.top, args.show)
