"""End-to-end smoke test of analyze_file on real paf source files."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from paf.capabilities.codebase_map._analyze import analyze_file

SCAN_ROOT = Path("/home/alice/projects/Personal/pydantic-agent-framework")

FILES = [
    "src/paf/capabilities/codebase_map/_queries.py",
    "src/paf/capabilities/codebase_map/_schema.py",
    "src/paf/config.py",
]

for f in FILES:
    print(f"\n=== {f} ===")
    fi = analyze_file(SCAN_ROOT / f, SCAN_ROOT)
    if fi is None:
        print("  -> None (skipped)")
        continue
    print(f"  language={fi.language}  symbols={fi.symbols}")
    print(f"  imports={fi.imports}")
    for s in fi.searchable_symbols[:6]:
        print(f"    {s.qualified}  [{s.kind}]  sig={s.signature!r}  doc={s.doc!r}")
