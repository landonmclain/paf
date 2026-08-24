"""Run the codebase-map index from the shell.

Usage::

    python -m paf.capabilities.codebase_map.cli [--root PATH] [--json]

Default ``PATH`` is the current working directory (cwd root). Rebuilds all codebase-map
artifacts (overview / index / manifest / tokens / snapshot) for that root via
``run_codebase_map`` and prints the summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import run_codebase_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="codebase-map",
        description="Re-index a repository's codebase map (overview/index/manifest/tokens/snapshot).",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Scan root (default: current working directory root).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    res = run_codebase_map(Path(args.root) if args.root else None)

    if args.json:
        json.dump(res, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"Re-indexed {res['scan_root']}")
        print(f"{res['extracted']} files extracted, {res['total_symbols']} symbols")
        print(f"{len(res['unparsed'])} unparsed, {len(res['skipped'])} skipped")
        print(f"index at {res['index_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
