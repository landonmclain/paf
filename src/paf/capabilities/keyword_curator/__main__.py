"""Run the keyword curator by hand: ``python -m paf.capabilities.keyword_curator``.

Reads the skill libraries and vault from an agent's spec (so it curates what that
agent can see), keeps the store under ``data/keywords/<agent>/``, and prints a
review report. Calls a live model unless ``--dry-run`` / ``--report``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from paf.config import DATA_DIR

from ._background import make_batches, run_keyword_pass
from ._report import find_issues, format_issues
from ._sources import scan_sources, snapshots
from ._store import read_entity_map
from ._watermark import KeywordWatermark


def _spec_sources(agent: str) -> dict:
    from paf.specs import _spec_skills_config, _spec_vault_config, get_spec_path

    spec_path = get_spec_path(agent)
    skills = _spec_skills_config(spec_path) or {}
    vault = _spec_vault_config(spec_path) or {}
    return {
        "skill_libraries": skills.get("libraries") or [DATA_DIR / "skills"],
        "vault_root": vault.get("root"),
        "vault_allow": vault.get("allow") or [],
        "vault_deny": vault.get("deny") or [],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m paf.capabilities.keyword_curator")
    ap.add_argument("--agent", default="demo_assistant")
    ap.add_argument("--dir", help="keyword store dir (default data/keywords/<agent>)")
    ap.add_argument("--skills", action="append", help="skill library dir (repeatable)")
    ap.add_argument("--vault", help="vault root (default: from the agent spec)")
    ap.add_argument("--no-notes", action="store_true", help="skip the vault")
    ap.add_argument("--no-skills", action="store_true", help="skip skill libraries")
    ap.add_argument("--limit", type=int, help="curate at most N changed sources")
    ap.add_argument("--dry-run", action="store_true", help="plan only; no model call")
    ap.add_argument("--report", action="store_true", help="print the review report")
    args = ap.parse_args(argv)

    directory = Path(args.dir) if args.dir else DATA_DIR / "keywords" / args.agent

    if args.report and not args.dry_run:
        print(format_issues(find_issues(read_entity_map(directory))))
        return 0

    cfg = _spec_sources(args.agent)
    if args.skills:
        cfg["skill_libraries"] = args.skills
    if args.vault:
        cfg["vault_root"] = args.vault
    if args.no_notes:
        cfg["vault_root"] = None
    if args.no_skills:
        cfg["skill_libraries"] = []
    sources = scan_sources(**cfg)
    wm = KeywordWatermark.load(directory)
    changed = wm.changed(snapshots(sources))
    if args.limit is not None:
        changed = changed[: args.limit]
    n_skill = sum(1 for r in sources if r.startswith("skill:"))
    print(
        f"store: {directory}\nsources: {len(sources)} "
        f"({n_skill} skills, {len(sources) - n_skill} notes); "
        f"to curate: {len(changed)}; vanished: {len(wm.vanished(snapshots(sources)))}"
    )
    batches = make_batches([sources[r] for r in changed])
    print(f"batches: {len(batches)}")
    if args.dry_run:
        return 0

    result = asyncio.run(run_keyword_pass(directory, sources, limit=args.limit))
    print(
        f"keyed {len(result.keyed)}, failed {len(result.failed)}, "
        f"removed {len(result.removed)}; {result.entities} entities"
    )
    for ref, err in result.failed.items():
        print(f"  FAILED {ref}: {err}", file=sys.stderr)
    print("\n" + format_issues(find_issues(read_entity_map(directory))))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
