"""Deterministic live-tool test of the index_new near-match guard.

Drives `_build_index_toolset` against the REAL live `IndexStore(data/memory)`
— the exact prod wiring — and verifies:

1. The live index loads (37 entries worth of real disk state).
2. `index_new` a realistic paf fact stages it.
3. Re-adding a near-dup of THAT STAGED fact raises ModelRetry naming the id
   and steering to confirm/supersede (the Option B guard).
4. A genuinely distinct fact passes untouched.
5. NOTHING is committed to disk: Stage only touches disk on `finalize()`, which
   this test never calls, so live data/memory is byte-untouched.

Zero tokens: no LLM involved; the toolset is driven directly.
"""

import asyncio
import hashlib
import sys

from pydantic_ai import ModelRetry

from archive.organizer import IndexStore
from archive.organizer._organize import _build_index_toolset

MEM_DIR = "data/memory"


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


async def main() -> int:
    fails = []

    # --- Baseline: hash live files to prove no disk mutation ---
    before = {p: sha(f"{MEM_DIR}/{p}") for p in
              ("memory-index.json", "memory-journal.jsonl", "MEMORY.md")}

    store = IndexStore(MEM_DIR)
    toolset = _build_index_toolset()(store)

    # --- 1. Guard loads against real live state ---
    live = store.load()
    print(f"[1] live index loaded: {len(live)} entries, "
          f"watermark={store.load_watermark()}")
    assert len(live) >= 30, "expected a populated live index"

    # --- 2. Stage a realistic near-dup of an existing LIVE pattern ---
    # (mem-29/mem-36 both say "paf already has a built-in sub-agent mechanism")
    out = await toolset.add_new(
        {}, line_no=999901, type="fact", subject="paf",
        fact="paf already has a built-in sub-agent mechanism",
        born_on="2026-08-23T00:00:00+00:00", source_ref="live-test:0",
    )
    new_id = [e.id for e in toolset._get_stage().entries][-1]
    print(f"[2] staged new entry {new_id} (post-edit staged={len(toolset._get_stage().entries)})")
    assert out["status"] == "updated"

    # --- 3. Re-add a NEAR-DUP of the just-staged fact -> ModelRetry ---
    try:
        await toolset.add_new(
            {}, line_no=999902, type="fact", subject="paf",
            fact="paf has a built-in sub-agent mechanism already",
            born_on="2026-08-23T00:00:00+00:00", source_ref="live-test:1",
        )
        print("[3] FAIL: near-dup did NOT raise ModelRetry")
        fails.append("near-dup not rejected")
    except ModelRetry as e:
        msg = str(e)
        print(f"[3] near-dup -> ModelRetry: {msg[:120]}")
        assert new_id in msg, "retry must name the staged entry id"
        assert "index_confirm" in msg and "index_supersede" in msg
        # Nothing added for the dup line -> staged count unchanged
        assert len(toolset._get_stage().entries) == len(live) + 1, \
            "dup line must not create an extra staged entry"
        print(f"     staged unchanged at {len(toolset._get_stage().entries)}")

    # --- 4. Genuinely distinct fact passes ---
    out2 = await toolset.add_new(
        {}, line_no=999903, type="fact", subject="paf",
        fact="paf's CLI entrypoint lives in the ui/ directory",
        born_on="2026-08-23T00:00:00+00:00", source_ref="live-test:2",
    )
    print(f"[4] distinct fact accepted: status={out2['status']}, "
          f"staged={len(toolset._get_stage().entries)}")
    assert out2["status"] == "updated"

    # --- 5. No disk commit (no finalize) ---
    after = {p: sha(f"{MEM_DIR}/{p}") for p in before}
    changed = [p for p in before if before[p] != after[p]]
    print(f"[5] disk mutation check: changed={changed or 'NONE'}")
    assert not changed, f"live files mutated: {changed}"

    if fails:
        print("\nFAILURES:", fails)
        return 1
    print("\nALL LIVE-GUARD CHECKS PASSED (0 tokens, live disk untouched)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
