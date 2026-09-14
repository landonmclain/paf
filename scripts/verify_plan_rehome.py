"""Task 7 E2E repro of the original planning bug, driven through the REAL toolset.

Mirrors the traced production failure: the model calls write_plan (auto-gen ids),
then update_task_statuses with *guessed* ids -> "Step with id '1' not found".
With the id-echo fix, a follow-up update_task_statuses using the echoed [id]s must
NOT fail, and the FilePlanStore must persist the status change to
data/plans/<cid>/plan.json.

Drives the actual vendored PlanningToolset + capability via the real store_resolver
path (resolve_store -> store_resolver(ctx)), so it exercises production code end to
end without an API call or a service restart. The fake ctx only needs the
`conversation_id` attribute because the _build_planning resolver reads exactly that.

Usage: PYTHONPATH= .venv/bin/python scripts/verify_plan_rehome.py
"""
from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

from pydantic_ai.tools import RunContext

from paf.capabilities.planning import (
    FilePlanStore,
    PlanItem,
    Planning,
    PlanStatusUpdate,
    TaskStatus,
)

CONVERSATION_ID = "verify-e2e"


class _FakeRunContext:
    """Minimal stand-in for RunContext: the store_resolver only reads `conversation_id`."""

    conversation_id: str

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        def resolver(ctx) -> FilePlanStore:
            cid = getattr(ctx, "conversation_id", None) or "default"
            return FilePlanStore(root=root, conversation_id=cid)

        capability = Planning[RunContext[object]](store_resolver=resolver)
        toolset = capability.get_toolset()
        ctx = _FakeRunContext(CONVERSATION_ID)

        # 1) write_plan — the model passes items WITHOUT ids (as it does today).
        plan_items = [
            PlanItem(content="Scaffold the project"),
            PlanItem(content="Write the tests"),
            PlanItem(content="Ship it"),
        ]
        write_reply = await toolset.write_plan(ctx, items=plan_items)
        print("=== write_plan reply ===")
        print(write_reply)

        # 2) The model follows the id-echo: copy the bracketed [id]s verbatim.
        ids = re.findall(r"\[([0-9a-f]{8})\]|\[([0-9a-f-]{36})\]|\[([0-9A-Fa-f-]{8,36})\]", write_reply)
        echoed = [a or b or c for a, b, c in ids]
        # Fall back to full-uuid tokens if the short form isn't echoed.
        if not echoed:
            echoed = re.findall(r"\[([0-9A-Fa-f]{8,36})\]", write_reply)
        print("=== echoed ids from write_plan ===")
        print(echoed)
        if len(echoed) != 3:
            raise SystemExit(f"FAIL: expected 3 echoed ids, got {len(echoed)}: {write_reply!r}")

        # 3) update_task_statuses using the REAL echoed ids (the old code guessed "1","2" -> crash).
        updates = [PlanStatusUpdate(task_id=echoed[0], status=TaskStatus.in_progress)]
        update_reply = await toolset.update_task_statuses(ctx, updates=updates)
        print("=== update_task_statuses reply (must not say 'not found') ===")
        print(update_reply)
        assert "not found" not in update_reply.lower(), f"FAIL: 'not found' in reply: {update_reply!r}"

        # 4) Verify the FilePlanStore persisted the status change to disk.
        plan_file = root / CONVERSATION_ID / "plan.json"
        assert plan_file.exists(), f"FAIL: plan.json not written at {plan_file}"
        plan_json = plan_file.read_text()
        assert "'in_progress'" in plan_json or '"in_progress"' in plan_json, (
            f"FAIL: status not persisted to {plan_file}:\n{plan_json}"
        )

        # 5) read_plan round-trips the current state (ids stable, one in_progress).
        read_reply = await toolset.read_plan(ctx)
        print("=== read_plan reply ===")
        print(read_reply)
        item = (await capability.resolve_store(ctx).get_items())[0]
        assert item.id == echoed[0] and item.status is TaskStatus.in_progress, (
            f"FAIL: store item mismatch: id={item.id!r} status={item.status}"
        )

        print("\n=== PLAN REHOME E2E VERIFY: PASS ===")
        print(f"plan file: {plan_file}")


if __name__ == "__main__":
    asyncio.run(main())
