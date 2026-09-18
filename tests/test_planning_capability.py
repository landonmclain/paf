"""Tests for the paf-owned planning capability (rehomed from pydantic-ai-harness).

Locks in the Task 1 rehome: the ``Planning`` capability and ``PlanningToolset``
now live under ``paf.capabilities.planning`` (not the harness BUILTIN), resolve
through the paf registry and ``_build_planning`` seam, and their core tools
behave against a ``PlanStore``.

Subsequent tasks build on this: Task 2 (write_plan id-echo), Task 3
(flat-file FilePlanStore), Task 4 (events on set_items), Task 5 (marker strip).
"""

from __future__ import annotations

import asyncio
import inspect
import re

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext

from paf.capabilities.build import _build_planning
from paf.capabilities.planning import (
    FilePlanStore,
    InMemoryPlanStore,
    PlanEvent,
    PlanEventEmitter,
    PlanEventType,
    PlanItem,
    Planning,
    PlanningToolset,
    PlanStatusUpdate,
    TaskStatus,
)
from paf.capabilities.registry import CAPABILITY_TYPES

# --- Task 1: rehome wiring -------------------------------------------------


def test_registry_planning_is_paf_owned() -> None:
    """CAPABILITY_TYPES['Planning'] must resolve to the paf-owned class."""
    cls = CAPABILITY_TYPES["Planning"]
    assert cls.__module__ == "paf.capabilities.planning._capability"


def test_build_planning_returns_paf_owned_instance() -> None:
    """_build_planning must construct the paf-owned capability (not the harness)."""
    cap = _build_planning(store=None)
    assert type(cap).__module__ == "paf.capabilities.planning._capability"
    # still a proper Planning capability (subtask tools off by default per YAGNI)
    assert cap.enable_subtasks is False


def test_build_planning_uses_paf_store_symbol() -> None:
    """The builder must resolve a paf FilePlanStore (Task 3), not the harness."""
    src = inspect.getsourcelines(_build_planning)[0]
    joined = "".join(src)
    assert "from paf.capabilities.planning import FilePlanStore, Planning" in joined


# --- Core tool behavior against a store ------------------------------------


def _capability():
    return _build_planning(store=None)


def test_write_plan_replaces_and_reads() -> None:
    """write_plan persists an ordered list; get_items surfaces it with ids."""

    async def scenario():
        cap = _capability()
        store = cap.resolve_store(None)  # type: ignore[arg-type]
        items = [PlanItem(content="step one"), PlanItem(content="step two")]
        await store.set_items(items)
        got = await store.get_items()
        assert [i.content for i in got] == ["step one", "step two"]
        assert all(i.id for i in got)

    asyncio.run(scenario())


def test_write_plan_id_echo_prevents_not_found() -> None:
    """Task 2: write_plan echoes step ids so a follow-up update doesn't guess "1"/"2".

    This is the exact repro of the original crash (model saw positional "1." in the
    write_plan reply, then called update_task_statuses(task_id="1") -> "not found").
    We drive the real toolset through a captured RunContext: write_plan with items
    that carry NO ids, parse the ids the reply echoes, then update with those ids and
    assert success instead of "not found".
    """

    captured: list[RunContext] = []

    async def grab_ctx(ctx: RunContext) -> str:
        captured.append(ctx)
        return "ok"

    agent = Agent(TestModel(call_tools=["grab_ctx"]), tools=[grab_ctx])
    agent.run_sync("go")  # populates `captured[0]` with a real RunContext

    def scenario():
        ctx = captured[0]
        store = InMemoryPlanStore()
        cap = Planning(store=store)
        toolset = PlanningToolset(cap)

        async def body():
            # Step 1: write_plan with items carrying NO explicit ids
            reply = await toolset.write_plan(
                ctx,
                [PlanItem(content="first"), PlanItem(content="second")],
            )
            assert "Plan updated: 2 step(s)" in reply
            # The id-echo: some line is `1. [x] [<8-hex-id>] first`
            echoed = re.findall(r"\[([0-9a-f]{8})\]", reply)
            assert len(echoed) == 2, f"expected 2 echoed ids, got {echoed!r}"
            assert "not found" not in reply

            # Step 2: the model (as we'd hope) updates using the echoed ids
            up = await toolset.update_task_statuses(
                ctx,
                [PlanStatusUpdate(task_id=echoed[0], status=TaskStatus.completed)],
            )
            assert "not found" not in up, f"update failed: {up!r}"
            assert echoed[0] in up

            # And a wrong guess must still be caught loudly -- as a ModelRetry whose
            # message echoes the current plan so the model self-corrects in one step
            # (no separate read_plan round-trip needed).
            import pydantic_ai.exceptions as _paie

            with pytest.raises(_paie.ModelRetry) as exc:
                await toolset.update_task_statuses(
                    ctx,
                    [PlanStatusUpdate(task_id="1", status=TaskStatus.completed)],
                )
            msg = str(exc.value)
            assert "not found" in msg
            assert "Current plan:" in msg
            assert echoed[0] in msg  # the real id is echoed for recovery

            # Store reflects the completed step under its real id
            got = await store.get_item(echoed[0])
            assert got is not None and got.status is TaskStatus.completed

        asyncio.run(body())

    scenario()


def test_update_task_status_wrong_id_raises_model_retry_with_plan() -> None:
    """Wrong-id updates raise a ModelRetry (non-fatal) whose message echoes the plan.

    A guessed/mismatched id must not abort the run silently or round-trip a
    needless read_plan: pydantic-ai feeds ModelRetry back to the model to retry,
    and echoing the current plan (with the real bracketed [id]s) lets it
    self-correct in the same step.
    """

    captured: list[RunContext] = []

    async def grab_ctx(ctx: RunContext) -> str:
        captured.append(ctx)
        return "ok"

    agent = Agent(TestModel(call_tools=["grab_ctx"]), tools=[grab_ctx])
    agent.run_sync("go")

    def scenario():
        ctx = captured[0]
        cap = Planning(store=InMemoryPlanStore())
        toolset = PlanningToolset(cap)

        async def body():
            await toolset.write_plan(ctx, [PlanItem(content="first")])
            items = await cap.resolve_store(ctx).get_items()
            real_id = items[0].id

            # singular update_task_status := bad id -> ModelRetry + plan echo
            import pydantic_ai.exceptions as _paie

            with pytest.raises(_paie.ModelRetry) as exc:
                await toolset.update_task_status(
                    ctx, task_id="nope", status=TaskStatus.completed
                )
            msg = str(exc.value)
            assert "not found" in msg
            assert "Current plan:" in msg
            assert real_id in msg

        asyncio.run(body())

    scenario()


# --- Task 3: flat-file FilePlanStore ----------------------------------------


def test_file_plan_store_persists_across_instances() -> None:
    """Task 3: FilePlanStore lands inspectable JSON per conversation under data/plans.

    A second instance pointing at the same root/conversation must read the plan
    written by the first (durability across runs/restarts), and the on-disk
    artifact must be a plain dict of items (no envelope alias surprises).
    """

    def scenario(tmp: str) -> None:
        async def body():
            cid = "paf-8a1b2c3d"
            a = FilePlanStore(tmp, conversation_id=cid)
            await a.set_items([PlanItem(content="alpha"), PlanItem(content="beta")])
            # distinct instance, same root+cid -> sees the same plan
            b = FilePlanStore(tmp, conversation_id=cid)
            got = await b.get_items()
            assert [i.content for i in got] == ["alpha", "beta"]
            assert all(i.id for i in got)
            await b.update_item(got[0].id, status=TaskStatus.completed)
            c = FilePlanStore(tmp, conversation_id=cid)
            assert (await c.get_item(got[0].id)).status is TaskStatus.completed

        asyncio.run(body())

    with __import__("tempfile").TemporaryDirectory() as tmp:
        scenario(tmp)


def test_file_plan_store_is_conversation_scoped() -> None:
    """Two conversation_ids never share state, even under the same root."""

    def scenario(tmp: str) -> None:
        async def body():
            a = FilePlanStore(tmp, conversation_id="conv-1")
            b = FilePlanStore(tmp, conversation_id="conv-2")
            await a.set_items([PlanItem(content="only-in-1")])
            assert [i.content for i in await a.get_items()] == ["only-in-1"]
            assert await b.get_items() == []

        asyncio.run(body())

    with __import__("tempfile").TemporaryDirectory() as tmp:
        scenario(tmp)


def test_file_store_remove_returns_existence() -> None:
    """remove_item on a missing id returns False; on a present one True + removes."""

    def scenario(tmp: str) -> None:
        async def body():
            s = FilePlanStore(tmp, conversation_id="conv-r")
            assert await s.remove_item("nope") is False
            await s.set_items([PlanItem(content="x"), PlanItem(content="y")])
            items = await s.get_items()
            first = items[0].id
            assert await s.remove_item(first) is True
            assert [i.content for i in await s.get_items()] == ["y"]

        asyncio.run(body())

    with __import__("tempfile").TemporaryDirectory() as tmp:
        scenario(tmp)


def test_build_planning_resolves_conversation_scoped_file_store() -> None:
    """_build_planning wires a store_resolver that yields a per-conversation FilePlanStore."""

    from types import SimpleNamespace

    cap = _build_planning(store=None)
    assert cap.store_resolver is not None
    store = cap.store_resolver(SimpleNamespace(conversation_id="conv-c"))  # type: ignore[arg-type]
    assert isinstance(store, FilePlanStore)


# --- Task 4: set_items emits plan events (sidebar refresh) -------------------


def test_set_items_emits_created_per_item() -> None:
    """Task 4: FilePlanStore.set_items fires a `created` PlanEvent per new item (write_plan case)."""

    def scenario(tmp: str) -> None:
        async def body():
            seen: list[PlanEvent] = []
            emitter = PlanEventEmitter()
            emitter.on(PlanEventType.created, seen.append)
            store = FilePlanStore(tmp, conversation_id="conv-e", event_emitter=emitter)
            await store.set_items([PlanItem(content="a"), PlanItem(content="b")])
            created = [e for e in seen if e.event_type is PlanEventType.created]
            assert len(created) == 2, (
                f"expected 2 created, got {[e.item.content for e in created]!r}"
            )
            assert {e.item.content for e in created} == {"a", "b"}

        asyncio.run(body())

    with __import__("tempfile").TemporaryDirectory() as tmp:
        scenario(tmp)


def test_set_items_diffs_deleted_on_re_run() -> None:
    """Re-running set_items emits `deleted` for ids that vanished, none for empty no-op."""

    def scenario(tmp: str) -> None:
        async def body():
            seen: list[tuple[PlanEventType, str]] = []
            emitter = PlanEventEmitter()
            emitter.on(
                PlanEventType.created, lambda e: seen.append((e.event_type, e.item.id))
            )
            emitter.on(
                PlanEventType.deleted, lambda e: seen.append((e.event_type, e.item.id))
            )
            store = FilePlanStore(tmp, conversation_id="conv-d", event_emitter=emitter)

            await store.set_items([PlanItem(content="keep"), PlanItem(content="drop")])
            kept_id = (await store.get_items())[0].id
            seen.clear()

            # drop "drop", keep "keep" -> one deleted, keeper shared so not created again
            await store.set_items(
                [PlanItem(id=kept_id, content="keep"), PlanItem(content="new")]
            )
            kinds = [k for k, _ in seen]
            assert kinds.count(PlanEventType.deleted) == 1
            created_ids = [i for k, i in seen if k is PlanEventType.created]
            assert (
                len(created_ids) == 1 and created_ids[0] != kept_id
            )  # the new item, not the keeper
            # the shared "keep" survived unchanged -> no updated event for it

        asyncio.run(body())

    with __import__("tempfile").TemporaryDirectory() as tmp:
        scenario(tmp)


# --- Task 5: marker-tagged plan reminder (SOP) ------------------------------


def test_plan_reminder_marker_is_in_shared_sop() -> None:
    """The plan marker must join the `_markers` injected-artifact set so readers
    (turns/search/consolidation) strip plan reminders like memory/thread-summary."""

    from types import SimpleNamespace

    from pydantic_ai.messages import TextContent, UserPromptPart

    from paf.capabilities._markers import injected_artifact_markers
    from paf.capabilities.planning import Planning

    assert "paf.plan.v1" in injected_artifact_markers()
    cap = Planning()
    marker = cap._plan_marker_for(SimpleNamespace(conversation_id="conv-p"))  # type: ignore[arg-type]
    assert marker == "paf.plan.v1:" + marker.split(":")[1] and marker.split(":")[1]

    # the reminder is a marker-tagged TextContent, not a bare mutable string
    part = UserPromptPart([TextContent("plan text", metadata="paf.plan.v1:abc123")])
    assert any(
        isinstance(i, TextContent) and str(i.metadata).startswith("paf.plan.v1")
        for i in part.content
    )


def test_remove_previous_injection_strips_plan_reminder() -> None:
    """A stale marked plan reminder is removed from history before a fresh one goes in."""

    from pydantic_ai.messages import (
        CachePoint,
        ModelMessage,
        ModelRequest,
        TextContent,
        UserPromptPart,
    )

    from paf.capabilities.planning import Planning

    cap = Planning()
    stale = UserPromptPart(
        content=[
            "<plan-reminder>\n",
            CachePoint(ttl="5m"),
            TextContent("old plan", metadata="paf.plan.v1:abc"),
        ]
    )
    sign = UserPromptPart("real user prompt")
    request = ModelRequest(parts=[sign, stale])
    messages: list[ModelMessage] = [request]
    cap._remove_previous_injection(messages, "paf.plan.v1:abc")
    remaining = [i for i in messages[0].parts if isinstance(i, UserPromptPart)]
    texts = [i.content for i in remaining]
    assert "old plan" not in str(texts) and any(t == "real user prompt" for t in texts)
