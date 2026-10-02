"""Tests for the paf-owned ``Subagents`` capability (sync delegation + persistence).

Parents and leaves run on ``FunctionModel`` so no network is touched; task
stores are rooted at ``tmp_path`` (never the real ``data/``). Follows the repo
convention of sync ``def test_*`` functions driving async code (no
pytest-asyncio).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import UsageLimits

from paf.background_processes._events import clear_agent_emitters, get_agent_emitter
from paf.capabilities.subagents import (
    Leaf,
    LeafPolicy,
    Subagents,
    SubagentTaskStore,
)
from paf.capabilities.subagents import _store as store_mod
from paf.capabilities.subagents._notify import NOTIFY_MAX_CHARS, excerpt, notify_text
from paf.capabilities.subagents._roster import TLDR_INSTRUCTION

CID = "demo:c1"
AGENT = "test_agent"


# -- helpers -------------------------------------------------------------------


def _text_leaf(
    text: str, name: str = "writer", policy: LeafPolicy | None = None
) -> Leaf:
    """A leaf whose model always answers ``text``."""

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(text)])

    return Leaf(
        name=name,
        description=f"{name} leaf",
        agent=Agent(FunctionModel(fn), name=name),
        policy=policy or LeafPolicy(),
    )


def _leaf_from(fn: Any, name: str = "writer", policy: LeafPolicy | None = None) -> Leaf:
    return Leaf(
        name=name,
        description="",
        agent=Agent(FunctionModel(fn), name=name),
        policy=policy or LeafPolicy(),
    )


def _parent(sub: Subagents, calls: list[tuple[str, dict[str, Any]]]) -> Agent[Any, str]:
    """A parent that makes ``calls`` one per step, then answers with the last result."""

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        made = sum(
            1
            for m in messages
            for p in getattr(m, "parts", [])
            if isinstance(p, ToolCallPart)
        )
        if made < len(calls):
            tool, args = calls[made]
            return ModelResponse(parts=[ToolCallPart(tool, args)])
        returns = [
            p
            for m in messages
            for p in getattr(m, "parts", [])
            if isinstance(p, ToolReturnPart | RetryPromptPart)
        ]
        return ModelResponse(parts=[TextPart(str(returns[-1].content))])

    return Agent(FunctionModel(fn), capabilities=[sub])


def _sub(tmp_path: Path, *leaves: Leaf) -> Subagents:
    return Subagents(leaves=list(leaves), processes_root=tmp_path, agent=AGENT)


def _store(tmp_path: Path, cid: str | None = CID) -> SubagentTaskStore:
    return SubagentTaskStore(tmp_path, AGENT, conversation_id=cid)


def _run(agent: Agent[Any, str], cid: str | None = CID) -> str:
    return agent.run_sync("go", conversation_id=cid).output


@pytest.fixture(autouse=True)
def _isolate_emitters():
    clear_agent_emitters()
    yield
    clear_agent_emitters()


# -- excerpt / notify text ---------------------------------------------------------


def test_excerpt_returns_short_output_whole() -> None:
    text = "x" * NOTIFY_MAX_CHARS
    assert excerpt(text, "t1") == text  # exactly at the limit: whole, no pointer


def test_excerpt_cuts_at_paragraph_then_line_then_hard() -> None:
    para = "A" * 1000 + "\n\n" + "B" * 1000
    out = excerpt(para, "t1")
    assert out.startswith("A" * 1000 + "\n\n… [truncated — ")
    assert "B" not in out.split("…")[0]
    assert "1002 more chars; check_task('t1')" in out  # "\n\n" + 1000 B's

    line = "A" * 1000 + "\n" + "B" * 1000
    assert excerpt(line, "t1").startswith("A" * 1000 + "\n\n… [truncated")

    # No boundary in the back half of the limit -> hard cut at the limit.
    hard = "A" * 100 + "\n" + "B" * 3000
    out = excerpt(hard, "t1")
    assert out.split("\n\n…")[0] == hard[:NOTIFY_MAX_CHARS]
    assert f"{len(hard) - NOTIFY_MAX_CHARS} more chars" in out


def test_notify_text_ok_and_error() -> None:
    assert notify_text("writer", "t1", True, "done!") == (
        "[system] Subagent writer task t1 finished (ok):\n\ndone!"
    )
    assert notify_text("writer", "t1", False, "boom") == (
        "[system] Subagent writer task t1 finished (error): boom"
    )


# -- sync delegation ----------------------------------------------------------------


def test_delegate_returns_output_and_persists_record_and_md(tmp_path: Path) -> None:
    sub = _sub(tmp_path, _text_leaf("TL;DR: done.\n\nFull body."))
    out = _run(
        _parent(sub, [("delegate_task", {"agent_name": "writer", "task": "write it"})])
    )
    assert out == "TL;DR: done.\n\nFull body."

    store = _store(tmp_path)
    rec = store.get(f"subagent.{CID}.1")
    assert rec is not None
    assert rec["status"] == "done"
    assert rec["mode"] == "sync"
    assert rec["agent"] == "writer"
    assert rec["task"] == "write it"
    assert rec["parent_run_id"]  # captured from the parent's ctx.run_id
    assert rec["conversation_id"] == CID
    assert rec["output_chars"] == len(out)
    assert rec["usage"]  # the child's own run usage

    md = Path(rec["output_path"])
    assert md.exists()
    assert md.parent.name == "subagent_outputs"
    raw = md.read_text(encoding="utf-8")
    front = yaml.safe_load(raw.split("---\n")[1])
    assert front["task_id"] == f"subagent.{CID}.1"
    assert front["agent"] == "writer"
    assert front["task"] == "write it"
    assert "status" not in front  # status lives only in the JSONL
    assert store.read_output(rec["task_id"]) == out


def test_task_ids_increment_per_conversation(tmp_path: Path) -> None:
    sub = _sub(tmp_path, _text_leaf("ok"))
    call = ("delegate_task", {"agent_name": "writer", "task": "t"})
    _run(_parent(sub, [call, call]))
    _run(_parent(sub, [call]), cid="demo:c2")
    assert set(_store(tmp_path).latest_all()) == {
        f"subagent.{CID}.1",
        f"subagent.{CID}.2",
    }
    assert set(_store(tmp_path, "demo:c2").latest_all()) == {"subagent.demo:c2.1"}


def test_colon_cid_gives_safe_filename_but_real_task_id(tmp_path: Path) -> None:
    sub = _sub(tmp_path, _text_leaf("ok"))
    _run(_parent(sub, [("delegate_task", {"agent_name": "writer", "task": "t"})]))
    rec = _store(tmp_path).get(f"subagent.{CID}.1")
    assert rec is not None
    name = Path(rec["output_path"]).name
    assert ":" not in name
    assert name == "subagent.demo_c1.1.md"


def test_leaf_exception_is_soft_failure_error_record_no_file(tmp_path: Path) -> None:
    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model exploded")

    sub = _sub(tmp_path, _leaf_from(boom))
    out = _run(_parent(sub, [("delegate_task", {"agent_name": "writer", "task": "t"})]))
    assert out.startswith("Subagent 'writer' failed: RuntimeError: model exploded")
    assert "recoverable" in out
    rec = _store(tmp_path).get(f"subagent.{CID}.1")
    assert rec is not None
    assert rec["status"] == "error"
    assert rec["output_path"] is None
    assert list(tmp_path.rglob("*.md")) == []


def test_timeout_and_usage_limit_are_errors(tmp_path: Path) -> None:
    async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(5)
        return ModelResponse(parts=[TextPart("late")])

    slow_leaf = _leaf_from(slow, name="slow", policy=LeafPolicy(timeout_seconds=0.05))
    capped = _text_leaf(
        "never",
        name="capped",
        policy=LeafPolicy(usage_limits=UsageLimits(request_limit=0)),
    )
    sub = _sub(tmp_path, slow_leaf, capped)
    out = _run(
        _parent(
            sub,
            [
                ("delegate_task", {"agent_name": "slow", "task": "t"}),
                ("delegate_task", {"agent_name": "capped", "task": "t"}),
            ],
        )
    )
    store = _store(tmp_path)
    assert "timed out after 0.05s" in store.get(f"subagent.{CID}.1")["detail"]  # type: ignore[index]
    assert store.get(f"subagent.{CID}.2")["status"] == "error"  # type: ignore[index]
    assert "UsageLimitExceeded" in out


def test_unknown_agent_is_model_retry(tmp_path: Path) -> None:
    toolset = _sub(tmp_path, _text_leaf("ok")).get_toolset()
    assert toolset is not None
    with pytest.raises(ModelRetry, match="Unknown subagent 'nope'. Available: writer"):
        toolset._leaf("nope")


def test_max_calls_per_run_is_enforced(tmp_path: Path) -> None:
    leaf = _text_leaf("ok", policy=LeafPolicy(max_calls_per_run=1))
    call = ("delegate_task", {"agent_name": "writer", "task": "t"})
    out = _run(_parent(_sub(tmp_path, leaf), [call, call]))
    assert "reached its call budget (1 per run)" in out
    assert len(_store(tmp_path).latest_all()) == 1  # the over-budget call ran nothing


def test_check_task_pages_the_output(tmp_path: Path) -> None:
    body = "".join(f"{i:05d}\n" for i in range(2000))  # 12000 chars
    sub = _sub(tmp_path, _text_leaf(body))
    tid = f"subagent.{CID}.1"
    out = _run(
        _parent(
            sub,
            [
                ("delegate_task", {"agent_name": "writer", "task": "t"}),
                ("check_task", {"task_id": tid, "offset": 6000, "max_chars": 60}),
            ],
        )
    )
    assert out.startswith(
        f"Task {tid} (writer): done — output chars 6000-6060 of 12000:"
    )
    assert "01000\n" in out
    assert f"call check_task('{tid}', offset=6060)" in out
    assert "---" not in out  # frontmatter stripped


def test_check_task_unknown_id(tmp_path: Path) -> None:
    sub = _sub(tmp_path, _text_leaf("ok"))
    out = _run(_parent(sub, [("check_task", {"task_id": "subagent.x.9"})]))
    assert out == "[Error: unknown task id 'subagent.x.9' in this conversation]"


def test_sync_delegation_emits_running_then_done(tmp_path: Path) -> None:
    events: list[tuple[str, str, str | None]] = []
    get_agent_emitter(AGENT).on_any(
        lambda e: events.append((e.id, e.status.value, e.scope))
    )
    sub = _sub(tmp_path, _text_leaf("ok"))
    _run(_parent(sub, [("delegate_task", {"agent_name": "writer", "task": "t"})]))
    tid = f"subagent.{CID}.1"
    assert events == [(tid, "running", CID), (tid, "done", CID)]


def test_reconcile_marks_previous_process_tasks_interrupted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rec = store.create(
        agent_name="writer", task="t", mode="background", parent_run_id="r"
    )
    store.update(rec["task_id"], status="running")
    # A fresh process: the per-process "already reconciled" memo is empty.
    store_mod._reconciled.clear()
    reopened = _store(tmp_path)
    final = reopened.get(rec["task_id"])
    assert final is not None
    assert final["status"] == "interrupted"
    assert final["detail"] == "server restarted"
    # Same process: opening again does not touch a task started after.
    live = reopened.create(
        agent_name="writer", task="t", mode="background", parent_run_id="r"
    )
    reopened.update(live["task_id"], status="running")
    assert _store(tmp_path).get(live["task_id"])["status"] == "running"  # type: ignore[index]


def test_duplicate_leaf_names_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate subagent names"):
        _sub(tmp_path, _text_leaf("a"), _text_leaf("b"))


def test_no_leaves_means_no_tools_or_instructions(tmp_path: Path) -> None:
    sub = _sub(tmp_path)
    assert sub.get_toolset() is None
    assert sub.get_instructions() is None


# -- real writer spec ------------------------------------------------------------


def test_writer_leaf_from_spec(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The real writer spec: runtime name, policy, TL;DR once, no parent-only tools."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")  # construction only, no calls
    from paf.capabilities.persistence import _build_store
    from paf.capabilities.subagents import load_leaf
    from paf.specs import load_agent_spec

    leaf = load_leaf("writer_subagent", store=_build_store(), filesystem_config=None)
    assert leaf.name == "writer"
    assert leaf.policy.timeout_seconds == 300
    assert leaf.policy.usage_limits.request_limit == 30
    assert leaf.policy.background == "allowed"

    seen: dict[str, Any] = {}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["instructions"] = info.instructions or ""
        seen["tools"] = {t.name for t in info.function_tools}
        seen["system_parts"] = [
            p
            for m in messages
            for p in getattr(m, "parts", [])
            if p.part_kind == "system-prompt"
        ]
        return ModelResponse(parts=[TextPart("ok")])

    leaf.agent.run_sync("hi", model=FunctionModel(fn))
    spec_text = str(load_agent_spec("writer_subagent").instructions).strip()
    assert seen["instructions"].count(spec_text[:80]) == 1  # prompt sent once
    assert seen["instructions"].rstrip().endswith(TLDR_INSTRUCTION)
    assert seen["system_parts"] == []
    assert "read_file" in seen["tools"]  # the shared FileSystem
    assert not seen["tools"] & {
        "delegate_task",
        "check_task",
        "search_conversation_history",
    }


def test_usage_dict_is_json_safe_with_decimal_cost() -> None:
    """Regression: OpenRouter reports cost as Decimal, which json.dumps rejects."""
    import json
    from dataclasses import dataclass, field
    from decimal import Decimal

    from paf.capabilities.subagents._runner import _usage_dict

    @dataclass
    class FakeUsage:
        requests: int = 2
        input_tokens: int = 10
        cache_read_tokens: int = 0
        details: dict[str, Any] = field(
            default_factory=lambda: {"cost": Decimal("0.0012"), "reasoning": 3}
        )

    out = _usage_dict(FakeUsage())
    assert out == {
        "requests": 2,
        "input_tokens": 10,
        "details": {"cost": 0.0012, "reasoning": 3},
    }
    json.dumps(out)  # must not raise


# -- background delegation ---------------------------------------------------------


def _gated_leaf(
    gate: asyncio.Event,
    log: dict[str, int],
    text: str = "bg result",
    name: str = "writer",
    policy: LeafPolicy | None = None,
) -> Leaf:
    """A leaf that blocks on ``gate`` and tracks how many runs are in flight."""

    async def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        log["inflight"] = log.get("inflight", 0) + 1
        log["peak"] = max(log.get("peak", 0), log["inflight"])
        try:
            await gate.wait()
        finally:
            log["inflight"] -= 1
        return ModelResponse(parts=[TextPart(text)])

    return _leaf_from(fn, name=name, policy=policy)


async def _settle() -> None:
    """Await every live background subagent task."""
    from paf.capabilities.subagents._background import background_tasks

    pending = background_tasks.pending()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def _until(pred: Any, timeout: float = 2.0) -> None:
    """Yield to the loop until ``pred()`` is true (bounded)."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not pred():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


def _bg_call(task: str = "t", agent_name: str = "writer") -> tuple[str, dict[str, Any]]:
    return (
        "delegate_task",
        {"agent_name": agent_name, "task": task, "background": True},
    )


def test_background_returns_immediately_then_completes(tmp_path: Path) -> None:
    async def scenario() -> None:
        gate, log = asyncio.Event(), {}
        sub = _sub(tmp_path, _gated_leaf(gate, log))
        out = (await _parent(sub, [_bg_call()]).run("go", conversation_id=CID)).output
        tid = f"subagent.{CID}.1"
        assert out == (
            f"Started subagent task {tid} (writer) in the background. "
            f"Poll check_task('{tid}') for its result."
        )  # no on_complete hook -> told to poll
        await _until(lambda: log.get("inflight") == 1)
        assert _store(tmp_path).get(tid)["status"] == "running"  # type: ignore[index]
        gate.set()
        await _settle()
        rec = _store(tmp_path).get(tid)
        assert rec is not None
        assert rec["status"] == "done"
        assert rec["mode"] == "background"
        assert _store(tmp_path).read_output(tid) == "bg result"

    asyncio.run(scenario())


def test_background_tasks_run_in_parallel_up_to_the_cap(tmp_path: Path) -> None:
    async def scenario() -> None:
        gate, log = asyncio.Event(), {}
        sub = Subagents(
            leaves=[_gated_leaf(gate, log)],
            processes_root=tmp_path,
            agent=AGENT,
            max_concurrent=2,
        )
        await _parent(sub, [_bg_call(), _bg_call(), _bg_call()]).run(
            "go", conversation_id=CID
        )
        await _until(lambda: log.get("inflight") == 2)
        await asyncio.sleep(0.05)  # give a 3rd task every chance to start
        assert log["inflight"] == 2  # capped
        statuses = {t: r["status"] for t, r in _store(tmp_path).latest_all().items()}
        assert sorted(statuses.values()) == ["queued", "running", "running"]
        gate.set()
        await _settle()
        assert log["peak"] == 2
        assert {r["status"] for r in _store(tmp_path).latest_all().values()} == {"done"}

    asyncio.run(scenario())


def test_cancel_task_running_and_queued(tmp_path: Path) -> None:
    async def scenario() -> None:
        gate, log = asyncio.Event(), {}
        sub = Subagents(
            leaves=[_gated_leaf(gate, log)],
            processes_root=tmp_path,
            agent=AGENT,
            max_concurrent=1,
        )
        await _parent(sub, [_bg_call(), _bg_call()]).run("go", conversation_id=CID)
        await _until(lambda: log.get("inflight") == 1)
        t1, t2 = f"subagent.{CID}.1", f"subagent.{CID}.2"
        out = (
            await _parent(
                sub,
                [
                    ("cancel_task", {"task_id": t2}),  # still queued
                    ("cancel_task", {"task_id": t1}),  # running
                ],
            ).run("go", conversation_id=CID)
        ).output
        assert out == f"Cancelling task {t1} (writer)."
        await _settle()
        store = _store(tmp_path)
        assert store.get(t1)["status"] == "cancelled"  # type: ignore[index]
        assert store.get(t2)["status"] == "cancelled"  # type: ignore[index]
        assert list(tmp_path.rglob("*.md")) == []
        again = (
            await _parent(sub, [("cancel_task", {"task_id": t1})]).run(
                "go", conversation_id=CID
            )
        ).output
        assert again == f"Task {t1} is not running (status: cancelled)."

    asyncio.run(scenario())


def test_background_policy_rejections(tmp_path: Path) -> None:
    never = _text_leaf("x", name="never", policy=LeafPolicy(background="never"))
    required = _text_leaf(
        "x", name="required", policy=LeafPolicy(background="required")
    )
    sub = _sub(tmp_path, never, required)
    out = _run(_parent(sub, [_bg_call(agent_name="never")]))
    assert (
        "cannot run in the background; call delegate_task with background=false" in out
    )
    out = _run(
        _parent(sub, [("delegate_task", {"agent_name": "required", "task": "t"})])
    )
    assert "only runs in the background; call delegate_task with background=true" in out
    assert _store(tmp_path).latest_all() == {}


def test_background_emits_queued_running_done(tmp_path: Path) -> None:
    async def scenario() -> None:
        events: list[str] = []
        get_agent_emitter(AGENT).on_any(lambda e: events.append(e.status.value))
        sub = _sub(tmp_path, _text_leaf("ok"))
        await _parent(sub, [_bg_call()]).run("go", conversation_id=CID)
        await _settle()
        assert events == ["queued", "running", "done"]

    asyncio.run(scenario())


def test_restore_frames_active_always_terminal_within_window(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    from paf.capabilities.subagents._store import conversation_subagent_frames

    store = _store(tmp_path)
    running = store.create(
        agent_name="writer", task="long job", mode="background", parent_run_id="r"
    )
    store.update(running["task_id"], status="running")
    old = store.create(agent_name="writer", task="t", mode="sync", parent_run_id="r")
    store.update(old["task_id"], status="error", detail="boom")
    fresh = store.create(agent_name="writer", task="t", mode="sync", parent_run_id="r")
    store.mark_done(fresh["task_id"], output="abc", usage=None)

    now = datetime.now(UTC)
    frames = {
        f["id"]: f for f in conversation_subagent_frames(tmp_path, AGENT, CID, now=now)
    }
    assert frames[running["task_id"]] == {
        "id": running["task_id"],
        "label": "subagent: writer",
        "status": "running",
        "progress": None,
        "detail": "long job",
    }
    assert frames[old["task_id"]]["detail"] == "error: boom"
    assert frames[fresh["task_id"]]["status"] == "done"
    assert frames[fresh["task_id"]]["detail"] == "3 chars"

    later = now + timedelta(minutes=16)
    later_ids = {
        f["id"] for f in conversation_subagent_frames(tmp_path, AGENT, CID, now=later)
    }
    assert later_ids == {running["task_id"]}  # terminal rows aged out


# -- completion notify -------------------------------------------------------------


def _collecting_hook(sink: list[tuple[str, str | None, bool, str]]) -> Any:
    async def hook(task_id: str, cid: str | None, ok: bool, text: str) -> None:
        sink.append((task_id, cid, ok, text))

    return hook


def test_on_complete_short_long_and_error(tmp_path: Path) -> None:
    async def scenario() -> None:
        calls: list[tuple[str, str | None, bool, str]] = []

        def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError("nope")

        long_text = "TL;DR: long.\n\n" + ("word " * 600)
        sub = Subagents(
            leaves=[
                _text_leaf("short answer", name="short"),
                _text_leaf(long_text, name="long"),
                _leaf_from(boom, name="bad"),
            ],
            processes_root=tmp_path,
            agent=AGENT,
            on_complete=_collecting_hook(calls),
        )
        out = (
            await _parent(
                sub,
                [
                    _bg_call(agent_name="short"),
                    _bg_call(agent_name="long"),
                    _bg_call(agent_name="bad"),
                ],
            ).run("go", conversation_id=CID)
        ).output
        assert "You will get a [system] message" in out  # hook wired -> no polling
        await _settle()
        by_id = {c[0]: c for c in calls}
        assert len(calls) == 3  # exactly once each

        t1, t2, t3 = (f"subagent.{CID}.{n}" for n in (1, 2, 3))
        assert by_id[t1] == (
            t1,
            CID,
            True,
            f"[system] Subagent short task {t1} finished (ok):\n\nshort answer",
        )
        _, _, ok, text = by_id[t2]
        assert ok
        assert text.startswith(
            f"[system] Subagent long task {t2} finished (ok):\n\nTL;DR: long."
        )
        assert f"check_task('{t2}') for the full output]" in text
        assert len(text) < 1700
        _, _, ok, text = by_id[t3]
        assert not ok
        assert text == (
            f"[system] Subagent bad task {t3} finished (error): RuntimeError: nope"
        )

    asyncio.run(scenario())


def test_cancelled_task_does_not_notify(tmp_path: Path) -> None:
    async def scenario() -> None:
        calls: list[tuple[str, str | None, bool, str]] = []
        gate, log = asyncio.Event(), {}
        sub = Subagents(
            leaves=[_gated_leaf(gate, log)],
            processes_root=tmp_path,
            agent=AGENT,
            on_complete=_collecting_hook(calls),
        )
        await _parent(sub, [_bg_call()]).run("go", conversation_id=CID)
        await _until(lambda: log.get("inflight") == 1)
        await _parent(sub, [("cancel_task", {"task_id": f"subagent.{CID}.1"})]).run(
            "go", conversation_id=CID
        )
        await _settle()
        assert calls == []

    asyncio.run(scenario())


def test_notify_turn_is_not_linked_to_the_spawning_run(tmp_path: Path) -> None:
    """End-to-end for the spawn-context fix: a notify turn started from a
    background completion must not record the spawning turn as its parent."""
    from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence

    step_store = FileStepStore(directory=tmp_path / "steps")
    notify_runs: list[str] = []

    async def scenario() -> None:
        async def hook(task_id: str, cid: str | None, ok: bool, text: str) -> None:
            # What Session.notify does on an idle lane: start a new turn.
            turn = Agent(
                FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("ack")])),
                capabilities=[StepPersistence(store=step_store)],
            )
            result = await turn.run(text, conversation_id=cid)
            notify_runs.append(result.run_id)

        sub = Subagents(
            leaves=[_text_leaf("done")],
            processes_root=tmp_path / "proc",
            agent=AGENT,
            on_complete=hook,
        )
        parent = _parent(sub, [_bg_call()])
        parent_with_steps = Agent(
            parent.model,
            capabilities=[sub, StepPersistence(store=step_store)],
        )
        await parent_with_steps.run("go", conversation_id=CID)
        await _settle()

    asyncio.run(scenario())
    assert len(notify_runs) == 1
    runs = {r.run_id: r for r in asyncio.run(step_store.list_runs())}
    assert runs[notify_runs[0]].parent_run_id is None


def test_sse_notify_message_verbatim_for_subagents_and_shell() -> None:
    from ui import sse

    text = "[system] Subagent writer task subagent.demo:c1.1 finished (ok):\n\nhi"
    assert sse._notify_message("subagent.demo:c1.1", text) == text
    shell = "[system] Background command shell.demo:c1.3 (exit 0): ls\n\nok"
    assert sse._notify_message("shell.demo:c1.3", shell) == shell
    assert sse._notify_message("other.1", "exit 0") == (
        "[system] Background process other.1 finished: exit 0"
    )


# -- spec-driven roster + tooled leaves ---------------------------------------------


def _tool_names(leaf: Leaf) -> set[str]:
    seen: dict[str, set[str]] = {}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen["tools"] = {t.name for t in info.function_tools}
        return ModelResponse(parts=[TextPart("ok")])

    leaf.agent.run_sync("hi", model=FunctionModel(fn))
    return seen["tools"]


def test_spec_subagents_config_parsing(tmp_path: Path) -> None:
    from paf.specs import _spec_runtime_configs, _spec_subagents_config

    good = tmp_path / "good.yaml"
    good.write_text(
        "subagents:\n  agents: [writer_subagent, researcher_subagent]\n  max_concurrent: 2\n"
    )
    assert _spec_subagents_config(good) == {
        "agents": ["writer_subagent", "researcher_subagent"],
        "max_concurrent": 2,
    }
    bad = tmp_path / "bad.yaml"
    bad.write_text("subagents:\n  agents: writer\n  max_concurrent: 0\n")
    assert _spec_subagents_config(bad) is None
    absent = tmp_path / "absent.yaml"
    absent.write_text("model: test\n")
    assert _spec_subagents_config(absent) is None
    assert _spec_runtime_configs(good)["subagents_config"]["max_concurrent"] == 2


def test_roster_and_cap_from_subagents_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    from paf.capabilities import AgentProfile
    from paf.capabilities.build import build_runtime_capabilities

    built = build_runtime_capabilities(
        AgentProfile(name="x", capability_names=("Subagents",)),
        subagents_config={
            "agents": ["writer_subagent", "researcher_subagent"],
            "max_concurrent": 2,
        },
        agent_name="demo_assistant",
    )
    sub = built[0]
    assert isinstance(sub, Subagents)
    assert [leaf.name for leaf in sub.leaves] == ["writer", "researcher"]
    assert sub.max_concurrent == 2
    # Default roster when no block: just the writer.
    default = build_runtime_capabilities(
        AgentProfile(name="x", capability_names=("Subagents",))
    )[0]
    assert [leaf.name for leaf in default.leaves] == ["writer"]
    assert default.max_concurrent == 5


def test_researcher_leaf_gets_its_profile_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    from paf.capabilities.persistence import _build_store
    from paf.capabilities.subagents import load_leaf

    leaf = load_leaf(
        "researcher_subagent", store=_build_store(), filesystem_config=None
    )
    assert leaf.name == "researcher"
    assert leaf.policy.timeout_seconds == 600
    tools = _tool_names(leaf)
    assert {"search_web", "fetch_web_page", "read_file"} <= tools
    assert not tools & {"delegate_task", "check_task", "cancel_task"}


def test_leaf_profile_never_gets_excluded_capabilities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Even a profile that allowlists them never gives a leaf persistence,
    recall, memory, the summarizer, or delegation (depth 1)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from paf.capabilities import AgentProfile
    from paf.capabilities.persistence import _build_store
    from paf.capabilities.profiles import PROFILES
    from paf.capabilities.subagents import load_leaf
    from paf.capabilities.subagents._roster import LEAF_EXCLUDED_CAPABILITIES

    monkeypatch.setitem(
        PROFILES,
        "everything-leaf",
        AgentProfile(
            name="everything-leaf",
            capability_names=(*sorted(LEAF_EXCLUDED_CAPABILITIES), "FileSystem"),
        ),
    )
    (tmp_path / "greedy_subagent.yaml").write_text(
        "name: greedy_subagent\n"
        "model: openrouter:openai/gpt-5.6-luna\n"
        "description: test leaf\n"
        "profile: everything-leaf\n"
        "instructions: Do the task.\n"
    )
    leaf = load_leaf(
        "greedy_subagent",
        store=_build_store(),
        filesystem_config=None,
        spec_directory=tmp_path,
    )
    assert leaf.name == "greedy"
    tools = _tool_names(leaf)
    assert "read_file" in tools
    assert not tools & {
        "delegate_task",
        "check_task",
        "search_conversation_history",
        "read_turn",
        "append_memory",
    }


def test_unknown_leaf_profile_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from paf.capabilities.persistence import _build_store
    from paf.capabilities.subagents import load_leaf

    (tmp_path / "odd_subagent.yaml").write_text(
        "name: odd_subagent\nmodel: openrouter:openai/gpt-5.6-luna\nprofile: nope\n"
    )
    with pytest.raises(KeyError, match="unknown profile 'nope'"):
        load_leaf(
            "odd_subagent",
            store=_build_store(),
            filesystem_config=None,
            spec_directory=tmp_path,
        )
