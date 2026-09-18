"""Tests for Phase 3: the in-context `<thread-summary>` present surface.

Mirrors ``tests/test_memory_view.py``'s injection tests: ``before_model_request``
injects once per **real user turn** (never mid tool-loop), strips any prior
``<thread-summary>`` block (matched by the scope-qualified TextContent metadata
marker), and re-injects a bounded render of the per-conversation summaries.
Plus direct render-format tests (oldest->newest, recency header, ``(most
recent)`` tag, window/char cap, empty store).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from paf.capabilities.thread_summarizer._capability import (
    ThreadSummarizer,
    render_thread_summary,
)
from paf.capabilities.thread_summarizer._store import ThreadSummaryRecord

# --------------------------------------------------------------------------- #
# Render format
# --------------------------------------------------------------------------- #


def _rec(
    cid: str,
    ts: str,
    start: str,
    end: str,
    refs: list[str],
    summary: str,
) -> ThreadSummaryRecord:
    return ThreadSummaryRecord(
        conversation_id=cid,
        run_id=refs[0].split(":")[0],
        turn_id=int(refs[0].split(":")[1]),
        summarized_at=ts,
        started_at=start,
        ended_at=end,
        refs=refs,
        summary=summary,
    )


def test_render_oldest_to_newest_most_recent_tag() -> None:
    """Blocks render oldest->newest; the newest carries the (most recent) tag."""
    old = _rec(
        "c1",
        "2026-09-08T09:10:00+00:00",
        "2026-09-07T14:30:00+00:00",
        "2026-09-08T09:10:00+00:00",
        [f"a:{i}" for i in range(10)],
        "block alpha",
    )
    new = _rec(
        "c1",
        "2026-09-09T13:05:00+00:00",
        "2026-09-09T11:20:00+00:00",
        "2026-09-09T13:05:00+00:00",
        [f"b:{i}" for i in range(6)],
        "block beta",
    )

    out = render_thread_summary([old, new])

    assert out.endswith("</thread-summary>")
    assert "<thread-summary>" in out
    assert out.index("block alpha") < out.index("block beta")  # oldest first
    assert "block beta" in out and " (most recent)\n" in out  # newest tagged
    # Header carries relative + absolute + turn count.
    assert "10 turns" in out
    assert "6 turns" in out
    assert "(most recent)" in out
    assert "→" in out  # absolute range
    # Footer depth reminder lands once.
    assert out.count("search_conversation()") == 1
    assert "read_turn()" in out


def test_render_single_block_is_most_recent() -> None:
    rec = _rec(
        "c1",
        "2026-09-09T13:05:00+00:00",
        "2026-09-09T11:20:00+00:00",
        "2026-09-09T13:05:00+00:00",
        [f"a:{i}" for i in range(3)],
        "solo",
    )
    out = render_thread_summary([rec])
    assert "(most recent)" in out
    assert "3 turns" in out


def test_render_legacy_no_timestamps_still_renders() -> None:
    """Pre-timestamp records fall back to summarized_at so the header isn't blank."""
    rec = ThreadSummaryRecord(
        conversation_id="c1",
        run_id="a",
        turn_id=4,
        summarized_at="2026-09-08T12:00:00+00:00",
        started_at="",
        ended_at="",
        refs=["a:1", "a:2", "a:3", "a:4"],
        summary="legacy beat",
    )
    out = render_thread_summary([rec])
    assert "legacy beat" in out
    assert "4 turns" in out


def test_render_bound_respects_max_chars_trims_oldest() -> None:
    """Over long blocks, oldest are trimmed first to fit max_chars."""
    blocks = [
        _rec(
            "c1",
            f"2026-09-0{i}T10:00:00+00:00",
            f"2026-09-0{i}T09:00:00+00:00",
            f"2026-09-0{i}T10:00:00+00:00",
            [f"r{i}:0"],
            f"summary number {i}",
        )
        for i in range(6, 0, -1)  # 6 blocks
    ]
    out = render_thread_summary(blocks, window=10, max_chars=120)
    assert "summary number 1" in out  # newest retained
    assert "summary number 6" not in out  # oldest trimmed under the hard cap
    assert len(out) <= 160


def test_render_empty_store_returns_empty() -> None:
    assert render_thread_summary([]) == ""


# --------------------------------------------------------------------------- #
# Injection (mirrors memory: once per real user turn)
# --------------------------------------------------------------------------- #


def _ctx(messages):
    from pydantic_ai.models import ModelRequestContext

    return ModelRequestContext(
        model="openrouter:deepseek/deepseek-v4-flash-0731",
        messages=messages,
        model_settings={},
        model_request_parameters={},
    )


def _run_ctx(conversation_id: str | None = None):
    from opentelemetry.trace import NoOpTracer
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.tools import RunContext
    from pydantic_ai.usage import RunUsage

    def _call(_messages, info):
        raise AssertionError("unused")

    return RunContext(
        deps=None,
        model=FunctionModel(_call),
        usage=RunUsage(),
        tracer=NoOpTracer(),
        conversation_id=conversation_id,
    )


def _seed_store(capture_dir: Path, cid: str) -> None:
    from paf.capabilities.thread_summarizer._store import ThreadSummaryStore

    store = ThreadSummaryStore(capture_dir, cid)
    store.append_summary(
        _rec(
            cid,
            "2026-09-09T13:05:00+00:00",
            "2026-09-09T11:20:00+00:00",
            "2026-09-09T13:05:00+00:00",
            [f"a:{i}" for i in range(6)],
            "thread beat alpha",
        )
    )


def test_inject_on_real_user_turn_only(tmp_path: Path) -> None:
    """Injects once on a real user turn; leaves tool-loop continuations untouched."""
    from pydantic_ai.messages import (
        ModelRequest,
        TextContent,
        ToolReturnPart,
        UserPromptPart,
    )

    cid = "conv-demo-1"
    capture_dir = tmp_path / "summaries"
    _seed_store(capture_dir, cid)
    cap = ThreadSummarizer(
        capture_directory=capture_dir,
        agent_name="assistant",
        inject_thread_summary=True,
    )

    async def _run() -> None:
        # 1. Real user turn -> inject a <thread-summary> part.
        user = ModelRequest(parts=[UserPromptPart(content="talk about the plan")])
        out = await cap.before_model_request(_run_ctx(cid), _ctx([user]))
        parts = out.messages[0].parts
        assert len(parts) == 2  # original + injected
        injected = parts[1]
        assert isinstance(injected, UserPromptPart)
        assert "<thread-summary>" in str(injected.content)
        assert "thread beat alpha" in str(injected.content)
        assert "(most recent)" in str(injected.content)

        # Marker is scope-qualified by conversation id.
        marker = next(
            (tc.metadata for tc in injected.content if isinstance(tc, TextContent)),
            None,
        )
        assert marker and marker.startswith("paf.thread-summary.v1:")
        assert marker.endswith(hashlib_hex(cid))

        # 2. Tool-loop continuation (tool return, no user prompt) -> SKIP.
        tool_msg = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="search_conversation",
                    tool_call_id="call_1",
                    content=[TextContent("ok")],
                )
            ]
        )
        before = _ctx([user, tool_msg])
        out2 = await cap.before_model_request(_run_ctx(cid), before)
        assert out2.messages[-1] is before.messages[-1]  # untouched
        assert all(
            p.tool_name == "search_conversation" for p in out2.messages[-1].parts
        )

    asyncio.run(_run())


def test_inject_strips_prior_block_no_accumulation(tmp_path: Path) -> None:
    """A second real user turn strips the old block and injects a fresh one."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    cid = "conv-demo-1"
    capture_dir = tmp_path / "summaries"
    _seed_store(capture_dir, cid)
    cap = ThreadSummarizer(
        capture_directory=capture_dir,
        agent_name="assistant",
        inject_thread_summary=True,
    )

    async def _run() -> None:
        first = ModelRequest(parts=[UserPromptPart(content="first")])
        out1 = await cap.before_model_request(_run_ctx(cid), _ctx([first]))
        n1 = str(out1.messages[0].parts[1].content)
        assert n1.count("<thread-summary>") == 1

        second = ModelRequest(parts=[UserPromptPart(content="second")])
        out2 = await cap.before_model_request(_run_ctx(cid), _ctx([first, second]))
        # Only one <thread-summary> across the whole message list -> no accumulation.
        joined = "".join(
            str(p.content) for p in out2.messages[0].parts for _ in [0]
        ) + "".join(str(p.content) for p in out2.messages[1].parts)
        assert joined.count("<thread-summary>") == 1

    asyncio.run(_run())


def test_no_inject_when_off_or_empty_store(tmp_path: Path) -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    cap_off = ThreadSummarizer(
        capture_directory=tmp_path / "s",
        agent_name="assistant",
        inject_thread_summary=False,
    )
    cap_empty = ThreadSummarizer(
        capture_directory=tmp_path / "s",
        agent_name="assistant",
        inject_thread_summary=True,
    )

    async def _run() -> None:
        user = ModelRequest(parts=[UserPromptPart(content="hi")])
        out_off = await cap_off.before_model_request(_run_ctx(), _ctx([user]))
        assert len(out_off.messages[0].parts) == 1  # nothing injected
        out_empty = await cap_empty.before_model_request(_run_ctx(), _ctx([user]))
        assert len(out_empty.messages[0].parts) == 1  # empty store -> nothing

    asyncio.run(_run())


def hashlib_hex(cid: str) -> str:
    return __import__("hashlib").sha256(cid.encode()).hexdigest()[:16]
