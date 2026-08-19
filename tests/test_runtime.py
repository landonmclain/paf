"""Tests for ``paf.runtime.Session`` — the UI-agnostic run lifecycle wrapper.

The default (non-live) tests exercise the pure helpers and importability, so
they run offline with no API key or network. The live-marked test runs the real
model end-to-end through the ``Session`` and is excluded from the default
``pytest`` run (``addopts = -m 'not live'``) — invoke explicitly with
``pytest -m live`` using ``.env``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    RunRecord,
)

from paf import runtime


def test_part_leading_text_str() -> None:
    """A str content is returned verbatim (first chunk of a text part)."""
    part = _Part(content="Hello")
    assert runtime._part_leading_text(part) == "Hello"


def test_part_leading_text_list_of_str() -> None:
    """List content (pydantic-ai TextPart / chunked) is joined."""
    part = _Part(content=["a", "b", "c"])
    assert runtime._part_leading_text(part) == "abc"


def test_part_leading_text_non_text() -> None:
    """Non-text parts (empty/None content) yield an empty string."""
    assert runtime._part_leading_text(_Part(content=None)) == ""
    assert runtime._part_leading_text(_Part(content=[])) == ""


def test_part_leading_text_missing_attr() -> None:
    """Parts without a ``content`` attr degrade to ``""`` (never raise)."""
    assert runtime._part_leading_text(_NoContent()) == ""


def test_session_importable_and_news() -> None:
    """Session constructs its doc-visible surface without a model key."""
    # Trace disabled so construction is offline-safe; build_agent lazily needs
    # the model config which we don't touch here.
    assert callable(runtime.Session)
    assert callable(runtime.Session.run_sync)


# -- resume (continue_run auto-seed) -----------------------------------------


async def _seed_run(store: FileStepStore, run_id: str, conv: str, msg: str) -> None:
    """Register a run + a complete snapshot with one user message, so continue_run has history."""
    await store.register_run(
        RunRecord(run_id=run_id, conversation_id=conv, agent_name="assistant")
    )
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id=run_id,
            step_index=0,
            messages=[
                ModelRequest(parts=[UserPromptPart(content=msg)]),
                ModelResponse(parts=[TextPart(content="ack")]),
            ],
            conversation_id=conv,
            parent_run_id=None,
            agent_name="assistant",
            timestamp=datetime.now(UTC),
            state="complete",
        )
    )


def test_session_restore_loads_prior_history(tmp_path: Path) -> None:
    """A fresh Session with an existing conversation resumes from the store."""
    store = FileStepStore(tmp_path / "store")

    async def scenario() -> None:
        await _seed_run(store, "r1", "conv", "seed me")
        s = runtime.Session("assistant", conversation_id="conv", trace=False, store=store)
        n = await s.restore()
        assert n == 2  # two prior messages (user + assistant)
        assert s.restored_count == 2
        assert len(s.history) == 2

    asyncio.run(scenario())


def test_session_restore_fresh_conversation_is_empty(tmp_path: Path) -> None:
    """A conversation with no prior runs resumes empty (a genuinely new session)."""
    store = FileStepStore(tmp_path / "store")

    async def scenario() -> None:
        s = runtime.Session("assistant", conversation_id="conv", trace=False, store=store)
        n = await s.restore()
        assert n == 0
        assert s.restored_count == 0
        assert s.history == []

    asyncio.run(scenario())


def test_session_restore_respects_resume_false(tmp_path: Path) -> None:
    """resume=False must NOT seed history even when prior runs exist (probe/test use)."""
    store = FileStepStore(tmp_path / "store")

    async def scenario() -> None:
        await _seed_run(store, "r1", "conv", "seed me")
        s = runtime.Session(
            "assistant", conversation_id="conv", trace=False, store=store, resume=False
        )
        n = await s.restore()
        assert n == 0
        assert s.history == []

    asyncio.run(scenario())


def test_session_restore_is_idempotent(tmp_path: Path) -> None:
    """Calling restore() twice does not double-seed history."""
    store = FileStepStore(tmp_path / "store")

    async def scenario() -> None:
        await _seed_run(store, "r1", "conv", "seed me")
        s = runtime.Session("assistant", conversation_id="conv", trace=False, store=store)
        first = await s.restore()
        second = await s.restore()
        assert first == 2
        assert second == 2
        assert len(s.history) == 2  # not 4

    asyncio.run(scenario())


class _Part:
    def __init__(self, content) -> None:
        self.content = content


class _NoContent:
    pass


@pytest.mark.live
def test_session_runs_one_turn_live() -> None:
    """Session.run_sync runs a real turn and appends to history."""
    with runtime.Session("assistant") as s:
        out = s.run_sync("Reply with exactly: pong")
        assert "pong" in out.lower()
        assert s.name == "assistant"
        # One user turn -> at least the user + assistant messages recorded.
        assert len(s.history) >= 2


@pytest.mark.live
def test_session_stream_emits_text_events_live() -> None:
    """Session.stream yields 'text' events and sets last_result after the run."""
    with runtime.Session("assistant") as s:
        events: list[runtime.StreamEvent] = []
        parts: list[str] = []

        async def run() -> None:
            async for ev in s.stream("Count from one to three in one line."):
                events.append(ev)
                if ev.kind == "text":
                    parts.append(ev.text)

        asyncio.run(run())

        assert events, "expected at least one event"
        assert any(ev.kind == "text" for ev in events), "expected text events"
        assert "".join(parts).strip(), "expected a non-empty assembled answer"
        assert s.last_result is not None, "expected last_result after stream"
        assert len(s.history) >= 2


@pytest.mark.live
def test_stream_event_repr_and_defaults() -> None:
    """StreamEvent exposes kind + optional fields with sane defaults."""
    ev = runtime.StreamEvent(kind="text", text="hi")
    assert ev.kind == "text"
    assert ev.text == "hi"
    assert ev.tool_name == ""
    assert ev.args == ""
