"""Tests for the conversation list/create endpoints (ui/sse.py).

Covers `GET /api/conversations` (list lanes for the left sidebar) and
`POST /api/conversations` (mint a fresh lane id for "New conversation").
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC
from typing import Any

from fastapi.testclient import TestClient

from ui import sse


def _reset_registry() -> None:
    sse._sessions.clear()
    sse._sessions_last_used.clear()


@contextmanager
def _empty_store(tmp_path=None, monkeypatch=None):
    """Isolate the store backstop behind an empty FileStepStore in a temp dir.

    Works with or without pytest fixtures: when no monkeypatch is supplied it
    patches via unittest.mock so the endpoint never scans the real data/paf dir
    during tests.
    """
    import tempfile

    from pydantic_ai_harness.step_persistence import FileStepStore


    root = tmp_path if tmp_path is not None else tempfile.mkdtemp()
    store = FileStepStore(directory=str(root))

    if monkeypatch is not None:
        monkeypatch.setattr("paf.capabilities.persistence._build_store", lambda: store)
        yield store
        return

    from unittest import mock

    with mock.patch("paf.capabilities.persistence._build_store", lambda: store):
        yield store


class _FakeSession:
    """Minimal stand-in with just the bits list_conversations touches."""

    def __init__(self, history: list[Any]) -> None:
        self.history = history


def test_list_conversations_returns_lane() -> None:
    _reset_registry()
    # Seed one lane for a different agent — must NOT appear (agent filter).
    sse._sessions_last_used["other_agent:lane-x"] = 100.0
    # Seed the default agent's lane with a session carrying a last prompt.
    key = sse._session_key(sse.DEFAULT_AGENT, "conv-1")
    sse._sessions[key] = _FakeSession([_mk_user("hello there")])
    sse._sessions_last_used[key] = 200.0

    with _empty_store():
        client = TestClient(sse.app)
        resp = client.get("/api/conversations")
        assert resp.status_code == 200
        body = resp.json()
        ids = [c["conversation_id"] for c in body["conversations"]]
        assert ids == ["conv-1"], f"expected only the default agent's lane, got {ids}"
        assert body["conversations"][0]["title"] == "hello there"


def test_list_conversations_newest_first(tmp_path, monkeypatch) -> None:
    _reset_registry()
    prefix = sse._session_key(sse.DEFAULT_AGENT, "")
    sse._sessions_last_used[prefix + "old"] = 100.0
    sse._sessions_last_used[prefix + "new"] = 300.0
    sse._sessions_last_used[prefix + "mid"] = 200.0

    with _empty_store(tmp_path, monkeypatch):
        client = TestClient(sse.app)
        body = client.get("/api/conversations").json()
        assert [c["conversation_id"] for c in body["conversations"]] == ["new", "mid", "old"]


def test_create_conversation_mints_id_and_registers(tmp_path, monkeypatch) -> None:
    _reset_registry()
    with _empty_store(tmp_path, monkeypatch):
        client = TestClient(sse.app)
        resp = client.post("/api/conversations")
        assert resp.status_code == 200
        data = resp.json()
        cid = data["conversation_id"]
        assert cid.startswith("paf-") and len(cid) > 8

        # The new lane must appear in the list.
        body = client.get("/api/conversations").json()
        assert cid in [c["conversation_id"] for c in body["conversations"]]


def test_create_conversation_agent_scoped(tmp_path, monkeypatch) -> None:
    _reset_registry()
    with _empty_store(tmp_path, monkeypatch):
        client = TestClient(sse.app)
        data = client.post("/api/conversations", params={"agent": "helper"}).json()
        body = client.get("/api/conversations", params={"agent": "helper"}).json()
        assert data["conversation_id"] in [
            c["conversation_id"] for c in body["conversations"]
        ]
        # And it must NOT leak into the default agent's list.
        default = client.get("/api/conversations").json()
        assert data["conversation_id"] not in [
            c["conversation_id"] for c in default["conversations"]
        ]


def test_list_conversations_new_conversation_fallback_title(tmp_path, monkeypatch) -> None:
    _reset_registry()
    # A freshly-minted lane is registered in _sessions_last_used but has no
    # session yet (lazily created on first /api/chat) — title must not be the
    # raw id; it falls back to "New conversation".
    prefix = sse._session_key(sse.DEFAULT_AGENT, "")
    sse._sessions_last_used[prefix + "fresh"] = 100.0

    with _empty_store(tmp_path, monkeypatch):
        client = TestClient(sse.app)
        body = client.get("/api/conversations").json()
        (conv,) = [c for c in body["conversations"] if c["conversation_id"] == "fresh"]
        assert conv["title"] == "New conversation"


def test_list_conversations_store_backstop_includes_profile_lane(
    tmp_path, monkeypatch
) -> None:
    """Persisted lanes scoped to the agent's profile survive a (simulated)
    restart — the store backstop surfaces them even with an empty live registry."""
    _reset_registry()

    from pydantic_ai_harness.step_persistence import FileStepStore

    store = FileStepStore(directory=str(tmp_path))

    async def seed() -> None:
        async def helper(cid: str, run_id: str, when) -> None:
            await _register_run(store, run_id, cid, when)

        # A real lane for the default agent's profile: demo:<lane>.
        await helper("demo:persisted-lane", "run-persist-1", 1000.0)
        # A lane for a DIFFERENT profile must not leak in.
        await helper("other-profile:their-lane", "run-other-1", 2000.0)
        # An auto-generated UUID conversation id (benchmark/test noise) must not appear.
        await helper("019f0000-0000-7000-8000-000000000000", "run-uuid-1", 3000.0)

    import asyncio
    asyncio.run(seed())

    monkeypatch.setattr("paf.capabilities.persistence._build_store", lambda: store)

    client = TestClient(sse.app)
    body = client.get("/api/conversations").json()
    ids = [c["conversation_id"] for c in body["conversations"]]
    assert "persisted-lane" in ids, f"store lane missing from list: {ids}"
    assert "their-lane" not in ids
    assert not any(c.startswith("019f") for c in ids)
    # Store lanes have no live session — the id is shown so the lane is
    # identifiable (not a generic placeholder).
    lane = next(c for c in body["conversations"] if c["conversation_id"] == "persisted-lane")
    assert lane["title"] == "persisted-lane"


async def _register_run(store, run_id: str, cid: str, started_epoch: float) -> None:
    """Write a minimal RunRecord into the temp store."""
    from datetime import datetime

    from pydantic_ai_harness.step_persistence import RunRecord

    record = RunRecord(
        run_id=run_id,
        conversation_id=cid,
        parent_run_id=None,
        agent_name=None,
        metadata={},
        started_at=datetime.fromtimestamp(started_epoch, tz=UTC),
    )
    await store.register_run(record)


def _mk_user(content: str) -> Any:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    return ModelRequest(parts=[UserPromptPart(content=content)])
