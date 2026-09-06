"""Tests for the conversation list/create endpoints (ui/sse.py).

Covers `GET /api/conversations` (list lanes for the left sidebar) and
`POST /api/conversations` (mint a fresh lane id for "New conversation").
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from ui import sse


def _reset_registry() -> None:
    sse._sessions.clear()
    sse._sessions_last_used.clear()


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

    client = TestClient(sse.app)
    resp = client.get("/api/conversations")
    assert resp.status_code == 200
    body = resp.json()
    ids = [c["conversation_id"] for c in body["conversations"]]
    assert ids == ["conv-1"], f"expected only the default agent's lane, got {ids}"
    assert body["conversations"][0]["title"] == "hello there"


def test_list_conversations_newest_first() -> None:
    _reset_registry()
    prefix = sse._session_key(sse.DEFAULT_AGENT, "")
    sse._sessions_last_used[prefix + "old"] = 100.0
    sse._sessions_last_used[prefix + "new"] = 300.0
    sse._sessions_last_used[prefix + "mid"] = 200.0

    client = TestClient(sse.app)
    body = client.get("/api/conversations").json()
    assert [c["conversation_id"] for c in body["conversations"]] == ["new", "mid", "old"]


def test_create_conversation_mints_id_and_registers() -> None:
    _reset_registry()
    client = TestClient(sse.app)
    resp = client.post("/api/conversations")
    assert resp.status_code == 200
    data = resp.json()
    cid = data["conversation_id"]
    assert cid.startswith("paf-") and len(cid) > 8

    # The new lane must appear in the list.
    body = client.get("/api/conversations").json()
    assert cid in [c["conversation_id"] for c in body["conversations"]]


def test_create_conversation_agent_scoped() -> None:
    _reset_registry()
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


def test_list_conversations_new_conversation_fallback_title() -> None:
    _reset_registry()
    # A freshly-minted lane is registered in _sessions_last_used but has no
    # session yet (lazily created on first /api/chat) — title must not be the
    # raw id; it falls back to "New conversation".
    prefix = sse._session_key(sse.DEFAULT_AGENT, "")
    sse._sessions_last_used[prefix + "fresh"] = 100.0

    client = TestClient(sse.app)
    body = client.get("/api/conversations").json()
    (conv,) = [c for c in body["conversations"] if c["conversation_id"] == "fresh"]
    assert conv["title"] == "New conversation"


def _mk_user(content: str) -> Any:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    return ModelRequest(parts=[UserPromptPart(content=content)])
