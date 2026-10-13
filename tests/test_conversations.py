"""Tests for the conversation list/create endpoints (ui/sse.py).

Covers `GET /api/conversations` (list lanes for the left sidebar) and
`POST /api/conversations` (mint a fresh lane id for "New conversation").
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC
from typing import Any

from fastapi.testclient import TestClient

from paf.capabilities._storage import FlatFileStorage
from ui import sse


def _reset_registry() -> None:
    sse._sessions.clear()
    sse._sessions_last_used.clear()
    sse._sessions_touched.clear()


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
        assert [c["conversation_id"] for c in body["conversations"]] == [
            "new",
            "mid",
            "old",
        ]


def test_reads_do_not_reorder_conversations(tmp_path, monkeypatch) -> None:
    """Reading a lane (history/plan/process feed/rejoin) must not make it newest.

    Regression: a page load read the fallback lane before the list resolved,
    bumping it to the top, so the UI auto-selected it over the real latest.
    """
    import asyncio

    _reset_registry()
    prefix = sse._session_key(sse.DEFAULT_AGENT, "")
    for cid, t in (("old", 100.0), ("new", 300.0)):
        sse._sessions[prefix + cid] = _FakeSession([])
        sse._sessions_last_used[prefix + cid] = t

    asyncio.run(sse._get_session("old", sse.DEFAULT_AGENT))  # a read
    assert sse._sessions_touched[prefix + "old"] > 300.0  # keepalive still bumps

    with _empty_store(tmp_path, monkeypatch):
        client = TestClient(sse.app)
        ids = [
            c["conversation_id"]
            for c in client.get("/api/conversations").json()["conversations"]
        ]
        assert ids == ["new", "old"]

        asyncio.run(sse._get_session("old", sse.DEFAULT_AGENT, activity=True))
        ids = [
            c["conversation_id"]
            for c in client.get("/api/conversations").json()["conversations"]
        ]
        assert ids == ["old", "new"]


def test_read_only_lane_not_listed(tmp_path, monkeypatch) -> None:
    """A lane that was only read (e.g. the agent-wide process feed's
    conversation-less session) has no activity and no history: not listed."""
    import asyncio

    _reset_registry()
    key = sse._session_key(sse.DEFAULT_AGENT, "None")
    sse._sessions[key] = _FakeSession([])
    asyncio.run(sse._get_session("None", sse.DEFAULT_AGENT))

    with _empty_store(tmp_path, monkeypatch):
        client = TestClient(sse.app)
        assert client.get("/api/conversations").json()["conversations"] == []


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


def test_list_conversations_new_conversation_fallback_title(
    tmp_path, monkeypatch
) -> None:
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
    lane = next(
        c for c in body["conversations"] if c["conversation_id"] == "persisted-lane"
    )
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


@contextmanager
def _temp_titles_dir(tmp_path):
    """Redirect the conversation-title store to a temp dir so PATCH/GET title
    tests never touch the real ``data/conversations/`` tree.

    ``_title_store`` builds a ``FlatFileStorage`` rooted at the project's
    ``data/conversations`` dir; we swap it for one rooted at ``tmp_path`` so
    renames write under the test sandbox (storage root is the only thing the
    seam needs to change — proving the seam is swappable).
    """
    real_store = sse._title_store
    sse._title_store = lambda agent: FlatFileStorage(tmp_path)
    try:
        yield
    finally:
        sse._title_store = real_store


def test_patch_title_persists_and_merges_over_derived(tmp_path, monkeypatch) -> None:
    """PATCH sets a durable title; GET overlays it over the auto-derived one,
    and it survives across requests (persistence through the seam)."""
    _reset_registry()
    # Live lane with a derived title (from last prompt).
    key = sse._session_key(sse.DEFAULT_AGENT, "conv-t")
    sse._sessions[key] = _FakeSession([_mk_user("derived title")])
    sse._sessions_last_used[key] = 300.0

    with _empty_store(tmp_path, monkeypatch), _temp_titles_dir(tmp_path):
        client = TestClient(sse.app)

        # Before rename: derived title.
        body = client.get("/api/conversations").json()
        assert any(
            c["conversation_id"] == "conv-t" and c["title"] == "derived title"
            for c in body["conversations"]
        )

        # Rename.
        resp = client.patch("/api/conversations/conv-t", json={"title": "My Rename"})
        assert resp.status_code == 200
        assert resp.json()["title"] == "My Rename"

        # Merge: the saved title wins over the derived one.
        body = client.get("/api/conversations").json()
        lane = next(
            c for c in body["conversations"] if c["conversation_id"] == "conv-t"
        )
        assert lane["title"] == "My Rename"


def test_patch_title_agent_namespaced(tmp_path, monkeypatch) -> None:
    """A title set for one agent's lane must not leak into another agent's list."""
    _reset_registry()
    # Same conversation id in two agents' lanes.
    k_a = sse._session_key("agent_a", "share")
    k_b = sse._session_key("agent_b", "share")
    sse._sessions_last_used[k_a] = 100.0
    sse._sessions_last_used[k_b] = 100.0

    with _empty_store(tmp_path, monkeypatch), _temp_titles_dir(tmp_path):
        client = TestClient(sse.app)
        client.patch(
            "/api/conversations/share",
            json={"title": "A's title"},
            params={"agent": "agent_a"},
        )

        # agent_a's list shows the rename.
        body_a = client.get("/api/conversations", params={"agent": "agent_a"}).json()
        assert any(
            c["conversation_id"] == "share" and c["title"] == "A's title"
            for c in body_a["conversations"]
        )
        # agent_b's list must NOT show it.
        body_b = client.get("/api/conversations", params={"agent": "agent_b"}).json()
        assert any(
            c["conversation_id"] == "share" and c["title"] != "A's title"
            for c in body_b["conversations"]
        )


def test_patch_title_clear_restores_derived(tmp_path, monkeypatch) -> None:
    """Clearing a title (empty string) removes it and the list falls back to the
    auto-derived title."""
    _reset_registry()
    key = sse._session_key(sse.DEFAULT_AGENT, "conv-c")
    sse._sessions[key] = _FakeSession([_mk_user("origin")])
    sse._sessions_last_used[key] = 200.0

    with _empty_store(tmp_path, monkeypatch), _temp_titles_dir(tmp_path):
        client = TestClient(sse.app)
        client.patch("/api/conversations/conv-c", json={"title": "Temp"})
        client.patch("/api/conversations/conv-c", json={"title": ""})
        body = client.get("/api/conversations").json()
        lane = next(
            c for c in body["conversations"] if c["conversation_id"] == "conv-c"
        )
        assert lane["title"] == "origin"  # derived title restored


def test_patch_title_requires_title_param(tmp_path, monkeypatch) -> None:
    _reset_registry()
    with _empty_store(tmp_path, monkeypatch), _temp_titles_dir(tmp_path):
        client = TestClient(sse.app)
        resp = client.patch("/api/conversations/conv-x", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Projects + conversation project assignment
# ---------------------------------------------------------------------------


@contextmanager
def _temp_projects(tmp_path):
    """Redirect the project registry to a temp dir (never the real data/projects)."""
    from paf.conversations import ProjectStore

    real = sse._project_store
    sse._project_store = lambda: ProjectStore(FlatFileStorage(tmp_path / "reg"))
    try:
        yield
    finally:
        sse._project_store = real


def test_project_routes_create_list_update(tmp_path) -> None:
    with _temp_projects(tmp_path):
        client = TestClient(sse.app)
        resp = client.post("/api/projects", json={"name": "paf", "description": "x"})
        assert resp.status_code == 201
        assert resp.json()["slug"] == "paf"

        assert client.post("/api/projects", json={"name": "PAF"}).status_code == 409
        assert client.post("/api/projects", json={}).status_code == 400

        resp = client.patch("/api/projects/paf", json={"description": "framework"})
        assert resp.json()["description"] == "framework"
        assert client.patch("/api/projects/nope", json={"name": "y"}).status_code == 404

        client.patch("/api/projects/paf", json={"archived": True})
        assert client.get("/api/projects").json()["projects"] == []
        archived = client.get("/api/projects", params={"include_archived": True})
        assert [p["slug"] for p in archived.json()["projects"]] == ["paf"]


def test_assign_project_to_conversation(tmp_path, monkeypatch) -> None:
    _reset_registry()
    key = sse._session_key(sse.DEFAULT_AGENT, "conv-p")
    sse._sessions[key] = _FakeSession([_mk_user("derived")])
    sse._sessions_last_used[key] = 100.0

    with (
        _empty_store(tmp_path, monkeypatch),
        _temp_titles_dir(tmp_path),
        _temp_projects(tmp_path),
    ):
        client = TestClient(sse.app)
        client.post("/api/projects", json={"name": "paf"})

        resp = client.patch("/api/conversations/conv-p", json={"project": "paf"})
        assert resp.status_code == 200
        assert resp.json()["project"] == "paf"

        lane = client.get("/api/conversations").json()["conversations"][0]
        assert lane["project"] == "paf"
        assert lane["title"] == "derived"  # project-only patch leaves the title alone

        # Unknown project -> 404; unassign with null.
        resp = client.patch("/api/conversations/conv-p", json={"project": "nope"})
        assert resp.status_code == 404
        resp = client.patch("/api/conversations/conv-p", json={"project": None})
        assert resp.json()["project"] is None


def test_assign_archived_project_rejected(tmp_path, monkeypatch) -> None:
    _reset_registry()
    with (
        _empty_store(tmp_path, monkeypatch),
        _temp_titles_dir(tmp_path),
        _temp_projects(tmp_path),
    ):
        client = TestClient(sse.app)
        client.post("/api/projects", json={"name": "old"})
        client.patch("/api/projects/old", json={"archived": True})
        resp = client.patch("/api/conversations/c", json={"project": "old"})
        assert resp.status_code == 400


def test_clearing_title_keeps_project(tmp_path, monkeypatch) -> None:
    """Regression: clearing a title used to pop the whole entry."""
    _reset_registry()
    key = sse._session_key(sse.DEFAULT_AGENT, "conv-k")
    sse._sessions[key] = _FakeSession([_mk_user("origin")])
    sse._sessions_last_used[key] = 100.0

    with (
        _empty_store(tmp_path, monkeypatch),
        _temp_titles_dir(tmp_path),
        _temp_projects(tmp_path),
    ):
        client = TestClient(sse.app)
        client.post("/api/projects", json={"name": "paf"})
        client.patch(
            "/api/conversations/conv-k", json={"title": "Temp", "project": "paf"}
        )
        client.patch("/api/conversations/conv-k", json={"title": ""})
        lane = client.get("/api/conversations").json()["conversations"][0]
        assert lane["title"] == "origin"
        assert lane["project"] == "paf"


def test_hide_and_unhide_conversation(tmp_path, monkeypatch) -> None:
    _reset_registry()
    sse._sessions_last_used[sse._session_key(sse.DEFAULT_AGENT, "conv-h")] = 100.0

    with _empty_store(tmp_path, monkeypatch), _temp_titles_dir(tmp_path):
        client = TestClient(sse.app)
        resp = client.patch("/api/conversations/conv-h", json={"hidden": True})
        assert resp.json()["hidden"] is True
        lane = client.get("/api/conversations").json()["conversations"][0]
        assert lane["hidden"] is True  # still listed; the UI decides visibility

        client.patch("/api/conversations/conv-h", json={"hidden": False})
        lane = client.get("/api/conversations").json()["conversations"][0]
        assert lane["hidden"] is False


def test_create_conversation_in_project(tmp_path, monkeypatch) -> None:
    _reset_registry()
    with (
        _empty_store(tmp_path, monkeypatch),
        _temp_titles_dir(tmp_path),
        _temp_projects(tmp_path),
    ):
        client = TestClient(sse.app)
        client.post("/api/projects", json={"name": "paf"})
        resp = client.post("/api/conversations", params={"project": "paf"})
        assert resp.status_code == 200
        cid = resp.json()["conversation_id"]
        lane = next(
            c
            for c in client.get("/api/conversations").json()["conversations"]
            if c["conversation_id"] == cid
        )
        assert lane["project"] == "paf"

        assert (
            client.post("/api/conversations", params={"project": "nope"}).status_code
            == 404
        )


def test_list_conversations_reports_server_side_running_turns() -> None:
    """A lane with a turn in flight is flagged running (any device's turn), so every
    client's sidebar can show it; idle and persisted-only lanes are not."""
    _reset_registry()
    busy = sse._session_key(sse.DEFAULT_AGENT, "busy")
    idle = sse._session_key(sse.DEFAULT_AGENT, "idle")
    busy_session = _FakeSession([_mk_user("long task")])
    busy_session.current_turn = object()  # type: ignore[attr-defined]  # a live TurnHandle
    idle_session = _FakeSession([_mk_user("done already")])
    idle_session.current_turn = None  # type: ignore[attr-defined]
    sse._sessions[busy], sse._sessions_last_used[busy] = busy_session, 200.0
    sse._sessions[idle], sse._sessions_last_used[idle] = idle_session, 100.0

    with _empty_store():
        client = TestClient(sse.app)
        lanes = {
            c["conversation_id"]: c
            for c in client.get("/api/conversations").json()["conversations"]
        }
    assert lanes["busy"]["running"] is True
    assert lanes["idle"]["running"] is False


def test_list_conversations_tolerates_sessions_without_current_turn() -> None:
    _reset_registry()
    key = sse._session_key(sse.DEFAULT_AGENT, "plain")
    sse._sessions[key] = _FakeSession([_mk_user("hi")])  # no current_turn attr at all
    sse._sessions_last_used[key] = 1.0
    with _empty_store():
        lane = TestClient(sse.app).get("/api/conversations").json()["conversations"][0]
    assert lane["running"] is False
