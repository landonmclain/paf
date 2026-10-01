"""Tests for paf.conversations — the project registry + conversation meta stores."""

from __future__ import annotations

import pytest

from paf.capabilities._storage import FlatFileStorage
from paf.conversations import (
    ConversationMetaStore,
    ProjectExistsError,
    ProjectNotFoundError,
    ProjectStore,
    slugify,
)


def _projects(tmp_path) -> ProjectStore:
    return ProjectStore(FlatFileStorage(tmp_path))


def _meta(tmp_path, agent: str = "agent_a") -> ConversationMetaStore:
    return ConversationMetaStore(FlatFileStorage(tmp_path), agent)


def test_slugify() -> None:
    assert slugify("example.net") == "example-net"
    assert slugify("  Paf  ") == "paf"
    assert slugify("Trading / QC") == "trading-qc"
    assert slugify("!!!") == ""


def test_create_and_list_projects(tmp_path) -> None:
    store = _projects(tmp_path)
    paf = store.create("paf", "Agent framework")
    store.create("Kitestring")
    assert paf["slug"] == "paf" and paf["description"] == "Agent framework"
    assert [p["slug"] for p in store.list()] == ["staging", "paf"]
    assert (tmp_path / "projects" / "projects.json").exists()


def test_create_rejects_duplicates_reserved_and_empty(tmp_path) -> None:
    store = _projects(tmp_path)
    store.create("paf")
    with pytest.raises(ProjectExistsError):
        store.create("PAF")  # same slug
    with pytest.raises(ValueError):
        store.create("none")
    with pytest.raises(ValueError):
        store.create("   ")


def test_update_keeps_slug_and_archive_hides(tmp_path) -> None:
    store = _projects(tmp_path)
    store.create("site")
    renamed = store.update("site", name="example.net", description="Astro site")
    assert renamed["slug"] == "site" and renamed["name"] == "example.net"

    store.update("site", archived=True)
    assert store.list() == []
    assert [p["slug"] for p in store.list(include_archived=True)] == ["site"]
    # An archived slug stays taken.
    with pytest.raises(ProjectExistsError):
        store.create("site")
    with pytest.raises(ProjectNotFoundError):
        store.update("missing", name="x")


def test_meta_clearing_title_keeps_project(tmp_path) -> None:
    meta = _meta(tmp_path)
    meta.set_title("c1", "My chat")
    meta.set_project("c1", "paf")
    meta.set_title("c1", "")
    assert meta.get("c1").get("title") is None
    assert meta.project_for("c1") == "paf"


def test_meta_entry_dropped_when_empty(tmp_path) -> None:
    meta = _meta(tmp_path)
    meta.set_project("c1", "paf")
    meta.set_project("c1", None)
    assert "c1" not in meta.load()


def test_meta_agent_scoped(tmp_path) -> None:
    _meta(tmp_path, "agent_a").set_project("c1", "paf")
    assert _meta(tmp_path, "agent_b").project_for("c1") is None


def test_meta_reads_legacy_title_entries(tmp_path) -> None:
    """Existing conversation-titles.json rows ({title, updated_at}) still load."""
    FlatFileStorage(tmp_path).write(
        "conversation-titles", "agent_a", {"c1": {"title": "old", "updated_at": 1.0}}
    )
    meta = _meta(tmp_path)
    assert meta.get("c1")["title"] == "old"
    assert meta.project_for("c1") is None


def test_meta_hidden_flag(tmp_path) -> None:
    meta = _meta(tmp_path)
    meta.set_hidden("c1", True)
    assert meta.get("c1")["hidden"] is True
    meta.set_project("c1", "paf")
    meta.set_hidden("c1", False)
    assert "hidden" not in meta.get("c1")
    assert meta.project_for("c1") == "paf"
    meta.set_project("c1", None)
    assert "c1" not in meta.load()


def test_lane_id_strips_only_the_profile_prefix() -> None:
    from paf.conversations import lane_id

    assert lane_id("demo:paf-1", "demo") == "paf-1"
    assert lane_id("paf-1", "demo") == "paf-1"  # bare ids pass through
    assert lane_id("other:paf-1", "demo") == "other:paf-1"  # not our prefix
    assert lane_id("demo:a:b", "demo") == "a:b"  # lane ids may contain colons
    assert lane_id("demo:paf-1", None) == "demo:paf-1"  # no profile, no strip


def test_project_for_conversation_scoped_and_bare(tmp_path) -> None:
    from paf.conversations import project_for_conversation

    meta = _meta(tmp_path, "demo_assistant")
    meta.set_project("paf-1", "paf")

    def lookup(cid, **kw):
        return project_for_conversation(
            "demo_assistant", cid, profile="demo", store=meta, **kw
        )

    assert lookup("demo:paf-1") == "paf"
    assert lookup("paf-1") == "paf"
    assert lookup("demo:paf-2") is None
    assert lookup(None) is None


def test_project_for_conversation_resolves_profile_from_spec(tmp_path) -> None:
    """Without an explicit profile, the agent spec's `profile:` key is used
    (demo_assistant.yaml declares `profile: demo`)."""
    from paf.conversations import project_for_conversation

    meta = _meta(tmp_path, "demo_assistant")
    meta.set_project("paf-1", "paf")
    assert project_for_conversation("demo_assistant", "demo:paf-1", store=meta) == "paf"
