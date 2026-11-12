"""Project roots: host policy, directory resolution, and assignment boundaries."""

import subprocess
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from paf.capabilities._storage import FlatFileStorage
from paf.conversations import ConversationMetaStore, ProjectStore
from paf.project_roots import (
    ProjectRootError,
    ProjectRootPolicy,
    check_root,
    conversation_root,
    is_git_work_tree,
    with_project_root,
)
from paf.specs import _spec_workspace_config, spec_project_root_policy


def test_check_root_directory_boundary(tmp_path):
    base = tmp_path / "projects"
    base.mkdir()
    inside = base / "repo"
    inside.mkdir()
    sibling = tmp_path / "projects-old"
    sibling.mkdir()
    file = base / "file"
    file.touch()
    link = base / "escape"
    link.symlink_to(sibling, target_is_directory=True)
    policy = ProjectRootPolicy((base,))
    assert check_root(str(base), policy) == base
    assert check_root(str(inside), policy) == inside
    for root in (sibling, link, file, base / "missing", Path("relative")):
        with pytest.raises(ProjectRootError):
            check_root(str(root), policy)


def test_conversation_resolution_and_recovery(tmp_path):
    projects = ProjectStore(FlatFileStorage(tmp_path / "registry"))
    meta = ConversationMetaStore(FlatFileStorage(tmp_path / "meta"), "demo")
    policy = ProjectRootPolicy((tmp_path,))

    def resolve(policy=policy, cid="profile:cid"):
        return conversation_root(
            "demo", cid, profile="profile", policy=policy, projects=projects, meta=meta
        )

    assert resolve() is None
    meta.set_project("cid", "missing")
    assert resolve(None) is None
    with pytest.raises(ProjectRootError, match="missing.*repair"):
        resolve()
    projects.create("missing")
    assert resolve() is None
    projects.update("missing", root=str(tmp_path), archived=True)
    assert resolve() == tmp_path
    assert resolve(cid=None) is None
    assert resolve(None) is None


def test_root_overrides_are_strings_and_preserve_options(tmp_path):
    original = {
        "filesystem_config": {"root_dir": "/old", "denied_patterns": ["private"]},
        "shell_config": {
            "executor": {"backend": "docker"},
            "command_policy": {"default": "deny"},
        },
        "python_config": None,
    }
    result = with_project_root(original, tmp_path)
    for key, attr in (
        ("filesystem_config", "root_dir"),
        ("shell_config", "cwd"),
        ("python_config", "cwd"),
        ("git_config", "root"),
        ("codebase_map_config", "scan_root"),
    ):
        assert result[key][attr] == str(tmp_path)
    assert result["shell_config"]["executor"] == original["shell_config"]["executor"]
    assert (
        result["shell_config"]["command_policy"]
        == original["shell_config"]["command_policy"]
    )
    assert result["filesystem_config"]["denied_patterns"] == ["private"]
    assert original["filesystem_config"]["root_dir"] == "/old"
    assert (
        with_project_root(original, tmp_path, keep_existing=True)["filesystem_config"][
            "root_dir"
        ]
        == "/old"
    )


@pytest.mark.parametrize("workspace", [{}, {"project_roots": {"within": []}}])
def test_policy_disabled_explicitly(tmp_path, workspace):
    path = tmp_path / "demo.yaml"
    path.write_text(yaml.safe_dump({"workspace": workspace}))
    assert spec_project_root_policy("demo", tmp_path) is None


@pytest.mark.parametrize(
    "block",
    [
        None,
        [],
        {},
        {"within": None},
        {"within": "~/projects"},
        {"within": [""]},
        {"within": [12]},
        {"within": ["relative"]},
        {"within": ["/valid", "relative"]},
    ],
)
def test_invalid_policy_never_disables(tmp_path, block):
    path = tmp_path / "demo.yaml"
    path.write_text(yaml.safe_dump({"workspace": {"project_roots": block}}))
    with pytest.raises(ProjectRootError, match="demo.yaml"):
        spec_project_root_policy("demo", tmp_path)
    with pytest.raises(ProjectRootError):
        _spec_workspace_config(path)


def test_policy_expands_and_resolves_and_errors_on_bad_spec(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "demo.yaml"
    path.write_text("workspace:\n  project_roots:\n    within: [~/projects]\n")
    assert spec_project_root_policy("demo", tmp_path).within == (tmp_path / "projects",)
    assert _spec_workspace_config(path)["project_roots"]["within"] == ["~/projects"]
    path.write_text("workspace: [")
    with pytest.raises(ProjectRootError, match="demo.yaml"):
        spec_project_root_policy("demo", tmp_path)
    with pytest.raises(ProjectRootError):
        spec_project_root_policy("missing", tmp_path)


def test_git_work_tree_detection(tmp_path):
    assert not is_git_work_tree(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    child = tmp_path / "child"
    child.mkdir()
    assert is_git_work_tree(tmp_path)
    assert is_git_work_tree(child)


def test_assignment_routes_refuse_outside_policy(tmp_path, monkeypatch):
    from ui import sse

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    projects = ProjectStore(FlatFileStorage(tmp_path / "registry"))
    projects.create("outside", root=str(tmp_path))
    meta = ConversationMetaStore(FlatFileStorage(tmp_path / "meta"), "demo")
    monkeypatch.setattr(sse, "_project_store", lambda: projects)
    monkeypatch.setattr(sse, "_meta_store", lambda agent: meta)
    monkeypatch.setattr(
        sse, "spec_project_root_policy", lambda agent: ProjectRootPolicy((allowed,))
    )
    client = TestClient(sse.app)
    for response in (
        client.post("/api/conversations", params={"project": "outside"}),
        client.patch(
            "/api/conversations/cid", json={"project": "outside", "title": "untouched"}
        ),
    ):
        assert response.status_code == 400
        assert "outside" in response.json()["detail"]
    assert meta.get("cid") == {}
    projects.update("outside", root=str(allowed))
    assert (
        client.patch("/api/conversations/cid", json={"project": "outside"}).status_code
        == 200
    )


def test_cli_assignment_refuses_invalid_root(tmp_path, monkeypatch):
    from paf.cli import chat

    app = object.__new__(chat.ChatApp)
    app.agent = "demo"
    app.cid = "cid"
    app.projects = ProjectStore(FlatFileStorage(tmp_path / "registry"))
    app.meta = ConversationMetaStore(FlatFileStorage(tmp_path / "meta"), "demo")
    app.projects.create("outside", root=str(tmp_path))
    monkeypatch.setattr(
        chat,
        "spec_project_root_policy",
        lambda agent: ProjectRootPolicy((tmp_path / "allowed",)),
    )
    output = []
    monkeypatch.setattr(chat, "_out", output.append)
    app._project_command("outside")
    assert app.meta.project_for("cid") is None
    assert "outside" in output[-1]


def _assert_built_roots(caps, root):
    expected = {
        "FileSystem": "root_dir",
        "Shell": "cwd",
        "Python": "cwd",
        "Git": "root",
        "CodebaseMap": "scan_root",
    }
    found = {type(c).__name__: c for c in caps}
    for name, attr in expected.items():
        if name in found:
            assert Path(getattr(found[name], attr)).resolve() == root
    workspace = found.get("Workspace")
    if workspace:
        assert all(r == root for label, _, _, r in workspace.roots if label != "Vault")
    return found


def test_build_agent_moves_all_tools_and_workspace(tmp_path, monkeypatch):
    from paf.capabilities import AgentProfile
    from paf.specs import build_agent

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    vault = tmp_path / "vault"
    vault.mkdir()
    (tmp_path / "demo.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "model": "test",
                "instructions": "test",
                "python": {"executor": {"backend": "local"}, "allow_unsandboxed": True},
                "vault": {"root": str(vault)},
                "workspace": {"notes": ["keep this note"]},
            }
        )
    )
    profile = AgentProfile(
        "test",
        ("FileSystem", "Shell", "Python", "Git", "CodebaseMap", "Workspace", "Vault"),
    )
    agent = build_agent(
        "demo", spec_directory=tmp_path, profile=profile, project_root=root
    )
    caps = _assert_built_roots(agent.root_capability.capabilities, root)
    assert {"FileSystem", "Shell", "Python", "Git", "CodebaseMap"} <= caps.keys()
    assert Path(caps["Vault"].vault_root) == vault
    assert "keep this note" in caps["Workspace"].render()
    assert "Example:" not in caps["Workspace"].render()
    default = build_agent("demo", spec_directory=tmp_path, profile=profile)
    assert (
        next(
            c
            for c in default.root_capability.capabilities
            if type(c).__name__ == "FileSystem"
        ).root_dir
        != root
    )


def test_non_repo_git_filter_is_shared_and_does_not_mutate(tmp_path):
    from paf.capabilities import AgentProfile
    from paf.capabilities.build import build_runtime_capabilities

    profile = AgentProfile("test", ("Git", "Workspace"))
    workspace = {"notes": ["keep"]}
    caps = build_runtime_capabilities(
        profile,
        git_config={"root": str(tmp_path)},
        workspace_config=workspace,
        project_root=tmp_path,
    )
    found = _assert_built_roots(caps, tmp_path)
    assert "Git" not in found
    assert "not a git repository" in found["Workspace"].render()
    assert "keep" in found["Workspace"].notes
    assert workspace == {"notes": ["keep"]}
    assert profile.capability_names == ("Git", "Workspace")


@pytest.mark.parametrize("explicit_repo", [None, False, True])
def test_leaf_common_root_and_git_availability(tmp_path, monkeypatch, explicit_repo):
    from paf.capabilities import AgentProfile
    from paf.capabilities.profiles import SUBAGENT_PROFILES
    from paf.capabilities.subagents import load_leaf

    root = tmp_path / "project"
    root.mkdir()
    if explicit_repo is not True:
        subprocess.run(["git", "init", "-q", str(root)], check=True)
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    if explicit_repo:
        subprocess.run(["git", "init", "-q", str(explicit)], check=True)
    monkeypatch.setitem(
        SUBAGENT_PROFILES,
        "root-test",
        AgentProfile(
            "root-test",
            ("FileSystem", "Shell", "Python", "Git", "CodebaseMap", "Workspace"),
        ),
    )
    spec = {
        "name": "leaf",
        "model": "test",
        "profile": "root-test",
        "instructions": "test",
        "shell": {"cwd": "/conflicting", "executor": {"backend": "local"}},
        "python": {"executor": {"backend": "local"}, "allow_unsandboxed": True},
    }
    if explicit_repo is not None:
        spec["filesystem"] = {"root_dir": str(explicit), "denied_patterns": ["private"]}
    (tmp_path / "leaf.yaml").write_text(yaml.safe_dump(spec))
    leaf = load_leaf(
        "leaf",
        store=None,
        filesystem_config={"root_dir": "/parent"},
        spec_directory=tmp_path,
        project_root=root,
    )
    effective = root if explicit_repo is None else explicit
    found = _assert_built_roots(leaf.agent.root_capability.capabilities, effective)
    assert ("Git" in found) == (explicit_repo is not False)
    if explicit_repo is not None:
        assert found["FileSystem"].denied_patterns == ["private"]
    if explicit_repo is False:
        assert "not a git repository" in found["Workspace"].render()


def test_coding_and_filesystem_only_leaves_follow_project(tmp_path, monkeypatch):
    from paf.capabilities.subagents import load_leaf

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for name in ("coding_subagent", "writer_subagent"):
        leaf = load_leaf(
            name,
            store=None,
            filesystem_config={"root_dir": "/parent"},
            project_root=tmp_path,
        )
        found = _assert_built_roots(leaf.agent.root_capability.capabilities, tmp_path)
        assert Path(found["FileSystem"].root_dir) == tmp_path
        if name == "coding_subagent":
            assert Path(found["CodebaseMap"].scan_root) == tmp_path


def test_leaf_rebuild_uses_loaded_spec_snapshot(tmp_path):
    from paf.capabilities.subagents import load_leaf

    path = tmp_path / "writer.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "writer",
                "model": "test",
                "instructions": "original",
                "filesystem": {"denied_patterns": ["original-denial"]},
            }
        )
    )
    snapshots = {}
    first = load_leaf(
        "writer",
        store=None,
        filesystem_config=None,
        spec_directory=tmp_path,
        project_root=tmp_path,
        leaf_snapshots=snapshots,
    )
    path.write_text("invalid: [")
    other = tmp_path / "other"
    other.mkdir()
    second = load_leaf(
        "writer",
        store=None,
        filesystem_config=None,
        spec_directory=tmp_path,
        project_root=other,
        leaf_snapshots=snapshots,
    )
    first_fs = next(
        c
        for c in first.agent.root_capability.capabilities
        if type(c).__name__ == "FileSystem"
    )
    second_fs = next(
        c
        for c in second.agent.root_capability.capabilities
        if type(c).__name__ == "FileSystem"
    )
    assert first_fs.root_dir == tmp_path
    assert second_fs.root_dir == other
    assert second_fs.denied_patterns == ["original-denial"]
    assert snapshots["writer"].spec.instructions == "original"
