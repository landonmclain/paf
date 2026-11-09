"""Turn-boundary root refresh, failure recovery, and stable session settings."""

import asyncio
from types import SimpleNamespace

import pytest
import yaml
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.step_persistence import FileStepStore

from paf import config, runtime, specs
from paf.capabilities import AgentProfile
from paf.conversations import default_conversation_meta_store, default_project_store
from paf.project_roots import ProjectRootError


@pytest.fixture
def rooted_session(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(runtime, "APPROVALS_DIR", tmp_path / "approvals")
    base = tmp_path / "projects"
    base.mkdir()
    a, b = base / "a", base / "b"
    a.mkdir()
    b.mkdir()
    path = tmp_path / "demo.yaml"
    raw = {
        "name": "demo",
        "model": "test",
        "instructions": "original instructions",
        "filesystem": {"root_dir": str(tmp_path)},
        "workspace": {
            "project_roots": {"within": [str(base)]},
            "notes": ["original note"],
        },
    }
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(
        runtime,
        "spec_project_root_policy",
        lambda name: specs.spec_project_root_policy(name, tmp_path),
    )
    monkeypatch.setattr(
        runtime, "build_boundary_compaction_for_agent", lambda name: None
    )
    monkeypatch.setattr(runtime, "spec_usage_limits", lambda name: None)
    builds = []

    async def respond(messages, info):
        yield "ok"

    model = FunctionModel(stream_function=respond)

    def build(name, **kwargs):
        builds.append(kwargs)
        return specs.build_agent(name, spec_directory=tmp_path, model=model, **kwargs)

    monkeypatch.setattr(runtime, "build_agent", build)
    projects = default_project_store()
    projects.create("a", root=str(a))
    projects.create("b", root=str(b))
    meta = default_conversation_meta_store("demo")
    profile = AgentProfile("root-test", ("FileSystem", "Workspace"))

    def make(**kwargs):
        return runtime.Session(
            "demo",
            profile=profile,
            conversation_id="cid",
            trace=False,
            store=FileStepStore(directory=tmp_path / "steps"),
            **kwargs,
        )

    def write_policy(within):
        raw["workspace"]["project_roots"] = {"within": within}
        path.write_text(yaml.safe_dump(raw))

    return SimpleNamespace(
        make=make,
        projects=projects,
        meta=meta,
        builds=builds,
        a=a,
        b=b,
        base=base,
        path=path,
        raw=raw,
        policy=write_policy,
    )


def _turn(session, cid=None):
    async def run():
        return [event async for event in session.stream("hello", conversation_id=cid)]

    return asyncio.run(run())


def test_session_follows_assignment_edits_and_clear(rooted_session):
    f = rooted_session
    session = f.make()
    background = session.background
    f.meta.set_project("cid", "a")
    _turn(session)
    history = list(session.history)
    assert [b["project_root"] for b in f.builds] == [None, f.a]
    _turn(session)
    assert len(f.builds) == 2
    f.meta.set_project("cid", "b")
    _turn(session)
    assert f.builds[-1]["project_root"] == f.b
    f.projects.update("b", root=str(f.a))
    _turn(session)
    assert f.builds[-1]["project_root"] == f.a
    f.meta.set_project("cid", None)
    _turn(session)
    assert f.builds[-1]["project_root"] is None
    assert session.history[: len(history)] == history
    assert len(session.history) == 10
    assert session.background is background


def test_broken_root_keeps_history_readable_and_refuses_turn(rooted_session):
    f = rooted_session
    f.meta.set_project("cid", "a")
    missing = f.a.with_name("moved")
    f.a.rename(missing)
    session = f.make()
    assert asyncio.run(session.restore()) == 0
    builds = len(f.builds)
    with pytest.raises(ProjectRootError, match="Project 'a'"):
        _turn(session)
    assert len(f.builds) == builds  # no fallback build for the refused turn
    missing.rename(f.a)
    _turn(session)
    assert f.builds[-1]["project_root"] == f.a


def test_live_policy_narrow_widen_disable_enable_and_invalid(rooted_session):
    f = rooted_session
    f.meta.set_project("cid", "a")
    session = f.make()
    _turn(session)
    f.policy([str(f.b)])
    before = len(f.builds)
    with pytest.raises(ProjectRootError, match="outside"):
        _turn(session)
    assert len(f.builds) == before
    f.policy([str(f.base)])
    _turn(session)
    f.policy(["relative"])
    with pytest.raises(ProjectRootError, match="policy"):
        _turn(session)
    assert len(f.builds) == before
    f.policy([])
    _turn(session)
    assert f.builds[-1]["project_root"] is None
    # A broken root is ignored while policy is explicitly off.
    f.a.rename(f.a.with_name("missing"))
    _turn(session)
    f.policy([str(f.base)])
    with pytest.raises(ProjectRootError):
        _turn(session)
    f.a.with_name("missing").rename(f.a)
    _turn(session)
    assert f.builds[-1]["project_root"] == f.a


def test_missing_project_record_and_symlink_retarget_refuse(rooted_session):
    f = rooted_session
    session = f.make()
    f.meta.set_project("cid", "missing")
    with pytest.raises(ProjectRootError, match="record is missing"):
        _turn(session)
    f.projects.create("missing")
    _turn(session)
    link = f.base / "link"
    link.symlink_to(f.a, target_is_directory=True)
    f.projects.update("missing", root=str(link))
    _turn(session)
    assert f.builds[-1]["project_root"] == f.a
    link.unlink()
    link.symlink_to(f.path.parent, target_is_directory=True)
    with pytest.raises(ProjectRootError, match="outside"):
        _turn(session)
    f.meta.set_project("cid", None)
    _turn(session)


def test_turn_uses_explicit_cid_and_preserves_maintenance_coordinators(rooted_session):
    f = rooted_session
    session = f.make()
    f.meta.set_project("other", "b")

    class Maintenance:
        def __init__(self):
            self.agent = object()
            self.conversation = object()

        async def after_turn(self, cid):
            pass

    maintenance = Maintenance()
    session.background = maintenance
    _turn(session, "other")
    assert f.builds[-1]["project_root"] == f.b
    assert session.background is maintenance
    _turn(session)
    assert f.builds[-1]["project_root"] is None
    assert session.background is maintenance


def test_rebuild_keeps_loaded_configuration(rooted_session):
    f = rooted_session
    session = f.make()
    f.raw["instructions"] = "changed instructions"
    f.raw["model"] = "invalid-model-after-session-start"
    f.raw["filesystem"]["denied_patterns"] = ["newly-denied"]
    f.raw["workspace"]["notes"] = ["changed note"]
    f.path.write_text(yaml.safe_dump(f.raw))
    f.meta.set_project("cid", "a")
    _turn(session)
    caps = {type(c).__name__: c for c in session.agent.root_capability.capabilities}
    assert "original note" in caps["Workspace"].notes
    assert "changed note" not in caps["Workspace"].notes
    assert "newly-denied" not in caps["FileSystem"].denied_patterns
    assert f.builds[-1]["snapshot"].spec.instructions == "original instructions"


def test_root_error_reaches_turn_handle(rooted_session):
    f = rooted_session
    session = f.make()
    f.meta.set_project("cid", "missing")

    async def run():
        handle = session.start_turn("hello")
        async for _ in handle.subscribe():
            pass
        assert isinstance(handle.error, ProjectRootError)
        assert "missing" in str(handle.error)
        assert handle.outcome == "error"

    asyncio.run(run())


def test_invalid_initial_policy_raises(rooted_session):
    f = rooted_session
    f.policy(["relative"])
    with pytest.raises(ProjectRootError, match="policy"):
        f.make()
    assert not f.builds


def test_rebuild_preserves_delivery_hooks(rooted_session):
    f = rooted_session

    async def completed(*args):
        pass

    async def push(request):
        pass

    queue = object()
    session = f.make(
        on_complete=completed, scheduled_tasks_queue=queue, push_notifier=push
    )
    original_push_hook = f.builds[0]["push_notifier"]
    f.meta.set_project("cid", "a")
    _turn(session)
    assert all(build["on_complete"] is completed for build in f.builds)
    assert all(build["scheduled_tasks_queue"] is queue for build in f.builds)
    assert all(build["push_notifier"] == original_push_hook for build in f.builds)
