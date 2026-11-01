"""Keyword curator: redaction, the queue, the pending gate, the trigger.

No live model: passes are driven with a fake curator. The coordinator trigger is
exercised against a throwaway processes root, with the opt-in settings and the
store location patched, so nothing touches the real ``data/`` tree.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic_ai_harness.step_persistence import FileStepStore

import paf.capabilities.keyword_curator as kc
from paf.background import BackgroundCoordinator
from paf.capabilities.keyword_curator import (
    CurateResult,
    Entity,
    KeywordCuratorQueue,
    KeywordJob,
    KeywordWatermark,
    PassResult,
    pending_changes,
    run_keyword_pass,
    scan_notes,
    scan_sources,
)
from paf.capabilities.keyword_curator._queue import summarize_pass
from paf.capabilities.keyword_curator._sources import REDACTED, redact_secrets
from paf.capabilities.profiles import AgentProfile

# --- redaction ---------------------------------------------------------------


@pytest.mark.parametrize(
    "secret_text",
    [
        "wp_password: vxBm lvKp fr8x hOmb Xtwc KSLV",
        "password = hunter2",
        'api_key = "sk-abc123def456ghi789jkl"',
        "Authorization: Bearer eyJhbGciOi.eyJzdWIiOiIx.SflKxwRJSMeKKF2QT4",
        "token=ghp_abcdefghijklmnopqrstuvwxyz0123",
        "AKIAABCDEFGHIJKLMNOP",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----",
        "xoxb-1234567890-abcdefghij",
        "the app password is vxBm lvKp fr8x hOmb Xtwc KSLV ok",
    ],
)
def test_redact_secrets_removes_credentials(secret_text: str) -> None:
    out = redact_secrets(secret_text)
    assert REDACTED in out
    for fragment in ("hunter2", "sk-abc", "ghp_", "AKIAABC", "MIIabc", "vxBm", "xoxb-"):
        assert fragment not in out


@pytest.mark.parametrize(
    "prose",
    [
        "this will work with some text here",
        "set the password policy to 12 characters",
        "tokens are counted per request; the token budget is 8k",
        "Use the Todoist API v1 and the REST key rotation doc",
        "authorization is handled by the gateway",
    ],
)
def test_redact_secrets_leaves_ordinary_prose(prose: str) -> None:
    assert redact_secrets(prose) == prose


def test_the_digest_redacts_but_the_source_file_is_untouched(tmp_path: Path) -> None:
    note = tmp_path / "Client" / "Index.md"
    note.parent.mkdir()
    raw = (
        "---\ntitle: Client site\nstatus: active\nwp_password: vxBm lvKp fr8x hOmb\n---\n"
        "# Client site\nLogin: password: hunter2 for the dev site.\n"
    )
    note.write_text(raw)
    item = scan_notes(tmp_path)["note:Client/Index.md"]
    assert "hunter2" not in item.text and "vxBm" not in item.text
    assert "Client site" in item.text  # the useful part survives
    assert note.read_text() == raw


# --- pending gate ------------------------------------------------------------


def _skill(root: Path, name: str, body: str = "Body.") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "SKILL.md"
    f.write_text(f"---\nname: {name}\ndescription: about {name}\n---\n{body}\n")
    return f


async def _fake_curate(batch, known):
    return [
        CurateResult(
            source_ref=it.ref,
            keywords=[it.label],
            entities=[Entity(canonical=it.label, aliases=[])],
        )
        for it in batch
    ]


def test_pending_changes_tracks_new_modified_and_vanished(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    f = _skill(skills, "alpha")
    _skill(skills, "beta")
    store = tmp_path / "kw"
    src = scan_sources(skill_libraries=[skills])
    assert pending_changes(store, src) == (["skill:alpha", "skill:beta"], [])
    asyncio.run(run_keyword_pass(store, src, curate=_fake_curate))
    assert pending_changes(store, src) == ([], [])
    f.write_text(f.read_text() + "\nmore text so the size changes\n")
    (skills / "beta" / "SKILL.md").unlink()
    src2 = scan_sources(skill_libraries=[skills])
    assert pending_changes(store, src2) == (["skill:alpha"], ["skill:beta"])


# --- the queue ---------------------------------------------------------------


def test_queue_runs_a_pass_and_reports_completion(tmp_path: Path) -> None:
    skills = tmp_path / "skills"
    _skill(skills, "alpha")
    store = tmp_path / "kw"
    seen: dict = {}
    done: list = []

    async def pass_fn(directory, sources, *, limit, seeds):
        seen.update(directory=directory, refs=sorted(sources), limit=limit, seeds=seeds)
        return await run_keyword_pass(
            directory, sources, curate=_fake_curate, limit=limit, seeds=seeds
        )

    async def on_complete(session_id, ok, detail):
        done.append((session_id, ok, detail))

    queue = KeywordCuratorQueue(pass_fn=pass_fn, on_complete=on_complete)

    async def scenario():
        queue.enqueue(
            "agent-x",
            KeywordJob(
                agent_name="agent-x",
                directory=str(store),
                sources_config={"skill_libraries": [skills]},
                limit=7,
                seed_projects=False,
            ),
        )
        await queue.drain("agent-x")

    asyncio.run(scenario())
    assert seen["refs"] == ["skill:alpha"] and seen["limit"] == 7
    assert seen["seeds"] == [] and seen["directory"] == store
    assert done == [("agent-x", True, "1 curated, 1 entities")]
    assert KeywordWatermark.load(store).files


def test_queue_survives_a_failing_pass_and_says_so(tmp_path: Path) -> None:
    done: list = []

    async def boom(*_a, **_k):
        raise RuntimeError("provider down")

    async def on_complete(session_id, ok, detail):
        done.append((ok, detail))

    queue = KeywordCuratorQueue(pass_fn=boom, on_complete=on_complete)

    async def scenario():
        job = KeywordJob("a", str(tmp_path / "kw"), {"skill_libraries": []})
        queue.enqueue("a", job)
        await queue.drain("a")
        # the worker is still alive for the next job
        queue.enqueue("a", job)
        await queue.drain("a")

    asyncio.run(scenario())
    assert len(done) == 2 and all(not ok for ok, _ in done)
    assert "RuntimeError: provider down" in done[0][1]


def test_summarize_pass_lines() -> None:
    r = PassResult(keyed=["a", "b"], failed={"c": "x"}, removed=["d"], entities=5)
    assert summarize_pass(r) == "2 curated, 1 failed, 1 dropped, 5 entities"
    assert summarize_pass(PassResult(entities=0)) == "0 curated, 0 entities"


# --- settings ----------------------------------------------------------------


def test_curate_is_off_by_default_and_for_unknown_agents() -> None:
    s = kc.curate_settings("no-such-agent-anywhere")
    assert s == {"curate": False, "curate_limit": 25, "curate_cooldown_s": 300.0}


def test_spec_keywords_block_parses_the_curate_opt_in(tmp_path: Path) -> None:
    import yaml

    from paf.specs import _spec_keywords_config

    spec = tmp_path / "a.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "name": "a",
                "model": "x:y",
                "keywords": {
                    "curate": True,
                    "curate_limit": 10,
                    "curate_cooldown_s": 60,
                },
            }
        )
    )
    assert _spec_keywords_config(spec) == {
        "curate": True,
        "curate_limit": 10,
        "curate_cooldown_s": 60.0,
    }
    spec.write_text(
        yaml.safe_dump(
            {
                "name": "a",
                "model": "x:y",
                "keywords": {"curate": "yes", "curate_limit": 0},
            }
        )
    )
    assert _spec_keywords_config(spec) is None


# --- the coordinator trigger -------------------------------------------------


@pytest.fixture
def _env(request):
    """Build :class:`_Env` instances and close their event loops afterwards."""
    made: list[_Env] = []

    def make(*args, **kwargs) -> _Env:
        env = _Env(*args, **kwargs)
        made.append(env)
        return env

    request.addfinalizer(lambda: [e.close() for e in made])
    return make


class _Env:
    """A coordinator wired to tmp dirs, with curation opt-in controllable."""

    def __init__(self, tmp_path: Path, monkeypatch, *, curate: bool, profile=None):
        self.skills = tmp_path / "skills"
        _skill(self.skills, "alpha")
        self.store = tmp_path / "kw"
        self.passes: list[dict] = []
        monkeypatch.setattr(kc, "keyword_dir_for", lambda _a: self.store)
        monkeypatch.setattr(
            kc, "sources_config_for", lambda _a: {"skill_libraries": [self.skills]}
        )
        monkeypatch.setattr(
            kc,
            "curate_settings",
            lambda _a: {"curate": curate, "curate_limit": 3, "curate_cooldown_s": 0.0},
        )

        async def pass_fn(directory, sources, *, limit, seeds):
            self.passes.append({"refs": sorted(sources), "limit": limit})
            return await run_keyword_pass(
                directory, sources, curate=_fake_curate, limit=limit, seeds=seeds
            )

        self._runner = asyncio.Runner()
        self.bg = BackgroundCoordinator(
            FileStepStore(directory=str(tmp_path / "steps")),
            agent_name="kc-agent",
            profile=profile
            or AgentProfile(name="kc", capability_names=("KeywordCurator",)),
            lane="agent",
            processes_root=str(tmp_path / "proc"),
            keyword_pass=pass_fn,
        )

    def close(self) -> None:
        self._runner.close()

    def run(self, coro) -> None:
        # One event loop for the whole scenario: the queue's internal events bind
        # to the loop that first uses them.
        self._runner.run(coro)

    def trigger(self) -> None:
        self.bg._keyword_last_check = float("-inf")  # defeat the scan throttle
        self.run(self._go())

    async def _go(self):
        await self.bg.after_turn("cid")
        queue = self.bg._keyword_queue
        if queue is not None:
            await queue.drain("kc-agent")

    def tasks(self):
        return [
            t for t in self.bg._background_tasks.values() if "keyword_curate" in t.id
        ]


def test_trigger_does_nothing_unless_curation_is_opted_in(
    tmp_path: Path, monkeypatch, _env
) -> None:
    env = _env(tmp_path, monkeypatch, curate=False)
    env.trigger()
    assert env.passes == [] and env.tasks() == []
    assert env.bg._keyword_queue is None  # not even built


def test_trigger_not_registered_without_the_capability(
    tmp_path: Path, monkeypatch, _env
) -> None:
    env = _env(
        tmp_path,
        monkeypatch,
        curate=True,
        profile=AgentProfile(name="none", capability_names=("StepPersistence",)),
    )
    env.trigger()
    assert env.passes == [] and env.tasks() == []
    assert "keyword_curate" not in {p.id_namespace for p in env.bg._processes}


def test_trigger_enqueues_a_capped_pass_and_publishes_a_task(
    tmp_path: Path, monkeypatch, _env
) -> None:
    env = _env(tmp_path, monkeypatch, curate=True)
    env.trigger()
    assert env.passes == [{"refs": ["skill:alpha"], "limit": 3}]
    [task] = env.tasks()
    assert task.status == "done" and "1 curated" in task.detail
    assert task.id.startswith("keyword_curate.kc-agent.")


def test_nothing_changed_means_no_pass_and_no_task_row(
    tmp_path: Path, monkeypatch, _env
) -> None:
    env = _env(tmp_path, monkeypatch, curate=True)
    env.trigger()
    assert len(env.passes) == 1
    env.trigger()  # sources unchanged -> silent
    assert len(env.passes) == 1 and len(env.tasks()) == 1


def test_a_changed_source_triggers_another_pass(
    tmp_path: Path, monkeypatch, _env
) -> None:
    env = _env(tmp_path, monkeypatch, curate=True)
    env.trigger()
    f = env.skills / "alpha" / "SKILL.md"
    f.write_text(f.read_text() + "\nedited so the size changes\n")
    env.trigger()
    assert len(env.passes) == 2 and len(env.tasks()) == 2


def test_the_source_scan_is_throttled_between_turns(
    tmp_path: Path, monkeypatch, _env
) -> None:
    env = _env(tmp_path, monkeypatch, curate=True)
    env.trigger()
    f = env.skills / "alpha" / "SKILL.md"
    f.write_text(f.read_text() + "\nedited so the size changes\n")

    async def again():
        await env.bg.after_turn("cid")  # last_check was just set -> throttled
        if env.bg._keyword_queue is not None:
            await env.bg._keyword_queue.drain("kc-agent")

    env.run(again())
    assert len(env.passes) == 1


def test_cooldown_blocks_a_second_pass_even_when_sources_changed(
    tmp_path: Path, monkeypatch, _env
) -> None:
    env = _env(tmp_path, monkeypatch, curate=True)
    monkeypatch.setattr(
        kc,
        "curate_settings",
        lambda _a: {"curate": True, "curate_limit": 3, "curate_cooldown_s": 10_000.0},
    )
    env.bg._keyword_settings_cache = None
    env.trigger()
    f = env.skills / "alpha" / "SKILL.md"
    f.write_text(f.read_text() + "\nedited so the size changes\n")
    env.trigger()
    assert len(env.passes) == 1


def test_a_scan_failure_never_breaks_the_turn(
    tmp_path: Path, monkeypatch, _env
) -> None:
    env = _env(tmp_path, monkeypatch, curate=True)

    def boom(_a):
        raise RuntimeError("no spec")

    monkeypatch.setattr(kc, "sources_config_for", boom)
    env.trigger()  # must not raise
    assert env.passes == [] and env.tasks() == []


def test_single_flight_blocks_while_running_but_not_after_a_pass_finishes(
    tmp_path: Path, monkeypatch, _env
) -> None:
    env = _env(tmp_path, monkeypatch, curate=True)
    skill = env.skills / "alpha" / "SKILL.md"
    orig = env.bg._keyword_pass

    async def scenario():
        gate, started = asyncio.Event(), asyncio.Event()

        async def slow(directory, sources, *, limit, seeds):
            started.set()
            await gate.wait()
            return await orig(directory, sources, limit=limit, seeds=seeds)

        env.bg._keyword_pass = slow
        env.bg._keyword_last_check = float("-inf")
        await env.bg.after_turn("c")  # enqueues the first pass
        await started.wait()
        skill.write_text(skill.read_text() + "\nedit while running\n")
        env.bg._keyword_last_check = float("-inf")
        await env.bg.after_turn("c")  # a pass is running -> skipped
        assert len(env.tasks()) == 1
        gate.set()
        await env.bg._keyword_queue.drain("kc-agent")
        # the worker is still alive (idle); that must NOT block the next pass
        env.bg._keyword_pass = orig
        env.bg._keyword_queue.pass_fn = orig
        env.bg._keyword_last_check = float("-inf")
        await env.bg.after_turn("c")
        await env.bg._keyword_queue.drain("kc-agent")

    env.run(scenario())
    assert len(env.passes) == 2 and len(env.tasks()) == 2


def test_registered_without_an_idle_row_so_the_panel_stays_clean(
    tmp_path: Path, monkeypatch, _env
) -> None:
    env = _env(tmp_path, monkeypatch, curate=True)
    [proc] = [p for p in env.bg._processes if p.id_namespace == "keyword_curate"]
    assert proc.idle_row is None and proc.label == "Curating keywords"
    # and the degraded status summary does not invent a row for it either
    rows = asyncio.run(env.bg.job_status_summary("cid"))
    assert not [r for r in rows if r.id.startswith("keyword_curate")]


# --- ordering and the cross-process lock -------------------------------------


def _item(ref: str, mtime: float = 1.0):
    from paf.capabilities.keyword_curator import SourceItem

    kind = ref.split(":", 1)[0]
    return SourceItem(ref, kind, ref.split(":", 1)[1], "text", (mtime, 10))


def test_prioritize_puts_skills_first_then_notes_newest_first() -> None:
    from paf.capabilities.keyword_curator import prioritize

    sources = {
        "note:Govt/old.md": _item("note:Govt/old.md", mtime=100.0),
        "note:Zed/newest.md": _item("note:Zed/newest.md", mtime=900.0),
        "skill:b-skill": _item("skill:b-skill"),
        "note:Mid/mid.md": _item("note:Mid/mid.md", mtime=500.0),
        "skill:a-skill": _item("skill:a-skill"),
    }
    assert prioritize(list(sources), sources) == [
        "skill:a-skill",
        "skill:b-skill",
        "note:Zed/newest.md",
        "note:Mid/mid.md",
        "note:Govt/old.md",
    ]
    # ties on mtime fall back to the ref, so the order is deterministic
    tie = {f"note:{n}.md": _item(f"note:{n}.md", mtime=5.0) for n in "cab"}
    assert prioritize(tie, tie) == ["note:a.md", "note:b.md", "note:c.md"]


def test_a_pass_works_the_backlog_in_priority_order_and_limit_takes_the_front(
    tmp_path: Path,
) -> None:
    seen: list[str] = []

    async def curate(batch, known):
        seen.extend(it.ref for it in batch)
        return await _fake_curate(batch, known)

    sources = {
        "note:A/old.md": _item("note:A/old.md", mtime=1.0),
        "note:B/new.md": _item("note:B/new.md", mtime=9.0),
        "skill:s": _item("skill:s"),
    }
    res = asyncio.run(
        run_keyword_pass(tmp_path / "kw", sources, curate=curate, limit=2)
    )
    assert seen == ["skill:s", "note:B/new.md"]  # skill, then the newest note
    assert res.keyed == ["skill:s", "note:B/new.md"]
    # the next pass picks up what is left
    res2 = asyncio.run(run_keyword_pass(tmp_path / "kw", sources, curate=curate))
    assert res2.keyed == ["note:A/old.md"]


def test_pass_lock_is_exclusive_and_released(tmp_path: Path) -> None:
    from paf.capabilities.keyword_curator import pass_lock

    with pass_lock(tmp_path) as first:
        assert first is True
        with pass_lock(tmp_path) as second:  # a second holder (even in-process)
            assert second is False
    with pass_lock(tmp_path) as again:  # released on exit
        assert again is True


def test_pass_lock_is_released_when_the_holder_raises(tmp_path: Path) -> None:
    from paf.capabilities.keyword_curator import pass_lock

    with pytest.raises(RuntimeError), pass_lock(tmp_path):
        raise RuntimeError("pass blew up")
    with pass_lock(tmp_path) as ok:
        assert ok is True


def test_a_pass_is_skipped_untouched_while_another_process_holds_the_lock(
    tmp_path: Path,
) -> None:
    from paf.capabilities.keyword_curator import pass_lock

    store = tmp_path / "kw"
    calls: list = []

    async def curate(batch, known):
        calls.append(batch)
        return await _fake_curate(batch, known)

    sources = {"skill:s": _item("skill:s")}
    with pass_lock(store) as held:
        assert held
        res = asyncio.run(run_keyword_pass(store, sources, curate=curate))
    assert res.skipped and res.keyed == [] and res.ok
    assert calls == []  # nothing was curated ...
    assert not (store / "keyword-journal.jsonl").exists()  # ... or written
    assert not (store / "keyword-watermark.json").exists()
    # once the lock is free the same pass runs normally
    res2 = asyncio.run(run_keyword_pass(store, sources, curate=curate))
    assert res2.keyed == ["skill:s"] and not res2.skipped


def test_two_real_processes_cannot_both_hold_the_pass_lock(tmp_path: Path) -> None:
    """flock across a genuine process boundary, not just two fds in one process."""
    import subprocess
    import sys
    import textwrap

    from paf.capabilities.keyword_curator import pass_lock

    code = textwrap.dedent(
        f"""
        from paf.capabilities.keyword_curator import pass_lock
        with pass_lock({str(tmp_path)!r}) as got:
            print("GOT" if got else "BLOCKED")
        """
    )
    with pass_lock(tmp_path) as held:
        assert held
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        assert out == "BLOCKED"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "GOT"


def test_queue_reports_a_skipped_pass_as_a_clean_completion(tmp_path: Path) -> None:
    from paf.capabilities.keyword_curator._queue import summarize_pass

    assert summarize_pass(
        PassResult(skipped="another process is curating this store")
    ) == ("skipped — another process is curating this store")
