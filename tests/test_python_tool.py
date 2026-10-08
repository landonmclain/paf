"""Tests for the sandboxed ``run_python`` capability (paf.capabilities.python).

Unit tests drive the toolset with a fake executor; the integration tests at the
bottom run real programs in Docker and are skipped when it is unavailable.
"""

from __future__ import annotations

import asyncio
import base64
import re
import shlex

import pytest
from pydantic_ai.exceptions import ModelRetry

from paf.capabilities._sandbox import DockerExecutor, ExecResult, LocalProcessExecutor
from paf.capabilities._sandbox._docker import docker_available
from paf.capabilities.build import _build_python
from paf.capabilities.python import Python, PythonToolset
from paf.specs import _spec_python_config


def _run(coro):
    return asyncio.run(coro)


class FakeExecutor:
    """Records the command and returns a canned result."""

    def __init__(self, result: ExecResult | None = None) -> None:
        self.result = result or ExecResult(stdout="", stderr="", exit_code=0)
        self.calls: list[dict] = []

    async def run(self, command, *, cwd, env, timeout):
        self.calls.append(
            {"command": command, "cwd": cwd, "env": env, "timeout": timeout}
        )
        return self.result


def _toolset(tmp_path, result=None, **kwargs):
    ex = FakeExecutor(result)
    return PythonToolset(cwd=tmp_path, executor=ex, **kwargs), ex


# --- command construction ------------------------------------------------------
def test_command_ships_code_as_base64_argv_under_timeout(tmp_path):
    ts, ex = _toolset(tmp_path)
    code = "print('héllo \"quoted\" $HOME `x`')\n"
    _run(ts.run_python(code))
    command = ex.calls[0]["command"]
    assert command.startswith("timeout -k 2 30 python -I -c ")
    payload = shlex.split(command)[-1]
    assert base64.b64decode(payload).decode() == code
    assert ex.calls[0]["timeout"] == 40.0  # backstop = timeout + 10


def test_timeout_is_clamped_to_max(tmp_path):
    ts, ex = _toolset(tmp_path, max_timeout=60.0)
    _run(ts.run_python("pass", timeout_seconds=9999))
    assert " 60 python" in ex.calls[0]["command"]
    _run(ts.run_python("pass", timeout_seconds=0.1))
    assert "timeout -k 2 1 python" in ex.calls[1]["command"]


def test_empty_and_oversized_code_are_model_retries(tmp_path):
    ts, _ = _toolset(tmp_path, max_code_chars=10)
    with pytest.raises(ModelRetry, match="empty"):
        _run(ts.run_python("   "))
    with pytest.raises(ModelRetry, match="limit is 10"):
        _run(ts.run_python("x" * 11))


# --- result formatting ---------------------------------------------------------
def test_success_output_is_labelled(tmp_path):
    ts, _ = _toolset(tmp_path, ExecResult("42\n", "warn\n", 0))
    out = _run(ts.run_python("print(42)"))
    assert out == "[stdout]\n42\n\n[stderr]\nwarn\n"


def test_no_output(tmp_path):
    ts, _ = _toolset(tmp_path)
    assert _run(ts.run_python("pass")) == "(no output)"


def test_nonzero_exit_reports_code(tmp_path):
    ts, _ = _toolset(tmp_path, ExecResult("", "Traceback...\n", 1))
    out = _run(ts.run_python("1/0"))
    assert out.endswith("[exit code: 1]") and "Traceback" in out


def test_timeout_exit_124_and_executor_backstop_are_reported(tmp_path):
    for res in (
        ExecResult("partial\n", "", 124),
        ExecResult("", "", 124, timed_out=True),
    ):
        ts, _ = _toolset(tmp_path, res)
        assert "[timed out after 30s]" in _run(ts.run_python("while True: pass"))


def test_exit_137_is_reported_as_killed(tmp_path):
    ts, _ = _toolset(tmp_path, ExecResult("", "", 137))
    assert "killed" in _run(ts.run_python("x = bytearray(10**12)"))


def test_output_is_truncated_keeping_the_tail(tmp_path):
    ts, _ = _toolset(
        tmp_path, ExecResult("a" * 500 + "END", "", 0), max_output_chars=100
    )
    out = _run(ts.run_python("print()"))
    assert len(out) <= 100 and out.endswith("END") and "truncated" in out


# --- capability: refusing the unsandboxed backend ------------------------------
def test_capability_requires_an_executor(tmp_path):
    with pytest.raises(ValueError, match="needs an executor"):
        Python(cwd=tmp_path)


def test_capability_refuses_local_backend_by_default(tmp_path):
    with pytest.raises(ValueError, match="refuses the local"):
        Python(cwd=tmp_path, executor=LocalProcessExecutor())


def test_capability_allows_local_backend_only_when_opted_in(tmp_path):
    cap = Python(cwd=tmp_path, executor=LocalProcessExecutor(), allow_unsandboxed=True)
    assert isinstance(cap.get_toolset(), PythonToolset)


# --- builder / spec ------------------------------------------------------------
def test_builder_refuses_local_backend_from_spec():
    with pytest.raises(ValueError, match="refuses the local"):
        _build_python(None, {"executor": {"backend": "local"}})


def test_builder_local_with_explicit_opt_in():
    cap = _build_python(
        None, {"executor": {"backend": "local"}, "allow_unsandboxed": True}
    )
    assert isinstance(cap.executor, LocalProcessExecutor)


def test_builder_defaults_to_strict_docker(monkeypatch):
    monkeypatch.setattr("paf.capabilities.build.docker_available", lambda: True)
    cap = _build_python(None, None)
    ex = cap.executor
    assert isinstance(ex, DockerExecutor)
    assert ex._mode == "run" and ex._workspace_readonly and ex._network == "none"
    assert ex._tmpfs == ["/tmp:rw,size=64m,noexec,nosuid"]


def test_builder_hard_fails_without_docker(monkeypatch):
    monkeypatch.setattr("paf.capabilities.build.docker_available", lambda: False)
    with pytest.raises(RuntimeError, match="Docker is not available"):
        _build_python(None, None)


def test_builder_spec_overrides_merge_onto_defaults(monkeypatch):
    monkeypatch.setattr("paf.capabilities.build.docker_available", lambda: True)
    cap = _build_python(
        None,
        {"default_timeout": 5, "executor": {"memory": "256m", "writable": ["out"]}},
    )
    ex = cap.executor
    assert ex._memory == "256m" and ex._writable == ["out"] and ex._mode == "run"
    assert cap.default_timeout == 5.0


def test_spec_python_config_reads_typed_keys(tmp_path):
    spec = tmp_path / "a.yaml"
    spec.write_text(
        "python:\n  cwd: ~/w\n  allow_unsandboxed: false\n  default_timeout: 10\n"
        "  bogus: 1\n  executor:\n    memory: 512m\n    pids_limit: 32\n"
        "    workspace_readonly: true\n    writable: [out, '']\n"
    )
    assert _spec_python_config(spec) == {
        "cwd": "~/w",
        "allow_unsandboxed": False,
        "default_timeout": 10,
        "executor": {
            "memory": "512m",
            "pids_limit": 32,
            "workspace_readonly": True,
            "writable": ["out"],
        },
    }
    assert _spec_python_config(tmp_path / "missing.yaml") is None


def test_writable_paths_must_stay_inside_the_workspace():
    for bad in ("/etc", "../x", "a/../../x", ""):
        with pytest.raises(ValueError, match="workspace-relative"):
            DockerExecutor(writable=[bad])


# --- Docker integration --------------------------------------------------------
needs_docker = pytest.mark.skipif(not docker_available(), reason="Docker not available")


def _docker_python(tmp_path, **executor_kwargs):
    kwargs = {
        "mode": "run",
        "workspace_readonly": True,
        "tmpfs": ["/tmp:rw,size=8m,noexec,nosuid"],
        **executor_kwargs,
    }
    return PythonToolset(cwd=tmp_path, executor=DockerExecutor(**kwargs))


@needs_docker
def test_docker_runs_a_program_and_reads_the_workspace(tmp_path):
    (tmp_path / "data.txt").write_text("hello")
    ts = _docker_python(tmp_path)
    out = _run(ts.run_python("print(open('data.txt').read(), 6 * 7)"))
    assert "hello 42" in out


@needs_docker
def test_docker_traceback_shows_user_frames_only(tmp_path):
    out = _run(_docker_python(tmp_path).run_python("def f():\n    1/0\nf()\n"))
    assert "ZeroDivisionError" in out and "[exit code: 1]" in out
    assert "line 2, in f" in out and "1/0" in out
    assert "base64" not in out and "_BOOTSTRAP" not in out


@needs_docker
def test_docker_blocks_network(tmp_path):
    code = (
        "import socket\n"
        "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    print('CONNECTED')\n"
        "except OSError as e:\n    print('blocked', type(e).__name__)\n"
    )
    out = _run(_docker_python(tmp_path).run_python(code))
    assert "blocked" in out and "CONNECTED" not in out


@needs_docker
def test_docker_workspace_is_read_only_but_tmp_is_scratch(tmp_path):
    code = (
        "import os\n"
        "try:\n    open('probe.txt','w').write('x')\n    print('WROTE_WORKSPACE')\n"
        "except OSError:\n    print('workspace-ro')\n"
        "open('/tmp/scratch.txt','w').write('x'); print('tmp-ok')\n"
    )
    out = _run(_docker_python(tmp_path).run_python(code))
    assert "workspace-ro" in out and "tmp-ok" in out and "WROTE_WORKSPACE" not in out
    assert not (tmp_path / "probe.txt").exists()


@needs_docker
def test_docker_writable_subpath_is_the_only_writable_workspace_path(tmp_path):
    code = (
        "open('out/result.txt','w').write('done')\n"
        "try:\n    open('elsewhere.txt','w').write('x'); print('WROTE')\n"
        "except OSError:\n    print('ro-elsewhere')\n"
    )
    out = _run(_docker_python(tmp_path, writable=["out"]).run_python(code))
    assert "ro-elsewhere" in out and "WROTE" not in out
    assert (tmp_path / "out" / "result.txt").read_text() == "done"


@needs_docker
def test_docker_env_files_are_shadowed(tmp_path):
    (tmp_path / ".env").write_text("SECRET=hunter2")
    ts = _docker_python(tmp_path, exclude=[".env"])
    out = _run(ts.run_python("print(repr(open('.env').read()))"))
    assert "hunter2" not in out and "''" in out


@needs_docker
def test_docker_infinite_loop_times_out_and_leaves_no_container(tmp_path):
    ts = _docker_python(tmp_path)
    out = _run(ts.run_python("while True: pass", timeout_seconds=2))
    assert "timed out" in out
    import subprocess

    names = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
    ).stdout
    assert not re.search(r"paf-run-", names)


@needs_docker
def test_docker_each_call_is_a_fresh_interpreter(tmp_path):
    ts = _docker_python(tmp_path)
    _run(ts.run_python("x = 1\nopen('/tmp/leak','w').write('1')"))
    out = _run(
        ts.run_python(
            "import os\nprint('leak' in os.listdir('/tmp'))\nprint('x' in dir())"
        )
    )
    assert out.count("False") == 2


@needs_docker
def test_docker_runs_as_non_root_without_host_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PAF_TEST_SECRET_VALUE", "do-not-leak")
    out = _run(
        _docker_python(tmp_path).run_python(
            "import os\nprint(os.getuid(), os.environ.get('PAF_TEST_SECRET_VALUE'))"
        )
    )
    assert "1000 None" in out
