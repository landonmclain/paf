"""Tests for the spec-file ``shell.executor`` resolution in build.py.

Covers ``_resolve_shell_executor`` (local default, docker config, hard-fail on
unavailable) and that ``_build_shell`` passes the resolved executor through to
the Shell capability.
"""

import pytest

from paf.capabilities._sandbox import DockerExecutor, LocalProcessExecutor
from paf.capabilities._sandbox._docker import docker_available
from paf.capabilities.build import _build_shell, _resolve_shell_executor


def test_resolve_none_is_local():
    assert isinstance(_resolve_shell_executor(None), LocalProcessExecutor)


def test_resolve_empty_dict_is_local():
    assert isinstance(_resolve_shell_executor({}), LocalProcessExecutor)


def test_resolve_explicit_local():
    assert isinstance(
        _resolve_shell_executor({"backend": "local"}), LocalProcessExecutor
    )


def test_resolve_unknown_backend_raises():
    with pytest.raises(ValueError, match="backend must be 'local' or 'docker'"):
        _resolve_shell_executor({"backend": "bogus"})


def test_resolve_docker_config():
    if not docker_available():
        with pytest.raises(RuntimeError, match="Docker is not available"):
            _resolve_shell_executor({"backend": "docker"})
        return
    ex = _resolve_shell_executor({"backend": "docker"})
    assert isinstance(ex, DockerExecutor)
    assert ex._mode == "exec"
    assert ex._image == "python:3.12-slim"


def test_resolve_docker_custom_mode_image():
    if not docker_available():
        pytest.skip("Docker not available")
    ex = _resolve_shell_executor(
        {"backend": "docker", "mode": "run", "image": "python:3.11-slim"}
    )
    assert isinstance(ex, DockerExecutor)
    assert ex._mode == "run"
    assert ex._image == "python:3.11-slim"


def test_resolve_docker_full_option_pass_through():
    if not docker_available():
        pytest.skip("Docker not available")
    ex = _resolve_shell_executor(
        {
            "backend": "docker",
            "mode": "run",
            "image": "python:3.11-slim",
            "network": "bridge",
            "user": "1234:1234",
            "read_only": False,
            "memory": "2g",
            "cpus": "2.0",
            "pids_limit": 512,
            "workspace": "/app",
            "container_name": "paf-test",
        }
    )
    assert isinstance(ex, DockerExecutor)
    assert ex._mode == "run"
    assert ex._image == "python:3.11-slim"
    assert ex._network == "bridge"
    assert ex._user == "1234:1234"
    assert ex._read_only is False
    assert ex._memory == "2g"
    assert ex._cpus == "2.0"
    assert ex._pids_limit == 512
    assert ex._workspace == "/app"
    assert ex._container_name == "paf-test"


def test_resolve_docker_defaults_when_unspecified():
    if not docker_available():
        pytest.skip("Docker not available")
    ex = _resolve_shell_executor({"backend": "docker"})
    assert isinstance(ex, DockerExecutor)
    assert ex._network == "none"
    assert ex._user == "1000:1000"
    assert ex._read_only is True
    assert ex._memory == "1g"
    assert ex._cpus == "1.0"
    assert ex._pids_limit == 256
    assert ex._workspace == "/workspace"
    assert ex._container_name is None


def test_build_shell_no_config_executor_none():
    # No shell_config -> executor stays None; the toolset defaults to local.
    assert _build_shell(store=None).executor is None


def test_build_shell_explicit_local():
    shell = _build_shell(store=None, shell_config={"executor": {"backend": "local"}})
    assert isinstance(shell.executor, LocalProcessExecutor)


def test_build_shell_passes_docker_executor():
    if not docker_available():
        pytest.skip("Docker not available")
    shell = _build_shell(store=None, shell_config={"executor": {"backend": "docker"}})
    assert isinstance(shell.executor, DockerExecutor)
