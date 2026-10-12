"""The size cap and hygiene on host-visible writable paths (``out/``).

The writable paths are the one channel from sandboxed code to the host: host-side
tools read them afterwards. So after every call the executor removes anything
that is not a regular file or directory, and enforces a total-size cap both while
the program runs (the container is killed) and afterwards (this run's files are
deleted, largest first). Unit tests cover the logic on a temp dir; the
Docker-marked tests overflow the cap for real.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest

from paf.capabilities._sandbox import DockerExecutor, parse_size
from paf.capabilities._sandbox._docker import (
    docker_available,
    enforce_writable_policy,
    tree_bytes,
)
from paf.capabilities.build import _PYTHON_EXECUTOR_DEFAULTS, _build_python
from paf.capabilities.python import PythonToolset
from paf.specs import _spec_python_config


def _run(coro):
    return asyncio.run(coro)


# --- sizes ---------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("100m", 100 * 1024**2),
        ("100M", 100 * 1024**2),
        ("2g", 2 * 1024**3),
        ("512k", 512 * 1024),
        ("10mb", 10 * 1024**2),
        (" 64 m ", 64 * 1024**2),
        ("4096", 4096),
        (4096, 4096),
    ],
)
def test_parse_size(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("bad", ["", "m", "-5m", "0", 0, -1, "1.5m", "ten", True, None])
def test_parse_size_rejects_nonsense(bad):
    with pytest.raises((ValueError, TypeError)):
        parse_size(bad)


# --- cleanup logic (no Docker) ---------------------------------------------------
def _file(path: Path, size: int, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_tree_bytes_counts_regular_files_only(tmp_path):
    out = tmp_path / "out"
    _file(out / "a", 100)
    _file(out / "sub" / "b", 50)
    (out / "link").symlink_to(tmp_path)  # must not be followed
    assert tree_bytes([out]) == 150
    assert tree_bytes([tmp_path / "missing"]) == 0


def test_non_regular_entries_are_removed(tmp_path):
    out = tmp_path / "out"
    keep = _file(out / "keep.txt", 10)
    os.mkfifo(out / "pipe")
    (out / "link").symlink_to("/etc/passwd")
    (out / "dirlink").symlink_to(tmp_path)
    notes = enforce_writable_policy([out], max_bytes=None, started=0)
    assert keep.exists()
    assert not (out / "pipe").exists() and not os.path.lexists(out / "link")
    assert not os.path.lexists(out / "dirlink")
    assert len(notes) == 3 and all("non-regular" in n for n in notes)


def test_over_cap_deletes_this_runs_largest_files_first(tmp_path):
    out = tmp_path / "out"
    old = _file(out / "old.bin", 400, mtime=1_000)  # earlier call: protected
    big = _file(out / "big.bin", 600, mtime=5_000)
    small = _file(out / "small.bin", 100, mtime=5_000)
    notes = enforce_writable_policy([out], max_bytes=700, started=2_000)
    # 1100 total, cap 700: big.bin (largest fresh) goes; 500 left, under the cap.
    assert not big.exists() and old.exists() and small.exists()
    assert len(notes) == 1 and "big.bin" in notes[0]
    assert tree_bytes([out]) == 500


def test_old_files_are_never_deleted_for_being_over_the_cap(tmp_path):
    out = tmp_path / "out"
    old = _file(out / "old.bin", 900, mtime=1_000)
    notes = enforce_writable_policy([out], max_bytes=100, started=2_000)
    assert old.exists() and notes == []  # nothing from this run to delete


def test_under_cap_is_a_noop(tmp_path):
    out = tmp_path / "out"
    f = _file(out / "a", 10, mtime=5_000)
    assert enforce_writable_policy([out], max_bytes=1000, started=0) == []
    assert f.exists()


def test_missing_root_is_fine(tmp_path):
    assert enforce_writable_policy([tmp_path / "nope"], max_bytes=1, started=0) == []


# --- construction / wiring --------------------------------------------------------
def test_cap_requires_run_mode():
    with pytest.raises(ValueError, match="mode='run'"):
        DockerExecutor(mode="exec", max_writable_bytes=1024)
    with pytest.raises(ValueError, match="positive"):
        DockerExecutor(mode="run", max_writable_bytes=0)
    DockerExecutor(mode="run", max_writable_bytes=1024)


def test_python_builder_caps_out_by_default(monkeypatch):
    monkeypatch.setattr("paf.capabilities.build.docker_available", lambda: True)
    assert _PYTHON_EXECUTOR_DEFAULTS["max_writable_size"] == "100m"
    ex = _build_python(None, {"executor": {"writable": ["out"]}}).executor
    assert ex._max_writable_bytes == 100 * 1024**2  # noqa: SLF001
    ex = _build_python(
        None, {"executor": {"writable": ["out"], "max_writable_size": "5m"}}
    ).executor
    assert ex._max_writable_bytes == 5 * 1024**2  # noqa: SLF001


def test_spec_passes_the_cap_through(tmp_path):
    spec = tmp_path / "a.yaml"
    spec.write_text("python:\n  executor:\n    max_writable_size: 20m\n")
    assert _spec_python_config(spec)["executor"] == {"max_writable_size": "20m"}


def test_the_shipped_secure_spec_sets_a_cap():
    spec = Path(__file__).resolve().parent.parent / "specs" / "secure_assistant.yaml"
    assert _spec_python_config(spec)["executor"]["max_writable_size"] == "100m"


# --- Docker: overflow it for real ---------------------------------------------------
needs_docker = pytest.mark.skipif(not docker_available(), reason="Docker not available")


def _toolset(tmp_path, cap="3m"):
    (tmp_path / "out").mkdir(exist_ok=True)
    return PythonToolset(
        cwd=tmp_path,
        executor=DockerExecutor(
            mode="run",
            workspace_readonly=True,
            writable=["out"],
            tmpfs=["/tmp:rw,size=8m,noexec,nosuid"],
            max_writable_bytes=parse_size(cap),
        ),
    )


FLOOD = (
    "import os\n"
    "chunk = b'x' * 1024 * 1024\n"
    "with open('out/flood.bin', 'wb') as f:\n"
    "    while True:\n"
    "        f.write(chunk); f.flush(); os.fsync(f.fileno())\n"
)


@needs_docker
def test_docker_a_runaway_writer_is_killed_and_cleaned_up(tmp_path):
    ts = _toolset(tmp_path)
    started = time.monotonic()
    out = _run(ts.run_python(FLOOD, timeout_seconds=30))
    elapsed = time.monotonic() - started
    assert "size cap" in out and "killed" in out
    assert elapsed < 20, (
        f"should die on the cap, not wait out the timeout ({elapsed:.1f}s)"
    )
    assert tree_bytes([tmp_path / "out"]) <= parse_size("3m")
    assert not (tmp_path / "out" / "flood.bin").exists()  # this run's file removed
    names = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True
    ).stdout
    assert "paf-run-" not in names


@needs_docker
def test_docker_a_normal_write_under_the_cap_is_kept(tmp_path):
    ts = _toolset(tmp_path)
    out = _run(ts.run_python("open('out/r.txt', 'w').write('x' * 1000)"))
    assert "cap" not in out
    assert (tmp_path / "out" / "r.txt").stat().st_size == 1000


@needs_docker
def test_docker_overflow_deletes_only_this_runs_files(tmp_path):
    ts = _toolset(tmp_path)
    _run(ts.run_python("open('out/earlier.bin', 'wb').write(b'e' * 1024 * 1024)"))
    out = _run(ts.run_python(FLOOD, timeout_seconds=30))
    assert "size cap" in out
    assert (tmp_path / "out" / "earlier.bin").stat().st_size == 1024 * 1024
    assert not (tmp_path / "out" / "flood.bin").exists()


@needs_docker
def test_docker_fifos_and_symlinks_planted_in_out_are_removed(tmp_path):
    ts = _toolset(tmp_path)
    code = (
        "import os\n"
        "os.mkfifo('out/pipe')\n"
        "os.symlink('/etc/passwd', 'out/link')\n"
        "open('out/real.txt', 'w').write('ok')\n"
    )
    out = _run(ts.run_python(code))
    assert "non-regular" in out
    out_dir = tmp_path / "out"
    assert not os.path.lexists(out_dir / "pipe")
    assert not os.path.lexists(out_dir / "link")
    assert (out_dir / "real.txt").read_text() == "ok"
