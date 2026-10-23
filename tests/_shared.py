"""Build expensive test fixtures once per run and share them across xdist workers.

Each pytest-xdist worker is its own process, so a session-scoped fixture alone is
rebuilt by every worker that needs it. With ``-n auto`` that means dozens of
identical scans competing for CPU. These helpers build an artifact once, under a
file lock in the run's shared temp directory, and hand every worker the same copy.
Without xdist they simply build.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path

import pytest
from filelock import FileLock


def shared_build[T](
    config: pytest.Config,
    tmp_path_factory: pytest.TempPathFactory,
    name: str,
    build: Callable[[], T],
) -> T:
    """Build ``name`` once per test run and share it across xdist workers.

    The first worker to get here builds the value under a file lock and pickles
    it into the run's shared temp directory; the rest load that copy. The value
    must pickle.
    """
    if not hasattr(config, "workerinput"):
        return build()
    shared = tmp_path_factory.getbasetemp().parent  # common to every worker of this run
    path = shared / f"{name}.pickle"
    with FileLock(str(path) + ".lock"):
        if path.is_file():
            return pickle.loads(path.read_bytes())
        value = build()
        path.write_bytes(pickle.dumps(value))
        return value


def shared_dir(
    config: pytest.Config,
    tmp_path_factory: pytest.TempPathFactory,
    name: str,
    build: Callable[[Path], None],
) -> Path:
    """Like :func:`shared_build`, for an artifact that is a directory on disk.

    ``build(path)`` populates an empty directory once per run; every worker gets
    the same path back. Callers must only read it.
    """
    if not hasattr(config, "workerinput"):
        path = tmp_path_factory.mktemp(name)
        build(path)
        return path
    root = tmp_path_factory.getbasetemp().parent / name
    done = root / ".built"
    with FileLock(str(root) + ".lock"):
        if not done.is_file():
            root.mkdir(parents=True, exist_ok=True)
            build(root)
            done.touch()
    return root
