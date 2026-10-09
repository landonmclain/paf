"""A cross-process lock around a curation pass.

The keyword store is written by read-then-rewrite (the journal append and the
watermark save each load the whole file and replace it), guarded in-process by a
thread lock. That is not enough here: the SSE server, the chat CLI, and a manual
``python -m`` run are separate processes sharing one store, and two concurrent
passes could drop each other's journal lines or overwrite each other's watermark.

``flock`` is released by the OS when the holder exits or crashes, so a dead
process can never leave the store locked. The lock is advisory and non-blocking:
a pass that cannot get it is skipped, not queued — the holder is already doing
the same work.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_FILENAME = ".curator.lock"


@contextmanager
def pass_lock(directory: str | os.PathLike) -> Iterator[bool]:
    """Hold the store's pass lock; yields ``False`` if another holder has it."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    fd = os.open(path / LOCK_FILENAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
