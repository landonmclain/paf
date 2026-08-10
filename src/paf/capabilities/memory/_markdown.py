"""MarkdownStorage: a ``MemoryStore`` whose bodies are real `.md` files.

Companion to :class:`~paf.capabilities.memory._store.InMemoryStore` and
:class:`~paf.capabilities.memory._store.FileStore`. Where ``FileStore`` keeps
its version/CAS metadata in a hidden SQLite journal, ``MarkdownStorage`` keeps
**everything flat and diffable**: the memory bodies are plain Markdown files and
the version/CAS bookkeeping lives in a ``.memory-state.json`` sidecar per scope
directory. Both are written atomically via ``agent_foundation``'s
``write_atomic`` helper.

Why it does *not* route through the ``StorageService`` *protocol*: that seam
models the **derived/index** shape (append-only JSONL logs + ``BlockRef`` byte
pointers — see ``TurnIndexStore``, its one true consumer). Memory is the
**authoritative/document** shape: mutable, CAS-versioned, edited in place by
``write_memory``/``delete_memory``, and it must stay readable ``.md`` on disk.
The seam has no expressible "compare-and-swap update of a named document", so
forcing memory through it mainly buries or bends the model. ``MarkdownStorage``
stays a plain flat-file backend and leans on the foundation's *atomic helper*
layer (``write_atomic`` / ``read_json``) — that is the part worth sharing.

Flat-file tradeoff vs ``FileStore``: operation-id idempotency receipts live in
memory for the process lifetime (like ``InMemoryStore``) rather than in a
durable journal, so a *model retry within a turn* is replayed exactly, but a
crash + external re-execution after restart is instead bounded by the CAS
version check (which *is* durable via the sidecar). That is the correct safety
line for model-written memory: retries are in-process; conflicts are caught.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio.to_thread
from agent_foundation.storage.atomic import read_json, write_atomic, write_json_atomic

from ._store import (
    MemoryConflictError,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemorySearchResult,
    SearchableMemoryStore,
    _check_operation,
    lexical_search,
    validate_store_path,
    validate_store_prefix,
)

_STATE_FILENAME = ".memory-state.json"
_MD_SUFFIX = ".md"


@dataclass(frozen=True)
class _StateEntry:
    """Version + content-fingerprint for one memory path."""

    version: int
    fingerprint: str
    last_operation_id: str | None = None


def _run_sync(sync_fn: Callable[[], object]) -> Any:
    """Run a blocking file call off the event loop (await the returned coroutine)."""
    return anyio.to_thread.run_sync(sync_fn)


class MarkdownStorage(SearchableMemoryStore):
    """Plain-Markdown memory store; bodies + a flat version sidecar.

    Layout under ``directory``::

        <directory>/
          <scope>/MEMORY.md          # readable markdown body
          <scope>/topic.md           # ...
          <scope>/.memory-state.json # version/CAS sidecar for that scope

    Bodies are real `.md` files (inspectable / editable / diffable — the M2
    flat-file principle). ``.memory-state.json`` records ``path -> {version,
    fingerprint, last_operation_id}`` so a concurrent or external edit to a body
    is detected on the next CAS and raises :class:`MemoryConflictError`, exactly
    like ``FileStore``. A per-scope sidecar keeps reads cheap (only the scope's
    entries are loaded).
    """

    def __init__(self, directory: str | Path = ".agent-memory") -> None:
        self._root = Path(directory)
        self._lock = threading.RLock()
        # In-process idempotency receipts: bounded by the process lifetime.
        self._receipts: dict[str, tuple[str, MemoryMutation]] = {}

    def _resolve(self, path: str) -> Path:
        validate_store_path(path)
        real_root = Path(os.path.realpath(self._root))
        resolved = Path(os.path.realpath(real_root / path))
        if not resolved.is_relative_to(real_root):
            raise ValueError(f"memory path {path!r} resolves outside the store directory")
        return resolved

    # --- state sidecar ----------------------------------------------------

    def _state_path(self, path: str) -> Path:
        return self._resolve(path).parent / _STATE_FILENAME

    def _read_state(self, scope_dir: Path) -> dict[str, dict]:
        data = read_json(scope_dir / _STATE_FILENAME, default={})
        return data if isinstance(data, dict) else {}

    def _load_state(self, path: str) -> dict[str, dict]:
        return self._read_state(self._resolve(path).parent)

    def _save_state(self, scope_dir: Path, state: dict[str, dict]) -> None:
        write_json_atomic(scope_dir / _STATE_FILENAME, state)

    # --- fingerprinting ----------------------------------------------------

    def _fingerprint(self, target: Path) -> str:
        stat = target.stat()
        return f"{stat.st_dev}:{stat.st_ino}:{stat.st_mtime_ns}:{stat.st_size}"

    # --- reads -------------------------------------------------------------

    def _current_path(self, path: str, max_chars: int) -> MemoryFile | None:
        """Read the body + version for ``path`` (caller holds ``self._lock``).

        Mirrors ``FileStore._current``: if the body's fingerprint no longer
        matches the recorded state (externally edited, or never recorded), the
        file is re-stamped with a fresh version. A caller holding an older
        ``expected_version`` then hits a ``MemoryConflictError`` — external
        edits are never silently overwritten.
        """
        target = self._resolve(path)
        if not target.is_file():
            return None
        size = target.stat().st_size
        truncated = max_chars > 0 and size > max_chars
        with target.open("r", encoding="utf-8") as fh:
            content = fh.read(max_chars) if max_chars > 0 else fh.read(size)
        state = self._load_state(path)
        entry = state.get(path)
        recorded_ok = (
            isinstance(entry, dict)
            and isinstance(entry.get("version"), int)
            and str(entry.get("fingerprint")) == self._fingerprint(target)
        )
        if not recorded_ok:
            # No state, or external edit — re-stamp with a fresh version so the
            # next CAS against an old expected_version conflicts.
            version = self._next_version(path)
            op_id = entry.get("last_operation_id") if isinstance(entry, dict) else None
            state[path] = {
                "version": version,
                "fingerprint": self._fingerprint(target),
                "last_operation_id": op_id if isinstance(op_id, str) else None,
            }
            self._save_state(target.parent, state)
            return MemoryFile(
                content=content,
                version=str(version),
                operation_id=op_id if isinstance(op_id, str) else None,
                truncated=truncated,
            )
        assert isinstance(entry, dict)
        version = int(entry["version"])
        op_id = entry.get("last_operation_id")
        return MemoryFile(
            content=content,
            version=str(version),
            operation_id=op_id if isinstance(op_id, str) else None,
            truncated=truncated,
        )

    def _next_version(self, path: str) -> int:
        state = self._load_state(path)
        versions = [
            int(e["version"])
            for e in state.values()
            if isinstance(e, dict) and isinstance(e.get("version"), int)
        ]
        return (max(versions) + 1) if versions else 1

    # --- MemoryStore protocol ----------------------------------------------

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        validate_store_path(path)
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        with self._lock:
            return self._current_path(path, max_chars)

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        with self._lock:
            return _check_operation(self._receipts, operation)

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        validate_store_path(path)
        with self._lock:
            if operation is not None and (receipt := _check_operation(self._receipts, operation)) is not None:
                return receipt
            current = self._current_path(path, 0)
            if (current.version if current else None) != expected_version:
                raise MemoryConflictError(f"memory path {path!r} changed before it could be written")
            existed = current is not None
            target = self._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_atomic(target, content)
            state = self._load_state(path)
            version = self._next_version(path)
            state[path] = {
                "version": version,
                "fingerprint": self._fingerprint(target),
                "last_operation_id": operation.id if operation else None,
            }
            self._save_state(target.parent, state)
            mutation = MemoryMutation(version=str(version), replayed=False, existed=existed)
            if operation is not None:
                self._receipts[operation.id] = (operation.fingerprint, mutation)
            return mutation

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        validate_store_path(path)
        with self._lock:
            if operation is not None and (receipt := _check_operation(self._receipts, operation)) is not None:
                return receipt
            current = self._current_path(path, 0)
            if (current.version if current else None) != expected_version:
                raise MemoryConflictError(f"memory path {path!r} changed before it could be deleted")
            existed = current is not None
            target = self._resolve(path)
            target.unlink(missing_ok=True)
            state = self._load_state(path)
            state.pop(path, None)
            self._save_state(target.parent, state)
            mutation = MemoryMutation(version=None, replayed=False, existed=existed)
            if operation is not None:
                self._receipts[operation.id] = (operation.fingerprint, mutation)
            return mutation

    async def list_paths(self, prefix: str = "", *, limit: int) -> list[str]:
        validate_store_prefix(prefix)
        if limit <= 0:
            raise ValueError("limit must be positive")
        return await _run_sync(lambda: self._list_paths_sync(prefix, limit))

    def _list_paths_sync(self, prefix: str, limit: int) -> list[str]:
        real_root = Path(os.path.realpath(self._root))

        def walk(directory: Path) -> Iterable[str]:
            if not directory.is_dir():
                return
            for entry in os.scandir(directory):
                if entry.is_dir(follow_symlinks=False):
                    yield from walk(Path(entry.path))
                elif (
                    entry.is_file(follow_symlinks=False)
                    and not entry.name.startswith(".")
                    and entry.name.endswith(_MD_SUFFIX)
                ):
                    relative = Path(entry.path).relative_to(real_root).as_posix()
                    if relative.startswith(prefix):
                        yield relative

        found = sorted(walk(real_root))
        return found[:limit]

    async def search(
        self,
        prefix: str,
        query: str,
        *,
        limit: int,
        max_files: int,
        max_chars: int,
        max_file_chars: int,
    ) -> MemorySearchResult:
        if not query.split() or limit <= 0 or max_files <= 0 or max_chars <= 0 or max_file_chars <= 0:
            return MemorySearchResult(matches=[], scanned=0, truncated=False)

        def run() -> MemorySearchResult:
            paths = self._list_paths_sync(prefix, max_files + 1)
            paths_truncated = len(paths) > max_files
            files: list[tuple[str, str]] = []
            truncated = paths_truncated
            for path in paths[:max_files]:
                target = self._resolve(path)
                try:
                    with target.open("r", encoding="utf-8") as fh:
                        content = fh.read(max_file_chars + 1)
                except FileNotFoundError:
                    truncated = True
                    continue
                truncated = truncated or len(content) > max_file_chars
                files.append((path, content[:max_file_chars]))
            result = lexical_search(
                files, query, limit=limit, max_files=max_files, max_chars=max_chars, score_prefix=prefix
            )
            return MemorySearchResult(
                matches=result.matches,
                scanned=result.scanned,
                truncated=truncated or result.truncated,
            )

        return await _run_sync(run)
