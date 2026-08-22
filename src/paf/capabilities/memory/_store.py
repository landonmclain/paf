"""Shared typed memory records, protocols, and validation for the memory capability.

This module retains the load-bearing shared types the live M2 memory
spine needs (``_journal``, ``_organize``, ``_render``, ``_view``, the owner's
``JournalAppendToolset``):

- ``MemoryFile`` / ``MemoryOperation`` / ``MemoryMutation`` / ``MemorySearchMatch``
  / ``MemorySearchResult``  — shared record types
- ``MemoryStore`` / ``SearchableMemoryStore``  — protocols
- ``InMemoryStore``  — process-lifetime store; the ``Memory`` field default and
  the owner's inert store in the journal model (held but never persisted/read back)
- ``validate_store_path`` / ``validate_store_prefix`` / ``lexical_search`` —
  shared helpers

The heavy legacy rehomed backends grew out of here and were removed with the
rest of the rehomed pydantic-ai builtin surface: ``FileStore`` (flat ``.md`` +
hidden SQLite journal + a bespoke ``_atomic_write``) and ``MarkdownStorage``
(in the retired ``_markdown.py``). None is used by the M2 journal model — the
owner appends to ``JournalStore`` (JSONL), the organizer maintains
``IndexStore``, and ``JournalViewStore`` renders ``MEMORY.md``. Atomic writes
now come from the single shared ``_atomic.py`` (the port of ``agent-foundation``).
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import anyio

_VALID_SEGMENT_RE = re.compile(r"[A-Za-z0-9_.-]{1,200}")


@dataclass(frozen=True)
class MemoryFile:
    """One memory file and its opaque compare-and-set version."""

    content: str
    version: str
    operation_id: str | None
    truncated: bool


@dataclass(frozen=True)
class MemoryOperation:
    """Stable identity and argument fingerprint for an idempotent mutation."""

    id: str
    fingerprint: str


@dataclass(frozen=True)
class MemoryMutation:
    """Result of a memory write or delete."""

    version: str | None
    replayed: bool
    existed: bool


@dataclass(frozen=True)
class MemorySearchMatch:
    """One bounded lexical-search match."""

    path: str
    snippet: str
    score: float


@dataclass(frozen=True)
class MemorySearchResult:
    """Bounded search results and scan metadata."""

    matches: list[MemorySearchMatch]
    scanned: int
    truncated: bool


class MemoryConflictError(RuntimeError):
    """The stored version did not match the mutation's expected version."""


class MemoryOperationConflictError(RuntimeError):
    """An operation id was reused with a different fingerprint."""


def validate_store_path(path: str) -> None:
    r"""Reject path strings that could escape a store's root directory."""
    if not all(
        _VALID_SEGMENT_RE.fullmatch(segment) and ".." not in segment
        for segment in path.split("/")
    ):
        raise ValueError(f"invalid memory path: {path!r}")


def validate_store_prefix(prefix: str) -> None:
    if prefix:
        validate_store_path(prefix.removesuffix("/"))


def _snippet(content: str, query: str, max_chars: int) -> str:
    if max_chars <= 0:  # pragma: no cover - lexical_search rejects non-positive budgets
        return ""
    lower = content.lower()
    positions = [lower.find(term) for term in query.lower().split()]
    found = [position for position in positions if position >= 0]
    center = min(found) if found else 0
    start = max(0, center - max_chars // 3)
    end = min(len(content), start + max_chars)
    start = max(0, end - max_chars)
    snippet = content[start:end]
    if start:
        snippet = f"...{snippet[3:]}" if len(snippet) >= 3 else "." * len(snippet)
    if end < len(content):
        snippet = f"{snippet[:-3]}..." if len(snippet) >= 3 else "." * len(snippet)
    return snippet


def lexical_search(
    files: Iterable[tuple[str, str]],
    query: str,
    *,
    limit: int,
    max_files: int,
    max_chars: int,
    score_prefix: str = "",
) -> MemorySearchResult:
    """Search a sorted file stream with deterministic scan and output bounds."""
    terms = [term for term in query.lower().split() if term]
    if (
        not terms or limit <= 0 or max_files <= 0 or max_chars <= 0
    ):  # pragma: no cover - stores prevalidate
        return MemorySearchResult(matches=[], scanned=0, truncated=False)
    scored: list[tuple[float, str, str]] = []
    scanned = 0
    truncated = False
    for path, content in files:
        if scanned >= max_files:
            truncated = True
            break
        scanned += 1
        lower_path = path.removeprefix(score_prefix).lower()
        lower_content = content.lower()
        score = float(
            sum(
                lower_content.count(term) + 2 * lower_path.count(term) for term in terms
            )
        )
        if score:
            scored.append((score, path, content))
    scored.sort(key=lambda item: (-item[0], item[1]))
    matches: list[MemorySearchMatch] = []
    remaining = max_chars
    for score, path, content in scored[:limit]:
        visible_path = path.removeprefix(score_prefix)
        available = remaining - len(visible_path)
        if available <= 0:
            truncated = True
            break
        snippet = _snippet(content, query, available)
        matches.append(MemorySearchMatch(path=path, snippet=snippet, score=score))
        remaining -= len(visible_path) + len(snippet)
    if len(scored) > len(matches):
        truncated = True
    return MemorySearchResult(matches=matches, scanned=scanned, truncated=truncated)


@runtime_checkable
class MemoryStore(Protocol):
    """Async versioned storage for agent memories."""

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        """Return the file at `path`, or `None` if it does not exist."""
        ...  # pragma: no cover

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        """Return a prior result for `operation`, validating its fingerprint."""
        ...  # pragma: no cover

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        """Create or replace `path` if its version equals `expected_version`."""
        ...  # pragma: no cover

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        """Delete `path` if its version equals `expected_version`."""
        ...  # pragma: no cover

    async def list_paths(self, prefix: str = "", *, limit: int) -> list[str]:
        """Return all stored paths starting with `prefix`, sorted."""
        ...  # pragma: no cover


@runtime_checkable
class SearchableMemoryStore(Protocol):
    """Optional bounded search extension for a `MemoryStore`."""

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
        """Search only paths below the trusted, application-resolved `prefix`."""
        ...  # pragma: no cover


@dataclass(init=False)
class InMemoryStore:
    """Process-lifetime versioned memory store.

    The `Memory` capability's default backend and the owner's inert store in
    the M2 journal model (the owner appends to the journal and injects via
    ``JournalViewStore`` — this store is never persisted or read back). Kept as
    a small, dependency-free placeholder so ``Memory(...)`` can be constructed
    without a real backend; it is not part of the durable M2 pipeline.
    """

    _files: dict[str, str] = field(default_factory=dict[str, str], repr=False)
    _versions: dict[str, int] = field(
        default_factory=dict[str, int], init=False, repr=False
    )
    _operation_ids: dict[str, str | None] = field(
        default_factory=dict[str, str | None], init=False, repr=False
    )
    _receipts: dict[str, tuple[str, MemoryMutation]] = field(
        default_factory=dict[str, tuple[str, MemoryMutation]], init=False, repr=False
    )
    _generation: int = field(default=0, init=False, repr=False)
    _paths: list[str] = field(default_factory=list[str], init=False, repr=False)
    _lock: anyio.Lock = field(default_factory=anyio.Lock, init=False, repr=False)

    def __init__(self, files: Mapping[str, str] | None = None) -> None:
        self._files = dict(files or {})
        for path in self._files:
            validate_store_path(path)
        self._versions = {}
        self._operation_ids = {}
        self._receipts = {}
        self._generation = 0
        self._paths = sorted(self._files)
        self._lock = anyio.Lock()

    @property
    def files(self) -> Mapping[str, str]:
        """A read-only view of stored content; mutate through `write` and `delete`."""
        return MappingProxyType(self._files)

    def _next_generation(self) -> int:
        self._generation += 1
        return self._generation

    def _current(self, path: str, max_chars: int | None = None) -> MemoryFile | None:
        content = self._files.get(path)
        if content is None:
            return None
        version = self._versions.get(path)
        if version is None:
            version = self._next_generation()
            self._versions[path] = version
        truncated = max_chars is not None and len(content) > max_chars
        return MemoryFile(
            content=content[:max_chars] if max_chars is not None else content,
            version=str(version),
            operation_id=self._operation_ids.get(path),
            truncated=truncated,
        )

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        validate_store_path(path)
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        async with self._lock:
            return self._current(path, max_chars)

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        async with self._lock:
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
        async with self._lock:
            if (
                operation is not None
                and (receipt := _check_operation(self._receipts, operation)) is not None
            ):
                return receipt
            current = self._current(path)
            if (current.version if current else None) != expected_version:
                raise MemoryConflictError(
                    f"memory path {path!r} changed before it could be written"
                )
            version = self._next_generation()
            self._files[path] = content
            if current is None:
                bisect.insort(self._paths, path)
            self._versions[path] = version
            self._operation_ids[path] = operation.id if operation else None
            mutation = MemoryMutation(
                version=str(version), replayed=False, existed=current is not None
            )
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
        async with self._lock:
            if (
                operation is not None
                and (receipt := _check_operation(self._receipts, operation)) is not None
            ):
                return receipt
            current = self._current(path)
            if (current.version if current else None) != expected_version:
                raise MemoryConflictError(
                    f"memory path {path!r} changed before it could be deleted"
                )
            existed = current is not None
            self._files.pop(path, None)
            if current is not None:
                self._paths.remove(path)
            self._versions.pop(path, None)
            self._operation_ids.pop(path, None)
            self._next_generation()
            mutation = MemoryMutation(version=None, replayed=False, existed=existed)
            if operation is not None:
                self._receipts[operation.id] = (operation.fingerprint, mutation)
            return mutation

    async def list_paths(self, prefix: str = "", *, limit: int) -> list[str]:
        validate_store_prefix(prefix)
        if limit <= 0:
            raise ValueError("limit must be positive")
        async with self._lock:
            start = bisect.bisect_left(self._paths, prefix)
            paths: list[str] = []
            for index in range(start, len(self._paths)):
                path = self._paths[index]
                if not path.startswith(prefix):
                    break
                paths.append(path)
                if len(paths) == limit:
                    break
            return paths

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
        validate_store_prefix(prefix)
        if (
            not query.split()
            or limit <= 0
            or max_files <= 0
            or max_chars <= 0
            or max_file_chars <= 0
        ):
            return MemorySearchResult(matches=[], scanned=0, truncated=False)
        async with self._lock:
            start = bisect.bisect_left(self._paths, prefix)
            selected: list[str] = []
            for index in range(start, len(self._paths)):
                path = self._paths[index]
                if not path.startswith(prefix):
                    break
                selected.append(path)
                if len(selected) > max_files:
                    break
            scanned_paths = selected[:max_files]
            content_truncated = any(
                len(self._files[path]) > max_file_chars for path in scanned_paths
            )
            files = [
                (path, self._files[path][:max_file_chars]) for path in scanned_paths
            ]
        result = lexical_search(
            files,
            query,
            limit=limit,
            max_files=max_files,
            max_chars=max_chars,
            score_prefix=prefix,
        )
        return MemorySearchResult(
            matches=result.matches,
            scanned=result.scanned,
            truncated=result.truncated
            or content_truncated
            or len(selected) > max_files,
        )


def _check_operation(
    receipts: dict[str, tuple[str, MemoryMutation]], operation: MemoryOperation
) -> MemoryMutation | None:
    receipt = receipts.get(operation.id)
    if receipt is None:
        return None
    fingerprint, mutation = receipt
    if fingerprint != operation.fingerprint:
        raise MemoryOperationConflictError(
            f"operation id {operation.id!r} was reused with different arguments"
        )
    return MemoryMutation(
        version=mutation.version, replayed=True, existed=mutation.existed
    )
