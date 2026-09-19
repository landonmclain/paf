"""Shared typed memory records, protocols, and validation for the memory capability.

This module holds the load-bearing shared types the memory spine needs
(``_journal``, ``_index``, ``_render``, ``_view``, the owner's
``JournalAppendToolset``):

- ``MemoryFile`` / ``MemoryOperation`` / ``MemoryMutation`` / ``MemorySearchMatch``
  / ``MemorySearchResult``  — shared record types
- ``MemoryStore`` / ``SearchableMemoryStore``  — protocols
- ``validate_store_path`` / ``validate_store_prefix`` / ``lexical_search`` —
  shared helpers

``Memory.store`` is required: the read path always supplies the read-only
``JournalViewStore`` of the rendered journal. Atomic writes come from the shared
``_storage._atomic``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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
