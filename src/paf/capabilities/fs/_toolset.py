"""paf-owned filesystem toolset (originally copied from pydantic-ai-harness, MIT).

Sandboxed file operations scoped to a root directory. Copy of the harness
`FileSystemToolset` so paf owns the surface. Security model (root confinement,
symlink resolution, allow/deny/protected glob filtering, binary detection) is
kept as-is; `expected_hash` gives optimistic-concurrency stale-write guarding.
"""

from __future__ import annotations

import asyncio
import fnmatch
import functools
import hashlib
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from pathlib import Path
from typing import Concatenate, ParamSpec

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from paf.capabilities._edit import apply_edits
from paf.capabilities._storage._atomic import write_atomic

_P = ParamSpec("_P")

# Errors that mean "the model asked for something the tool couldn't do" -- a
# missing file, a denied path, a stale edit. pyai only feeds `ModelRetry` back
# to the model; any other exception aborts the whole run. `_recoverable`
# converts these so the agent can correct itself and continue.
_RECOVERABLE_ERRORS = (
    PermissionError,
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    ValueError,
)

# ModelRetry budget for the fs tools. They all raise `ModelRetry` (via
# `_recoverable`) on recoverable fs errors, and pydantic-ai defaults an
# un-declared tool to `ToolManager.default_max_retries` (1) — too low for the
# mid-turn parallel-call case, where a transient read/stat failure would abort
# the whole run. Matching the search toolset's `_SEARCH_TOOL_RETRIES` convention.
_FS_TOOL_RETRIES = 5

# How many files a single recursive content/glob walk may visit before it bails
# with a truncation marker. This is the hard backstop against walking an
# enormous tree (e.g. `data/benchmarks/` with ~1.8M files / the paf repo at
# ~1.87M files) — without it a `search_files`/`find_files` over `.` spins for
# minutes on enumeration before any per-file guard even runs. The 50K default
# bounds CPU AND wall-clock; a targeted search with `path` narrows past it.
_WALK_VISIT_BUDGET = 50_000
# Wall-clock budget for a recursive content/glob walk. There is NO timeout today
# (the fs `_recoverable` decorator only wraps exceptions into `ModelRetry`, it
# never bounds runtime) — a 1.8M-file rglob wedges the process until it's
# killed. `asyncio.wait_for` cancels the awaited walk on expiry.
_WALK_TIMEOUT_S = 30.0

# Directories we refuse to descend into during a recursive walk, regardless of
# budget. Dot-directories (`.git`, `.venv`, `node_modules`, `__pycache__` …) are
# never source an agent wants to content-search; skipping them at the dir level
# keeps the walk focused. We deliberately do NOT hard-skip non-dot subtrees like
# `data/` — a root search may legitimately want them — their scale is surfaced
# by list_directory and guarded by the budget + timeout instead.
_SKIP_DIRNAME = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".hg",
    ".svn",
}


class _WalkReport:
    """What a bounded recursive walk touched, for the truncation marker."""

    __slots__ = ("visited", "stopped_in", "hit_budget", "timed_out")

    def __init__(self) -> None:
        self.visited = 0
        self.stopped_in: str | None = None
        self.hit_budget = False
        self.timed_out = False


def _iter_walk_files(
    root: Path, real_root: Path, budget: int, report: _WalkReport
) -> Iterator[tuple[Path, str]]:
    """Lazily yield ``(absolute_path, rel_str)`` for files under ``root``.

    Replaces ``sorted(rglob('*'))`` — the eager, order-materializing walk that
    enumerated *every* path in a 25GB tree before any guard ran (a stall). This
    generator streams files top-down with ``os.walk``, pruning
    dot-directories at the directory level, and abides by a visit budget so the
    caller can stop early. ``rel_str`` is relative to ``real_root`` (the
    canonical root), matching the existing walkers' dot/access filtering.
    """
    top = str(root)
    for dirpath, dirnames, filenames in os.walk(top, topdown=True, followlinks=False):
        # Prune dot-dirs in place so os.walk won't descend into them.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRNAME]

        dp = Path(dirpath)
        if report.visited >= budget:
            # We haven't looked at this dir yet, so we can't claim it was
            # searched; note where the walk ran out of budget.
            if report.stopped_in is None:
                try:
                    report.stopped_in = str(dp.relative_to(real_root))
                except ValueError:
                    report.stopped_in = str(dp)
            report.hit_budget = True
            return
        for fn in filenames:
            if report.visited >= budget:
                report.hit_budget = True
                return
            fp = dp / fn
            try:
                rel_str = str(fp.relative_to(real_root))
            except ValueError:
                continue
            report.visited += 1
            yield fp, rel_str


def _truncation_marker(report: _WalkReport, reason: str) -> str:
    """Build the marker shown when a bounded walk stops early."""
    head = f"[... {reason} after visiting {report.visited} files"
    if report.stopped_in:
        head += f' (walk ran out of budget around "{report.stopped_in}")'
    head += '; narrow the search with the "path" argument to a specific '
    head += "subtree to search it fully]"
    return head


def _human_size(num: int) -> str:
    """Format a byte count for the directory listing scale indicator."""
    size = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _dir_scale(p: Path, cap: int = 5_000) -> tuple[int, int, bool]:
    """Boundedly count files + sum bytes under ``p`` (for list_directory).

    Walks the full subtree lazily up to ``cap`` files so `list_directory` can
    show an honest *scale* hint for a large subdirectory (e.g. paf's `data/` is
    ~1.8M files / ~25GB) without doing a full 1.8M-file sweep for every subdir.
    Returns ``(count, total_bytes, capped)`` where ``capped`` means the walk hit
    ``cap`` files and ``total_bytes`` is a lower bound (marked ``+`` by caller).
    """
    count = 0
    total = 0
    for dirpath, _, filenames in os.walk(str(p), topdown=True, followlinks=False):
        base = Path(dirpath)
        for fn in filenames:
            if count >= cap:
                return count, total, True
            count += 1
            try:
                total += (base / fn).lstat().st_size
            except OSError:  # pragma: no cover
                continue
    return count, total, False


async def _aiter_files(
    root: Path, real_root: Path, budget: int, report: _WalkReport
) -> AsyncIterator[tuple[Path, str]]:
    """Asynchronous wrapper over ``_iter_walk_files`` with a wall-clock budget.

    Yields each ``(abs_path, rel_str)`` as ``_iter_walk_files`` produces it,
    offloaded to a thread executor so per-file ``stat()``/``read_bytes()`` and
    the ``os.walk`` dirname scanning don't block the asyncio loop. The caller
    wraps the ``async for`` in ``asyncio.timeout()``; on expiry that cancels this
    generator so it stops awaiting further thread pulls and the search returns
    promptly instead of spinning through a 1.8M-file tree. The generator simply
    ends when the underlying walk is exhausted or hits the visit budget.
    """
    src = _iter_walk_files(root, real_root, budget, report)
    while True:
        item = await asyncio.to_thread(_sync_next, src)
        if item is _SENTINEL:
            return
        assert isinstance(item, tuple), "unexpected walker yield"
        yield item


class _Sentinel:
    __slots__ = ()


_SENTINEL = _Sentinel()


def _sync_next[T](it: Iterator[T]) -> T | _Sentinel:
    """``next(it)`` that returns a sentinel on exhaustion instead of raising.

    ``StopIteration`` cannot cross an ``asyncio.to_thread`` boundary (asyncio
    refuses to raise it into a Future), so we convert it to a sentinel here and
    translate it back in the async generator.
    """
    try:
        return next(it)
    except StopIteration:
        return _SENTINEL


async def _aiter_single(
    resolved: Path, real_root: Path
) -> AsyncIterator[tuple[Path, str]]:
    """Yield exactly one file for a direct single-file ``search_files``."""
    try:
        rel_str = str(resolved.relative_to(real_root))
    except ValueError:  # pragma: no cover
        return
    yield resolved, rel_str


def _recoverable(  # noqa: UP047  # ParamSpec/Concatenate decorator — PEP 695 inline params can't express this signature
    fn: Callable[Concatenate[FileSystemToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[FileSystemToolset, _P], Awaitable[str]]:
    """Surface model-correctable tool errors as `ModelRetry`."""

    @functools.wraps(fn)
    async def wrapper(
        self: FileSystemToolset, *args: _P.args, **kwargs: _P.kwargs
    ) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except _RECOVERABLE_ERRORS as e:
            raise ModelRetry(str(e)) from e

    return wrapper


def _format_lines(lines: Sequence[str], offset: int, limit: int) -> str:
    """Format pre-split lines with line numbers and continuation hint."""
    total = len(lines)

    if total == 0:
        return "(empty file)\n"

    if offset >= total:
        raise ValueError(f"Offset {offset} exceeds file length ({total} lines).")

    selected = lines[offset : offset + limit]
    numbered = [f"{i:>6}\t{line}" for i, line in enumerate(selected, start=offset + 1)]
    result = "".join(numbered)
    if not result.endswith("\n"):
        result += "\n"

    remaining = total - (offset + len(selected))
    if remaining > 0:
        next_offset = offset + len(selected)
        result += f"... ({remaining} more lines. Use offset={next_offset} to continue reading.)\n"

    return result


def _is_binary(data: bytes, sample_size: int = 8192) -> bool:
    """Detect binary content by checking for null bytes in the sample."""
    return b"\x00" in data[:sample_size]


def _content_hash(content: str) -> str:
    """Compute a short content hash for conflict detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


class FileSystemToolset(FunctionToolset[AgentDepsT]):
    """Toolset providing filesystem operations scoped to a root directory.

    Security model:
    - All paths resolved relative to root with canonical path checks
    - Symlinks resolved before authorization (prevents TOCTTOU)
    - Glob-based allow/deny filtering
    - Protected path patterns (e.g. `.git/`, `.env`)
    - Binary file detection blocks text operations
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        allowed_patterns: Sequence[str],
        denied_patterns: Sequence[str],
        protected_patterns: Sequence[str],
        max_read_lines: int,
        max_search_results: int,
        max_find_results: int,
        max_search_file_bytes: int,
        search_excerpt_chars: int,
        max_search_total_bytes: int,
        walk_visit_budget: int = _WALK_VISIT_BUDGET,
    ) -> None:
        super().__init__()
        self._root = root_dir.resolve()
        self._real_root = Path(os.path.realpath(self._root))
        self._allowed_patterns = list(allowed_patterns)
        self._denied_patterns = list(denied_patterns)
        self._protected_patterns = list(protected_patterns)
        self._max_read_lines = max_read_lines
        self._max_search_results = max_search_results
        self._max_find_results = max_find_results
        self._max_search_file_bytes = max_search_file_bytes
        self._search_excerpt_chars = search_excerpt_chars
        self._max_search_total_bytes = max_search_total_bytes
        self._walk_visit_budget = walk_visit_budget

        self.add_function(self.read_file, name="read_file", retries=_FS_TOOL_RETRIES)
        self.add_function(self.write_file, name="write_file", retries=_FS_TOOL_RETRIES)
        self.add_function(self.edit_file, name="edit_file", retries=_FS_TOOL_RETRIES)
        self.add_function(
            self.list_directory, name="list_directory", retries=_FS_TOOL_RETRIES
        )
        self.add_function(
            self.search_files, name="search_files", retries=_FS_TOOL_RETRIES
        )
        self.add_function(self.find_files, name="find_files", retries=_FS_TOOL_RETRIES)
        self.add_function(
            self.create_directory, name="create_directory", retries=_FS_TOOL_RETRIES
        )
        self.add_function(self.file_info, name="file_info", retries=_FS_TOOL_RETRIES)

    def _matches(self, path: str, pattern: str) -> bool:
        """Glob-match a relative path, treating a leading `**/` as 'any directory, including the root'.

        `fnmatch` has no recursive `**`, so a bare `**/secrets*` would miss a
        root-level `secrets.yaml` -- there's no leading directory to match.
        Retrying with the `**/` prefix stripped covers the zero-directory case.
        """
        if fnmatch.fnmatch(path, pattern):
            return True
        if pattern.startswith("**/"):
            return fnmatch.fnmatch(path, pattern[3:])
        return False

    def _first_matching_pattern(self, path: str, patterns: list[str]) -> str | None:
        """Return the first pattern that matches path, or None."""
        return next((p for p in patterns if self._matches(path, p)), None)

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to root, rejecting traversal.

        Uses os.path.realpath for symlink resolution before checking containment.
        """
        candidate = (self._root / path).resolve()
        real = Path(os.path.realpath(candidate))
        if not real.is_relative_to(self._real_root):
            raise PermissionError(f"Path {path!r} resolves outside the root directory.")

        return real

    def _check_access(
        self, path: str, *, write: bool = False, check_allowed: bool = True
    ) -> None:
        """Validate path against allow/deny/protected patterns.

        `check_allowed=False` skips the `allowed_patterns` gate. Walkers
        (`list_directory`, `search_files`, `find_files`) pass it so their root
        directory isn't required to match `allowed_patterns` itself -- `.` or
        `src` would never match a file pattern like `src/*.py`. The walk's
        entries are still filtered against `allowed_patterns` per-entry via
        `_is_accessible`. Denied and protected patterns continue to gate the
        root.
        """
        if write and self._protected_patterns:
            matched = self._first_matching_pattern(path, self._protected_patterns)
            if matched:
                raise PermissionError(
                    f"Path {path!r} is protected (matches {matched!r})."
                )

        if self._denied_patterns:
            matched = self._first_matching_pattern(path, self._denied_patterns)
            if matched:
                raise PermissionError(
                    f"Path {path!r} is denied by pattern {matched!r}."
                )

        if check_allowed and self._allowed_patterns:
            if not any(self._matches(path, p) for p in self._allowed_patterns):
                raise PermissionError(
                    f"Path {path!r} does not match any allowed pattern."
                )

    def _is_accessible(self, path: str, *, write: bool = False) -> bool:
        """Predicate form of `_check_access` for filtering recursive walkers.

        Used by `list_directory`, `search_files`, and `find_files` to skip
        children that would be rejected if accessed directly. Note this only
        checks the relative path against patterns; it does not resolve symlinks.
        """
        if write and self._protected_patterns:
            if self._first_matching_pattern(path, self._protected_patterns) is not None:
                return False
        if self._denied_patterns:
            if self._first_matching_pattern(path, self._denied_patterns) is not None:
                return False
        if self._allowed_patterns and not any(
            self._matches(path, p) for p in self._allowed_patterns
        ):
            return False
        return True

    def _relative_to_root(self, resolved: Path) -> str:
        """Canonical path of a resolved location relative to the real root."""
        return str(resolved.relative_to(self._real_root))

    def _safe_resolve(
        self, path: str, *, write: bool = False, check_allowed: bool = True
    ) -> Path:
        """Resolve and access-check a path in one step.

        Resolution happens first so the access check matches patterns against
        the canonical path relative to the root, collapsing `.`/`..`/`//`
        segments that would otherwise slip past a literal pattern (e.g.
        `config/./secret.txt` evading a `config/secret.txt` deny rule).
        """
        resolved = self._resolve_path(path)
        self._check_access(
            self._relative_to_root(resolved), write=write, check_allowed=check_allowed
        )
        return resolved

    @_recoverable
    async def read_file(
        self, path: str, *, offset: int = 0, limit: int | None = None
    ) -> str:
        """Read a text file with line numbers.

        Args:
            path: File path relative to the root directory.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return (default: 2000).

        Returns:
            File content with line numbers, plus metadata header.
        """
        if limit is None:
            limit = self._max_read_lines
        resolved = self._safe_resolve(path)
        if not resolved.is_file():
            if resolved.is_dir():
                raise FileNotFoundError(
                    f"'{path}' is a directory, not a file (current root: '{self._root}')."
                )
            raise FileNotFoundError(
                f"File not found: {path} (current root: '{self._root}')"
            )

        raw = resolved.read_bytes()
        if _is_binary(raw):
            size = len(raw)
            return f"[Binary file: {size} bytes. Use a binary-aware tool to inspect.]"

        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        content_hash = _content_hash(text)

        header = f"[{path} | {len(lines)} lines | hash:{content_hash}]\n"
        return header + _format_lines(lines, offset, limit)

    @_recoverable
    async def write_file(
        self, path: str, content: str, *, expected_hash: str | None = None
    ) -> str:
        """Create or overwrite a file with conflict detection.

        Args:
            path: File path relative to the root directory.
            content: The text content to write.
            expected_hash: If provided, the write is rejected when the file exists
                and its current hash doesn't match (optimistic concurrency).

        Returns:
            Confirmation message with new hash.
        """
        resolved = self._safe_resolve(path, write=True)

        # Optimistic concurrency: reject stale writes
        if expected_hash is not None and resolved.is_file():
            current = resolved.read_text(encoding="utf-8")
            current_hash = _content_hash(current)
            if current_hash != expected_hash:
                raise ValueError(
                    f"Conflict: file {path!r} has changed (expected hash:{expected_hash}, "
                    f"got hash:{current_hash}). Re-read the file and retry."
                )

        if not resolved.parent.exists():
            parent_rel = str(resolved.parent.relative_to(self._root))
            raise FileNotFoundError(
                f"Parent directory '{parent_rel}' does not exist (current root: '{self._root}'). "
                "Use create_directory first."
            )
        write_atomic(resolved, content)
        new_hash = _content_hash(content)
        lines = len(content.splitlines())
        return (
            f"Wrote {len(content)} chars ({lines} lines) to {path}. [hash:{new_hash}]"
        )

    @_recoverable
    async def edit_file(
        self, path: str, edits: list[dict], *, expected_hash: str | None = None
    ) -> str:
        """Edit a file by exact string replacement with conflict detection.

        ``edits`` is a list of ``{"old_text", "new_text"}`` pairs applied in one
        atomic write. Each ``old_text`` must appear exactly once in the file;
        include surrounding context lines to ensure uniqueness. A single edit is
        ``edits=[{"old_text": "...", "new_text": "..."}]``; pass several pairs to
        batch independent changes to the same file in one call.

        Args:
            path: File path relative to the root directory.
            edits: ``[{old_text, new_text}, ...]`` pairs to apply, in order.
            expected_hash: If provided, rejects the edit when the file's
                current hash doesn't match (optimistic concurrency).

        Returns:
            Summary with new hash + per-pair diffs; raises a recoverable error
            on any failure (never reports success for an unpersisted edit).
        """
        resolved = self._safe_resolve(path, write=True)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"File not found: {path} (current root: '{self._root}')"
            )

        result = apply_edits(
            resolved,
            edits,
            expected_version=expected_hash,
            read=lambda p: p.read_text(encoding="utf-8"),
            write=write_atomic,
        )
        return result.summary(path)

    @_recoverable
    async def list_directory(self, path: str = ".") -> str:
        """List the contents of a directory.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            A newline-separated listing with type indicators and sizes.
        """
        # The listing root is gated by denied/protected patterns but not by
        # allowed_patterns: a directory like '.' never matches a file pattern.
        # Entries are filtered per-entry against allowed_patterns below.
        resolved = self._safe_resolve(path, check_allowed=False)
        if not resolved.is_dir():
            raise NotADirectoryError(
                f"Not a directory: {path} (current root: '{self._root}')"
            )

        entries: list[str] = []
        for entry in sorted(resolved.iterdir()):
            try:
                rel_path = entry.relative_to(self._real_root)
            except ValueError:  # pragma: no cover
                continue
            # Skip dotfiles and dot-directories, matching search_files and
            # find_files so the three walkers agree on what exists.
            if any(part.startswith(".") for part in rel_path.parts):
                continue
            rel = str(rel_path)
            # Apply the same allow/deny/protected filtering used for direct
            # access so a directory listing can't leak patterns the agent
            # couldn't otherwise read or write.
            if not self._is_accessible(rel, write=True):
                continue
            if entry.is_dir():
                n, total, capped = _dir_scale(entry)
                cnt_suffix = "+" if capped else ""
                entries.append(
                    f"{rel}/  ({n:,}{cnt_suffix} files, {_human_size(total)}{cnt_suffix})"
                )
            else:
                try:
                    size = entry.stat().st_size
                except (
                    OSError
                ):  # pragma: no cover  # file deleted between iterdir and stat
                    size = 0
                entries.append(f"{rel}  ({size} bytes)")
        return "\n".join(entries) if entries else "(empty directory)"

    @_recoverable
    async def search_files(
        self,
        pattern: str,
        *,
        path: str = ".",
        include_glob: str | None = None,
        timeout: float | None = _WALK_TIMEOUT_S,
    ) -> str:
        """Search file contents using a regular expression.

        Args:
            pattern: Regex pattern to search for.
            path: Directory to search in, relative to the root directory.
            include_glob: If provided, only search files matching this glob (e.g. '*.py').
            timeout: Wall-clock budget (seconds) for the recursive walk. The
                walk is cancelled on expiry. There was no timeout before — a
                content search over a huge tree (paf's `data/` is ~1.8M files)
                could spin for minutes and wedge the run, because the old
                `rglob('*')` enumerated and sorted every path before any
                per-file guard ran.

        Returns:
            str: Matching lines formatted as file:line_number:text.
        """
        # See list_directory: the search root isn't gated by allowed_patterns;
        # matched files are filtered per-entry below.
        resolved = self._safe_resolve(path, check_allowed=False)
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        results: list[str] = []
        total_bytes = 0
        real_root = Path(os.path.realpath(self._root))
        report = _WalkReport()
        budget = self._walk_visit_budget

        stop_reason: str | None = None
        try:
            async with asyncio.timeout(
                timeout if timeout is not None else float("inf")
            ):
                files_aiter = (
                    _aiter_single(resolved, real_root)
                    if resolved.is_file()
                    else _aiter_files(resolved, real_root, budget, report)
                )
                async for file_path, rel_str in files_aiter:
                    if not file_path.is_file():
                        continue
                    try:
                        rel_parts = file_path.relative_to(real_root).parts
                    except ValueError:  # pragma: no cover
                        continue
                    if any(part.startswith(".") for part in rel_parts):
                        continue
                    # Apply the same allow/deny/protected filtering used for direct
                    # access so a recursive search can't read patterns the agent
                    # couldn't otherwise read.
                    if not self._is_accessible(rel_str, write=True):
                        continue
                    if include_glob and not fnmatch.fnmatch(rel_str, include_glob):
                        continue
                    # Guard 1: skip oversized files (e.g. memory journals / paf
                    # snapshots are huge single-line JSON). A content search rarely
                    # wants a multi-MB blob; read_file is the tool for that. Reading
                    # every giant file in a walked tree is what turned a simple search
                    # into a minutes-long, megabyte-result stall.
                    try:
                        size = file_path.stat().st_size
                    except OSError:  # pragma: no cover
                        continue
                    if size > self._max_search_file_bytes:
                        continue
                    try:
                        raw = file_path.read_bytes()
                    except OSError:  # pragma: no cover
                        continue
                    if _is_binary(raw):
                        continue
                    text = raw.decode("utf-8", errors="replace")
                    for line_num, line in enumerate(text.splitlines(), start=1):
                        if not compiled.search(line):
                            continue
                        # Guard 2: excerpt the matching line rather than returning the
                        # whole (possibly 300KB) single-line record.
                        excerpt = line[: self._search_excerpt_chars]
                        if len(line) > self._search_excerpt_chars:
                            excerpt += "..."
                        entry = f"{rel_str}:{line_num}:{excerpt}"
                        # Guard 3: hard byte cap on the whole result string so a walk
                        # over many small matches can't balloon the tool return either.
                        total_bytes += len(entry)
                        results.append(entry)
                        if total_bytes >= self._max_search_total_bytes:
                            results.append(
                                f"[... truncated at {self._max_search_total_bytes} bytes]"
                            )
                            return "\n".join(results)
                    if len(results) >= self._max_search_results:
                        results.append(
                            f"[... truncated at {self._max_search_results} matches]"
                        )
        except TimeoutError:
            report.timed_out = True
            stop_reason = f"timed out after {timeout}s"
        else:
            if report.hit_budget:
                stop_reason = "stopped at the file-walk budget"

        if (report.hit_budget or report.timed_out) and not results:
            results.append(_truncation_marker(report, stop_reason or "walk stopped"))

        return "\n".join(results) if results else "No matches found."

    @_recoverable
    async def find_files(
        self, pattern: str, *, path: str = ".", timeout: float | None = _WALK_TIMEOUT_S
    ) -> str:
        """Find files by glob pattern (name matching, not content search).

        Args:
            pattern: Glob pattern to match (e.g. '*.py', '**/*.json').
            path: Directory to search in, relative to the root directory.
            timeout: Wall-clock budget (seconds) for the glob walk. Previously
                the ``sorted(glob(pattern))`` eagerly materialized every match
                in the tree before any cap ran, so a broad ``**`` pattern over
                a huge tree (paf's `data/` is ~1.8M files) could spin for
                minutes. The glob is now consumed lazily and cancelled on
                timeout.

        Returns:
            Newline-separated list of matching file paths relative to root.
        """
        # See list_directory: the find root isn't gated by allowed_patterns;
        # matched entries are filtered per-entry below.
        resolved = self._safe_resolve(path, check_allowed=False)
        if not resolved.is_dir():
            raise NotADirectoryError(
                f"Not a directory: {path} (current root: '{self._root}')"
            )

        matches: list[str] = []
        real_root = Path(os.path.realpath(self._root))
        it = iter(resolved.glob(pattern))  # lazy generator; consumed via to_thread
        stop_reason: str | None = None
        try:
            async with asyncio.timeout(
                timeout if timeout is not None else float("inf")
            ):
                while True:
                    m = await asyncio.to_thread(_sync_next, it)
                    if m is _SENTINEL:
                        break
                    assert isinstance(m, Path)
                    try:
                        rel_parts = m.relative_to(real_root).parts
                    except ValueError:  # pragma: no cover
                        continue
                    if any(part.startswith(".") for part in rel_parts):
                        continue
                    rel = str(m.relative_to(real_root))
                    # Apply the same allow/deny/protected filtering used for direct
                    # access so a glob find can't surface patterns the agent
                    # couldn't otherwise see.
                    if not self._is_accessible(rel, write=True):
                        continue
                    suffix = "/" if m.is_dir() else ""
                    matches.append(f"{rel}{suffix}")
                    if len(matches) >= self._max_find_results:
                        matches.append(
                            f"[... truncated at {self._max_find_results} matches]"
                        )
                        break
        except TimeoutError:
            stop_reason = f"timed out after {timeout}s on a broad glob"
            if not matches:
                matches.append(
                    f'[... {stop_reason}; narrow the search with the "path" argument '
                    "or a more specific glob]"
                )

        return "\n".join(matches) if matches else "No matches found."

    @_recoverable
    async def create_directory(self, path: str) -> str:
        """Create a directory and any missing parents.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            Confirmation message.
        """
        resolved = self._safe_resolve(path, write=True)
        resolved.mkdir(parents=True, exist_ok=True)
        return f"Created directory: {path}"

    @_recoverable
    async def file_info(self, path: str) -> str:
        """Get metadata about a file or directory.

        Args:
            path: File or directory path relative to the root directory.

        Returns:
            Formatted metadata including size, type, and permissions.
        """
        resolved = self._safe_resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Path not found: {path} (current root: '{self._root}')"
            )

        # Check if the original (pre-resolve) path is a symlink
        original = self._root / path
        is_link = original.is_symlink()

        stat = resolved.stat()
        kind = "directory" if resolved.is_dir() else "file"
        size = stat.st_size

        parts = [f"path: {path}", f"type: {kind}", f"size: {size} bytes"]

        if resolved.is_file():
            raw = resolved.read_bytes()
            is_bin = _is_binary(raw)
            parts.append(f"binary: {is_bin}")
            if not is_bin:
                text = raw.decode("utf-8", errors="replace")
                parts.append(f"lines: {len(text.splitlines())}")
                parts.append(f"hash: {_content_hash(text)}")

        if is_link:
            parts.append(f"symlink_target: {os.readlink(original)}")

        return "\n".join(parts)
