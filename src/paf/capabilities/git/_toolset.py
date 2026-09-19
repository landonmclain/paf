"""paf-owned git toolset -- safe repository inspection and checkpointing.

Root-confined git tools (AsyncFunctionToolset over anyio). Session logging is
dropped (not paf's pattern) and `ModelRetry` is used for model-correctable
errors.

Safety rules encoded in the tool docstrings for the agent:
- Prefer git_status and git_changed_files before git_diff.
- Prefer git_diff with explicit paths when reviewing changes.
- Do not call git_checkpoint unless the user asked to
  checkpoint/commit/save changes or the workflow explicitly requires a safety
  checkpoint.
- Do not call git_restore_paths unless the user explicitly asked to discard,
  revert, restore, undo, or throw away local changes.
- Never use broad `git add -A` behavior -- checkout requires explicit paths.
"""

from __future__ import annotations

import functools
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Concatenate, ParamSpec

import anyio
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

_P = ParamSpec("_P")

DEFAULT_DIFF_CHUNK_LINES = 400
MAX_DIFF_CHUNK_LINES = 1200

# A subprocess wraps up with these exit-code suffixes; we only read raw stdout
# for diff chunking (git_diff needs clean diff text).
_DEFAULT_TIMEOUT = 60.0

# Recovery-prone git tools get a real retry budget (not pydantic-ai's default of
# 1), so transient/model-correctable ModelRetry failures feed back to the model
# to fix instead of killing the run. Mirrors _FS_TOOL_RETRIES in fs.
_GIT_TOOL_RETRIES = 5


def _recoverable(  # noqa: UP047  # ParamSpec/Concatenate decorator — PEP 695 inline params can't express this signature
    fn: Callable[Concatenate[GitToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[GitToolset, _P], Awaitable[str]]:
    """Surface model-correctable git errors as `ModelRetry`.

    A path outside the workspace, a bad path, or an empty-path refusal is
    something the model can fix and retry; pyai only feeds `ModelRetry` back
    to the model. Any other exception aborts the run.
    """

    @functools.wraps(fn)
    async def wrapper(self: GitToolset, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except (PermissionError, ValueError) as e:
            raise ModelRetry(str(e)) from e

    return wrapper


def _resolve_safe_path(root: Path, user_path: str) -> Path:
    """Resolve `user_path` under `root`, raising if it escapes.

    Mirrors the fs capability's root confinement: an absolute or relative path
    is resolved (`..`/symlinks followed) and must resolve inside `root` or a
    `ValueError` is raised so the model can recover.
    """
    p = Path(user_path)
    full = (root / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        full.relative_to(root)
    except ValueError as e:
        raise ValueError(f"Path outside allowed workspace: {full}") from e
    return full


class GitToolset(FunctionToolset):
    """Safe git inspection + checkpoint tools rooted at a workspace root.

    Mostly read-only inspection (`git_status`, `git_changed_files`,
    `git_diff`) plus two guarded write tools (`git_checkpoint`,
    `git_restore_paths`) that refuse empty path lists so the model can never
    broad-`add -A` or wipe more than it named.
    """

    def __init__(
        self,
        *,
        root: Path,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__()
        self._root = root.resolve()
        self._timeout = timeout

        self.add_function(self.git_status, name="git_status", retries=_GIT_TOOL_RETRIES)
        self.add_function(
            self.git_changed_files, name="git_changed_files", retries=_GIT_TOOL_RETRIES
        )
        self.add_function(self.git_diff, name="git_diff", retries=_GIT_TOOL_RETRIES)
        self.add_function(
            self.git_checkpoint, name="git_checkpoint", retries=_GIT_TOOL_RETRIES
        )
        self.add_function(
            self.git_restore_paths, name="git_restore_paths", retries=_GIT_TOOL_RETRIES
        )

    async def for_run(self, ctx: RunContext) -> AbstractToolset:
        """Return a fresh instance per run (stateless here, but keep isolation)."""
        return GitToolset(root=self._root, timeout=self._timeout)

    def _git_cwd(self, rel_path: str) -> Path:
        """Resolve the git command working directory under the root."""
        cwd = _resolve_safe_path(self._root, rel_path)
        if cwd.is_file():
            cwd = cwd.parent
        if not cwd.is_dir():
            raise ValueError(f"Working directory is not a directory: {cwd}")
        return cwd

    def _validated_paths(self, paths: list[str] | None) -> list[str]:
        """Validate user-provided git paths against the root.

        Returns workspace-relative POSIX paths safe to pass after `--`.
        """
        if not paths:
            return []
        safe: list[str] = []
        for user_path in paths:
            resolved = _resolve_safe_path(self._root, user_path)
            safe.append(resolved.relative_to(self._root).as_posix())
        return safe

    async def _run_git(self, cwd: Path, parts: list[str]) -> tuple[int, str, str]:
        """Run a git command, returning (exit_code, stdout, stderr)."""
        proc = await anyio.open_process(
            parts,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout_stream = proc.stdout
            stderr_stream = proc.stderr
            assert stdout_stream is not None
            assert stderr_stream is not None
            out: list[bytes] = []
            err: list[bytes] = []

            async def _read_out() -> None:
                async for chunk in stdout_stream:
                    out.append(chunk)

            async def _read_err() -> None:
                async for chunk in stderr_stream:
                    err.append(chunk)

            with anyio.fail_after(self._timeout):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(_read_out)
                    tg.start_soon(_read_err)
                await proc.wait()
        except TimeoutError:
            await _kill_process(proc)
            with anyio.CancelScope(shield=True):
                await proc.wait()
            raise TimeoutError(
                f"git command timed out after {self._timeout}s: {' '.join(parts)}"
            ) from None
        finally:
            await proc.aclose()

        exit_code = proc.returncode if proc.returncode is not None else 0
        stdout = b"".join(out).decode("utf-8", errors="replace")
        stderr = b"".join(err).decode("utf-8", errors="replace")
        return exit_code, stdout, stderr

    def _format_result(self, *, stdout: str, stderr: str, exit_code: int) -> str:
        """Assemble a labeled result string like the shell toolset's output."""
        parts: list[str] = []
        if stdout:
            parts.append(f"[stdout]\n{stdout}")
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        output = "\n".join(parts) if parts else "(no output)"
        if exit_code != 0:
            output = f"{output}\n[exit code: {exit_code}]"
        return output

    # -- tools -------------------------------------------------------------

    @_recoverable
    async def git_status(self, rel_path: str = ".") -> str:
        """Show concise git working tree status.

        Safe read-only inspection tool. Use this before editing, checkpointing,
        restoring, or reviewing changes when you need to know whether the repo
        has modified, staged, deleted, renamed, or untracked files.

        Prefer this as the first git tool when the user asks: "what changed?",
        "are there unstaged changes?", "check git", "review the repo state",
        or "can we checkpoint this?".

        This returns `git status --short`, not full diffs.

        Args:
            rel_path: Workspace-relative directory or file inside the git repo.
                Usually "." unless working in a specific subproject.
        """
        cwd = self._git_cwd(rel_path)
        exit_code, stdout, stderr = await self._run_git(
            cwd, ["git", "status", "--short"]
        )
        return self._format_result(stdout=stdout, stderr=stderr, exit_code=exit_code)

    @_recoverable
    async def git_changed_files(self, rel_path: str = ".") -> str:
        """Show changed tracked files without returning full diff content.

        Safe read-only inspection tool. Use this before git_diff when there may
        be many changes and you need a compact list of modified/deleted/renamed
        tracked files.

        This runs `git diff --name-status`, so it shows unstaged tracked-file
        differences. It does not show untracked files -- use git_status if
        untracked files matter.

        Prefer this when: git_status shows many modified files; you need to
        choose which paths to inspect with git_diff; you want a quick
        changed-file summary before checkpointing.

        Args:
            rel_path: Workspace-relative directory or file inside the git repo.
                Usually "." unless working in a specific subproject.
        """
        cwd = self._git_cwd(rel_path)
        exit_code, stdout, stderr = await self._run_git(
            cwd, ["git", "diff", "--name-status"]
        )
        return self._format_result(stdout=stdout, stderr=stderr, exit_code=exit_code)

    @_recoverable
    async def git_diff(
        self,
        rel_path: str = ".",
        paths: list[str] | None = None,
        start_line: int = 1,
        max_lines: int = DEFAULT_DIFF_CHUNK_LINES,
    ) -> str:
        """Inspect unstaged git diff in numbered chunks.

        Safe read-only inspection tool. Use this to review code changes before
        explaining, testing, checkpointing, or deciding what to do next.

        Prefer passing explicit `paths` when possible. This keeps output smaller
        and reduces the chance that unrelated local changes distract the agent.

        Recommended workflow:
        1. git_status
        2. git_changed_files, if many files changed
        3. git_diff with paths=[...] for the relevant files
        4. Continue paging with start_line if the diff is truncated

        This runs `git diff` (with `-- <paths>` when paths are provided). It
        does not show staged diffs.

        Args:
            rel_path: Workspace-relative directory or file inside the git repo.
                Usually "." unless working in a specific subproject.
            paths: Optional explicit workspace-relative paths to diff.
            start_line: 1-based line number for paging through large diffs.
            max_lines: Maximum diff lines to return, capped at
                MAX_DIFF_CHUNK_LINES.
        """
        cwd = self._git_cwd(rel_path)
        safe_paths = self._validated_paths(paths)

        if safe_paths:
            parts = ["git", "diff", "--", *safe_paths]
        else:
            parts = ["git", "diff"]

        exit_code, stdout, stderr = await self._run_git(cwd, parts)
        if exit_code != 0:
            return self._format_result(
                stdout=stdout, stderr=stderr, exit_code=exit_code
            )
        return _format_diff_chunk(
            diff_text=stdout,
            start_line=start_line,
            max_lines=max_lines,
        )

    @_recoverable
    async def git_checkpoint(
        self, message: str, paths: list[str], rel_path: str = "."
    ) -> str:
        """Create a git checkpoint commit for explicit paths only.

        Write tool. Stages explicit files and creates a commit.

        Use this only when the user explicitly asks to checkpoint, commit, save,
        or preserve changes, or when the workflow clearly requires a safety
        checkpoint before risky edits.

        This tool intentionally refuses to run without explicit `paths`. It does
        not support broad `git add -A` because that can accidentally commit
        unrelated local changes.

        Recommended workflow before calling:
        1. git_status
        2. git_changed_files or git_diff
        3. Choose explicit paths that belong in the checkpoint
        4. Call git_checkpoint with a concise message and paths=[...]

        Do not use this tool to commit unknown/unreviewed broad changes.

        Args:
            message: Commit message for the checkpoint.
            paths: Explicit workspace-relative files to stage and commit.
                Required. The tool refuses to run if this is empty or omitted.
            rel_path: Workspace-relative directory or file inside the git repo.
                Usually "." unless working in a specific subproject.
        """
        if not paths:
            return (
                "Refusing to create checkpoint without explicit paths.\n\n"
                "Pass the specific changed files to checkpoint using paths=[...]. "
                "This avoids accidentally committing unrelated local changes."
            )

        cwd = self._git_cwd(rel_path)
        safe_paths = self._validated_paths(paths)

        add_ec, add_out, add_err = await self._run_git(
            cwd, ["git", "add", "--", *safe_paths]
        )
        commit_ec, commit_out, commit_err = await self._run_git(
            cwd, ["git", "commit", "-m", message]
        )
        add_block = self._format_result(
            stdout=add_out, stderr=add_err, exit_code=add_ec
        )
        commit_block = self._format_result(
            stdout=commit_out, stderr=commit_err, exit_code=commit_ec
        )
        return f"{add_block}\n\n---\n\n{commit_block}"

    @_recoverable
    async def git_restore_paths(self, paths: list[str], rel_path: str = ".") -> str:
        """Discard unstaged changes for explicit paths.

        Destructive write tool. Use this only when the user explicitly asks to
        restore, revert, discard, undo, or throw away uncommitted changes for
        specific files.

        This runs `git restore -- <paths>`. It can permanently discard
        uncommitted local edits in the specified paths.

        Do not call this just because tests failed. Do not call this as a
        general cleanup step. Do not call this without a clear user request or
        an explicit rollback workflow.

        This tool requires explicit `paths` and refuses an empty path list.

        Recommended workflow before calling:
        1. git_status
        2. git_diff with the specific paths
        3. Confirm the paths are exactly the changes to discard
        4. Call git_restore_paths only for those paths

        Args:
            paths: Explicit workspace-relative files whose unstaged changes
                should be discarded.
            rel_path: Workspace-relative directory or file inside the git repo.
                Usually "." unless working in a specific subproject.
        """
        if not paths:
            return "No paths provided for restore."

        cwd = self._git_cwd(rel_path)
        safe_paths = self._validated_paths(paths)
        exit_code, stdout, stderr = await self._run_git(
            cwd, ["git", "restore", "--", *safe_paths]
        )
        return self._format_result(stdout=stdout, stderr=stderr, exit_code=exit_code)


async def _kill_process(proc: Any) -> None:
    """Best-effort terminate a timed-out subprocess."""
    try:
        proc.terminate()
        with anyio.move_on_after(2.0):
            await proc.wait()
            return
        proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _format_diff_chunk(
    *,
    diff_text: str,
    start_line: int,
    max_lines: int,
) -> str:
    """Format a git diff as a numbered, paged line chunk.

    Args:
        diff_text: Raw git diff text.
        start_line: 1-based starting line to display.
        max_lines: Maximum number of lines to display.

    Returns:
        Human-readable chunk with total line count and continuation metadata.
    """
    if start_line < 1:
        return "start_line must be >= 1"
    if max_lines < 1:
        return "max_lines must be >= 1"

    max_lines = min(max_lines, MAX_DIFF_CHUNK_LINES)
    lines = diff_text.splitlines()
    total_lines = len(lines)

    if total_lines == 0:
        return "Diff lines: 0\nNo diff."

    start_idx = start_line - 1
    if start_idx >= total_lines:
        return (
            f"Diff lines: {total_lines}\nstart_line {start_line} is beyond end of diff."
        )

    end_idx = min(start_idx + max_lines, total_lines)
    selected = lines[start_idx:end_idx]
    next_start_line = end_idx + 1 if end_idx < total_lines else None

    body = "\n".join(
        f"{line_no}: {line}" for line_no, line in enumerate(selected, start=start_line)
    )

    return (
        f"Diff lines: {total_lines}\n"
        f"Showing lines {start_line}-{end_idx} of {total_lines}\n"
        f"Next start_line: {next_start_line}\n"
        f"Use git_diff with start_line={next_start_line} to continue if needed.\n\n"
        f"{body}"
    )
