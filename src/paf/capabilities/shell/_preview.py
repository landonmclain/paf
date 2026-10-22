"""What a destructive command would touch, worked out WITHOUT running it.

For an approval request the person deserves more than the command text: how
many files ``rm -rf build/ dist/`` would delete, whether any of them are tracked
by git (work you would lose), whether a path lies outside the shell's directory,
and whether a glob matched nothing. Everything here only reads the filesystem
(plus ``git ls-files``), bounded in entries and time so a huge tree cannot stall
the request.
"""

from __future__ import annotations

import glob
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from ._analyze import SimpleCommand

_MAX_ENTRIES = 20_000
_MAX_SECONDS = 1.5
_GLOB_CHARS = set("*?[")


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def _expand(target: str, cwd: Path) -> tuple[list[Path], bool]:
    """(existing paths, was_a_glob) for one target as the shell would expand it."""
    home = str(Path.home())
    text = os.path.expanduser(target.replace("$HOME", home).replace("${HOME}", home))
    base = text if text.startswith("/") else str(cwd / text)
    if _GLOB_CHARS & set(text):
        return [Path(p) for p in sorted(glob.glob(base))], True
    path = Path(base)
    return ([path] if path.exists() or path.is_symlink() else []), False


def _measure(paths: list[Path], deadline: float) -> tuple[int, int, bool]:
    """(files, bytes, truncated) under ``paths``, not following symlinks."""
    files = size = seen = 0
    for root in paths:
        if root.is_symlink() or root.is_file():
            files += 1
            try:
                size += root.lstat().st_size
            except OSError:
                pass
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                seen += 1
                files += 1
                try:
                    size += os.lstat(os.path.join(dirpath, name)).st_size
                except OSError:
                    pass
                if seen >= _MAX_ENTRIES or time.monotonic() > deadline:
                    return files, size, True
            dirnames[:] = [
                d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))
            ]
    return files, size, False


def _git_tracked(paths: list[Path], cwd: Path) -> int | None:
    if not paths:
        return 0
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", *[str(p) for p in paths]],
            cwd=cwd,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None  # not a repository
    return len([p for p in out.stdout.split(b"\0") if p])


def _rel(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd)) or "."
    except ValueError:
        return str(path)


def _find_roots(argv: tuple[str, ...]) -> list[str]:
    roots = []
    for arg in argv[1:]:
        if arg.startswith("-") or arg in ("(", "!", ")"):
            break
        roots.append(arg)
    return roots or ["."]


def preview(commands: tuple[SimpleCommand, ...], cwd: Path) -> dict[str, Any] | None:
    """A small, human-readable summary for the approval request (or ``None``)."""
    deadline = time.monotonic() + _MAX_SECONDS
    found: list[Path] = []
    missing: list[str] = []
    empty_globs: list[str] = []
    notes: list[str] = []
    overwrites: list[str] = []
    for cmd in commands:
        if "overwrite" in cmd.effects and "delete" not in cmd.effects:
            for target in cmd.targets:
                paths, _ = _expand(target, cwd)
                for path in paths:
                    try:
                        size = _human(path.lstat().st_size)
                    except OSError:
                        size = "?"
                    overwrites.append(f"{_rel(path, cwd)} ({size})")
            if not cmd.targets:
                notes.append(f"{cmd.argv[0]}: destination is computed at run time")
            continue
        if "data" in cmd.effects:
            notes.append(f"{cmd.name} runs destructive statements; there is no undo")
            continue
        if "delete" not in cmd.effects:
            continue
        if cmd.name == "find":
            roots = ", ".join(cmd.targets or _find_roots(cmd.argv))
            notes.append(f"find deletes whatever its expression matches under {roots}")
            continue
        if cmd.name == "rsync":
            dest = cmd.targets[0] if cmd.targets else "the destination"
            notes.append(f"rsync deletes files in {dest} that are not in the source")
            continue
        if cmd.name in ("docker", "podman", "nerdctl", "crontab"):
            notes.append(
                f"{' '.join(cmd.argv[:3])} removes what it names; there is no undo"
            )
            continue
        if cmd.name == "git":
            notes.append(
                "git would remove files (try the same command with -n to list them)"
            )
            continue
        if not cmd.targets:
            notes.append(
                f"{cmd.name}: targets are computed at run time ({' '.join(cmd.via) or 'dynamic'})"
            )
            continue
        for target in cmd.targets:
            paths, was_glob = _expand(target, cwd)
            if paths:
                found.extend(paths)
            elif was_glob:
                empty_globs.append(target)
            else:
                missing.append(target)
    if not (found or missing or empty_globs or notes or overwrites):
        return None
    result: dict[str, Any] = {}
    if overwrites:
        shown = ", ".join(overwrites[:6]) + (" …" if len(overwrites) > 6 else "")
        result["overwrites"] = f"{shown}: current contents are lost"
    if found:
        unique = list(
            dict.fromkeys(p.resolve() if not p.is_symlink() else p for p in found)
        )
        files, size, truncated = _measure(unique, deadline)
        shown = ", ".join(_rel(p, cwd) for p in unique[:6]) + (
            " …" if len(unique) > 6 else ""
        )
        count = f"{'at least ' if truncated else ''}{files:,} file{'s' if files != 1 else ''}, {_human(size)}"
        result["deletes"] = f"{shown}: {count}"
        warnings = []
        root = cwd.resolve()
        # Path containment, not a string prefix: /tmp/proj-old is not in /tmp/proj.
        outside = [p for p in unique if not Path(p).is_relative_to(root)]
        if outside:
            warnings.append(f"{len(outside)} path(s) outside {cwd}")
        tracked = _git_tracked(unique, cwd)
        if tracked:
            warnings.append(
                f"{tracked} file(s) tracked by git (restorable if committed; "
                f"uncommitted edits to them would be lost)"
            )
        if warnings:
            result["warning"] = "; ".join(warnings)
    if missing:
        result["not found"] = ", ".join(missing)
    if empty_globs:
        result["matched nothing"] = ", ".join(empty_globs)
    if notes:
        result["note"] = "; ".join(notes)
    return result
