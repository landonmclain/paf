"""DockerExecutor: the sandbox backend (Linux / macOS / Windows via WSL2).

Runs commands inside a locked-down Docker container: no network, read-only
filesystem, non-root user, resource limits. Two modes:

- ``mode="exec"`` (default): one long-lived container, ``docker exec`` per
  command (~60ms). State persists between commands.
- ``mode="run"``: fresh ``docker run --rm`` per command (~250ms). Clean slate
  each time, strongest isolation.

Background processes map to detached containers; ``check`` / ``stop`` become
``docker logs`` / ``docker stop``.
"""

from __future__ import annotations

import atexit
import fnmatch
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import anyio
from anyio.to_thread import run_sync

from paf.capabilities._sandbox._protocol import (
    BackgroundHandle,
    BackgroundStatus,
    ExecResult,
)

# The host Docker daemon socket. A sandboxed ``docker`` command must never reach
# the host daemon, so this path is never mounted into the container (see
# ``_overlay_mounts``).
_HOST_DOCKER_SOCKET = "/var/run/docker.sock"


# Directories the exclude walk does not descend into: vendored or generated trees
# that are huge (the walk runs on every command) and full of legitimate
# ``*.pem``/``*.key`` files (``certifi``'s CA bundle), where blanking would break
# the toolchain. Matching exclude patterns are still honoured everywhere else.
_WALK_PRUNE = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)


def find_excluded(
    cwd: Path,
    patterns: Sequence[str],
    on_external_link: Callable[[Path, Path], None] | None = None,
) -> list[Path]:
    """Every path under ``cwd`` that an exclude pattern selects.

    A pattern with no ``/`` and no ``**`` is a BASENAME glob matched at any depth
    (``.env`` finds ``sub/.env``; ``.git`` finds a nested submodule's ``.git``).
    Anything else is a path glob relative to ``cwd`` (the original semantics).
    A matched directory is returned whole and not descended into. A symlink that
    matches is followed to its target if that is inside ``cwd`` (so aliasing a
    secret under another name does not expose it). One whose target is outside
    ``cwd`` is skipped (nothing outside the workspace is mounted into the
    sandbox, so there is nothing to blank) after being reported to
    ``on_external_link(link, target)``, which may raise to refuse loudly.
    """
    name_pats = [p for p in patterns if "/" not in p and "**" not in p]
    path_pats = [p for p in patterns if p not in name_pats]
    found: dict[Path, None] = {}

    def add(path: Path) -> None:
        if path.is_symlink():
            target = Path(os.path.realpath(path))
            if not target.is_relative_to(Path(os.path.realpath(cwd))):
                if on_external_link is not None:
                    on_external_link(path, target)
                return
            path = cwd / target.relative_to(Path(os.path.realpath(cwd)))
        found[path] = None

    for pattern in path_pats:
        for hit in sorted(cwd.glob(pattern)):
            add(hit)
    if name_pats:
        for dirpath, dirnames, filenames in os.walk(cwd):
            keep = []
            for d in dirnames:
                if any(fnmatch.fnmatch(d, p) for p in name_pats):
                    add(Path(dirpath) / d)
                elif d not in _WALK_PRUNE:
                    keep.append(d)
            dirnames[:] = keep
            for f in filenames:
                if any(fnmatch.fnmatch(f, p) for p in name_pats):
                    add(Path(dirpath) / f)
    return sorted(found)


def parse_size(value: str | int) -> int:
    """``"100m"`` / ``"2g"`` / ``"512k"`` / ``4096`` -> bytes (binary multiples)."""
    if isinstance(value, bool):
        raise ValueError(f"not a size: {value!r}")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"size must be positive, got {value}")
        return value
    match = re.fullmatch(r"\s*(\d+)\s*([kmg]?)b?\s*", str(value).lower())
    if not match or int(match.group(1)) == 0:
        raise ValueError(f"not a size (use e.g. '100m'): {value!r}")
    unit = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[match.group(2)]
    return int(match.group(1)) * unit


def _walk(root: Path) -> Iterator[tuple[Path, os.stat_result]]:
    """Every entry under ``root`` (not following symlinks), with its ``lstat``."""
    try:
        entries = list(os.scandir(root))
    except OSError:
        return
    for entry in entries:
        try:
            st = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        path = Path(entry.path)
        yield path, st
        if stat.S_ISDIR(st.st_mode):
            yield from _walk(path)


def tree_bytes(roots: Sequence[Path]) -> int:
    """Total apparent size of the regular files under ``roots``."""
    return sum(
        st.st_size
        for root in roots
        for _, st in _walk(root)
        if stat.S_ISREG(st.st_mode)
    )


def enforce_writable_policy(
    roots: Sequence[Path], *, max_bytes: int | None, started: float
) -> list[str]:
    """Clean up what sandboxed code left in the host-visible writable paths.

    The writable paths are the one channel from untrusted code to the host, and
    host-side tools read them afterwards, so this runs after every call:

    1. Anything that is not a regular file or directory (symlink, FIFO, socket,
       device) is removed. A FIFO would hang a host-side read; a symlink can
       point where the sandbox could not.
    2. If the total is over ``max_bytes``, files written during this call
       (``mtime >= started``) are deleted, largest first, until it fits. Files
       from earlier calls are never deleted for being over.

    Returns human-readable notes for what was removed (empty if nothing).
    """
    notes: list[str] = []
    for root in roots:
        for path, st in list(_walk(root)):
            if not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
                try:
                    path.unlink()
                    notes.append(f"removed non-regular entry {_rel(path, roots)}")
                except OSError:
                    pass
    if max_bytes is None:
        return notes
    total = tree_bytes(roots)
    if total <= max_bytes:
        return notes
    fresh = sorted(
        (
            (st.st_size, path)
            for root in roots
            for path, st in _walk(root)
            if stat.S_ISREG(st.st_mode) and st.st_mtime >= started
        ),
        reverse=True,
    )
    removed = []
    for size, path in fresh:
        if total <= max_bytes:
            break
        try:
            path.unlink()
        except OSError:
            continue
        total -= size
        removed.append(_rel(path, roots))
    if removed:
        notes.append(
            f"output cap of {max_bytes} bytes exceeded; deleted files written by "
            f"this run: {', '.join(removed)}"
        )
    return notes


def _rel(path: Path, roots: Sequence[Path]) -> str:
    for root in roots:
        try:
            return str(path.relative_to(root.parent))
        except ValueError:
            continue
    return path.name


# --- container lifecycle: labels, reaping, cleanup -----------------------------------
#
# Every container the executor creates carries these labels. `exec` mode keeps one
# `sleep infinity` container per executor for the life of the process; if the
# process ends without removing it (restart, crash, Docker/host restart) the husk
# used to stay forever. Two layers now keep that bounded: `close()` + an `atexit`
# hook remove what this process made, and `reap_stale_containers()` clears what a
# process that could not clean up left behind.
LABEL = "paf.sandbox"
_LABEL_KIND = "paf.sandbox.kind"
_LABEL_PID = "paf.sandbox.pid"
_LEGACY_NAME_PREFIX = "paf-sandbox-"
_REAP_RUNNING_KINDS = frozenset({"exec", "run"})

_owned: set[str] = set()
"""Names of long-lived (exec-mode) containers created by this process."""
_reaped = False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def select_reapable(
    rows: Sequence[tuple[str, str, str, str]],
    is_alive: Callable[[int], bool] = _pid_alive,
) -> list[str]:
    """Which ``(id, state, kind, pid)`` rows are safe to remove.

    - An exited or dead container is a husk only once nobody owns it. A finished
      background job keeps its logs until the process that started it reads them
      and calls ``stop()``; a sweep from a *different* process would delete output
      its owner has not read yet. Remove it when its owner pid is gone, or when it
      carries no label at all (pre-label ``paf-sandbox-*`` husks).
    - A ``created`` container is usually mid-start: ``docker run -d`` shows that
      state until the container is running. Remove it only when its owner pid is
      gone (a crash between create and start); a live owner is about to use it.
    - A RUNNING ``exec``/``run`` container whose owner pid is gone was orphaned by
      a crash: remove it. A live owner (this process or another, e.g. the CLI next
      to ``paf-sse``) is never touched; neither is a running container with no
      readable owner, nor a running background (``bg``) one, which is allowed to
      outlive its starter.
    """
    reap: list[str] = []
    for cid, state, kind, pid_text in rows:
        if state in ("exited", "dead"):
            try:
                owner_alive = is_alive(int(pid_text))
            except ValueError:
                owner_alive = False  # unlabeled legacy husk: nobody owns it
            if not owner_alive:
                reap.append(cid)
        elif state == "created":
            try:
                owner_alive = is_alive(int(pid_text))
            except ValueError:
                owner_alive = False  # no readable owner: nobody will start it
            if not owner_alive:
                reap.append(cid)
        elif state == "running" and kind in _REAP_RUNNING_KINDS:
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            if not is_alive(pid):
                reap.append(cid)
    return reap


def reap_stale_containers(is_alive: Callable[[int], bool] = _pid_alive) -> list[str]:
    """Remove sandbox containers nobody owns any more; returns the removed ids.

    Best effort: any Docker failure returns quietly, because housekeeping must
    never stop a command from running. Also removes exited containers from before
    labels existed (``paf-sandbox-*`` names).
    """
    template = (
        '{{.ID}}\t{{.State}}\t{{.Label "' + _LABEL_KIND + '"}}\t'
        '{{.Label "' + _LABEL_PID + '"}}'
    )
    rows: dict[str, tuple[str, str, str, str]] = {}
    queries = (
        ["--filter", f"label={LABEL}=1"],
        ["--filter", f"name={_LEGACY_NAME_PREFIX}", "--filter", "status=exited"],
    )
    try:
        for flt in queries:
            out = subprocess.run(
                ["docker", "ps", "-a", *flt, "--format", template],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            ).stdout
            for line in out.splitlines():
                parts = (line.split("\t") + ["", "", ""])[:4]
                rows.setdefault(parts[0], (parts[0], parts[1], parts[2], parts[3]))
        doomed = select_reapable(list(rows.values()), is_alive)
        if doomed:
            subprocess.run(
                ["docker", "rm", "-f", *doomed],
                capture_output=True,
                timeout=60,
                check=False,
            )
        return doomed
    except (OSError, subprocess.SubprocessError):
        return []


def _close_owned_sync() -> None:
    """``atexit``: remove the long-lived containers this process created."""
    for name in list(_owned):
        try:
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        _owned.discard(name)


atexit.register(_close_owned_sync)


async def _reap_once() -> None:
    """Reap stale containers the first time this process needs Docker."""
    global _reaped
    if _reaped:
        return
    _reaped = True
    await run_sync(reap_stale_containers)


def docker_available() -> bool:
    """Whether the Docker CLI and daemon are reachable on this host.

    Used at build time to hard-fail when a spec configures the ``docker``
    backend but Docker is not actually available, rather than silently
    falling back to an unsandboxed local process.
    """
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
        return True
    except (subprocess.SubprocessError, OSError):
        return False


@dataclass
class _DockerBackgroundState:
    """State for a background process: the detached container's name/id."""

    container_id: str
    output_path: Path | None = None
    overlay_root: Path | None = None
    """Temp dir holding the secret-file overlays; removed on ``stop``."""


def _save_logs(state: _DockerBackgroundState, out: str, err: str) -> tuple[str, str]:
    """Mirror the container's logs into the caller-owned output file (combined),
    returning the (stdout, stderr) to report — merged when there is a file."""
    if state.output_path is None:
        return out, err
    text = out + (err if not out or out.endswith("\n") else "\n" + err)
    state.output_path.parent.mkdir(parents=True, exist_ok=True)
    state.output_path.write_text(text, encoding="utf-8")
    return text, ""


def default_sandbox_user() -> str:
    """The container user: the host's own ``uid:gid``, never root.

    The workspace is bind-mounted with its host permissions, so a fixed uid only
    works for the host user who happens to have it: anyone else gets a workspace
    the container cannot read. Matching the host user keeps the files readable
    without loosening their permissions. A root host (or a platform without
    ``os.getuid``) falls back to ``1000:1000`` so the container is never root.
    """
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None or getuid() == 0:
        return "1000:1000"
    return f"{getuid()}:{getgid()}"


class DockerExecutor:
    """Runs commands inside a locked-down Docker container."""

    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        mode: str = "exec",
        network: str = "none",
        user: str | None = None,
        read_only: bool = True,
        memory: str = "1g",
        cpus: str = "1.0",
        pids_limit: int = 256,
        workspace: str = "/workspace",
        container_name: str | None = None,
        exclude: Sequence[str] = (),
        workspace_readonly: bool = False,
        writable: Sequence[str] = (),
        tmpfs: Sequence[str] = (),
        max_writable_bytes: int | None = None,
    ) -> None:
        if mode not in ("exec", "run"):
            raise ValueError("mode must be 'exec' or 'run'")
        self._image = image
        self._mode = mode
        self._network = network
        self._user = user or default_sandbox_user()
        self._read_only = read_only
        self._memory = memory
        self._cpus = cpus
        self._pids_limit = pids_limit
        self._workspace = workspace
        self._container_name = container_name
        self._exclude = list(exclude)
        self._workspace_readonly = workspace_readonly
        self._writable = [self._clean_relative(w) for w in writable]
        self._tmpfs = list(tmpfs)
        if max_writable_bytes is not None:
            if mode != "run":
                raise ValueError(
                    "max_writable_bytes needs mode='run': enforcing it kills the "
                    "container, which in 'exec' mode is shared by every command."
                )
            if max_writable_bytes <= 0:
                raise ValueError("max_writable_bytes must be positive")
        self._max_writable_bytes = max_writable_bytes
        self._container_id: str | None = None
        self._exec_overlay_root: Path | None = None

    @staticmethod
    def _clean_relative(path: str) -> str:
        """Validate a workspace-relative path (no absolute, no ``..``)."""
        p = Path(path)
        if p.is_absolute() or ".." in p.parts or not p.parts:
            raise ValueError(
                f"writable path must be workspace-relative without '..', got {path!r}"
            )
        return str(p)

    def _workspace_args(self, cwd: Path) -> tuple[list[str], Path | None]:
        """``-v``/``-w`` args for the workspace mount plus its overlays.

        With ``workspace_readonly`` the whole workspace is mounted ``:ro`` and
        only the ``writable`` subpaths are re-mounted read-write on top, so the
        container can write exactly where the spec allows and nowhere else.
        Secret-file overlays are mounted last so they shadow either mode. Returns
        the args and the overlay temp dir (``None`` if nothing is excluded).
        """
        mode = ":ro" if self._workspace_readonly else ""
        args = ["-v", f"{cwd}:{self._workspace}{mode}", "-w", self._workspace]
        for rel in self._writable:
            host = cwd / rel
            host.mkdir(parents=True, exist_ok=True)
            args.extend(["-v", f"{host}:{self._workspace}/{rel}"])
        mounts, overlay_root = self._overlay_mounts(cwd)
        args.extend(mounts)
        return args, overlay_root

    def _overlay_mounts(self, cwd: Path) -> tuple[list[str], Path | None]:
        """Build ``-v`` args that shadow each excluded path with an empty file/dir.

        Excluded paths are found by :func:`find_excluded` (name patterns match at
        any depth). Each is shadowed by an empty file (or empty dir, for
        directories like ``.git``) mounted over its container path. The real host
        file is never touched: reads inside the container see the empty overlay,
        writes land on the overlay, and the host file stays intact.

        Returns the mounts and the temp directory holding the overlays; the
        caller removes it once the container no longer needs it.

        The host Docker socket is never mounted, even if a pattern somehow
        resolves to it - a sandboxed ``docker`` command must not reach the host
        daemon.
        """
        if not self._exclude:
            return [], None
        # Resolve the socket once: on many systems /var/run is a symlink to /run,
        # so a realpath comparison must be against the resolved socket path.
        host_socket = os.path.realpath(_HOST_DOCKER_SOCKET)
        overlay_root = Path(tempfile.mkdtemp(prefix="paf-overlay-"))
        mounts: list[str] = []

        def refuse_socket(link: Path, target: Path) -> None:
            if os.path.realpath(target) == host_socket:
                raise RuntimeError(
                    "Refusing to mount the host Docker socket into the sandbox."
                )

        try:
            excluded = find_excluded(cwd, self._exclude, refuse_socket)
        except RuntimeError:
            shutil.rmtree(overlay_root, ignore_errors=True)
            raise
        for host_path in excluded:
            if os.path.realpath(host_path) == host_socket:
                shutil.rmtree(overlay_root, ignore_errors=True)
                raise RuntimeError(
                    "Refusing to mount the host Docker socket into the sandbox."
                )
            rel = host_path.relative_to(cwd)
            overlay_path = overlay_root / rel
            if host_path.is_dir():
                overlay_path.mkdir(parents=True, exist_ok=True)
            else:
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                overlay_path.touch()
            mounts.extend(["-v", f"{overlay_path}:{self._workspace}/{rel}"])
        return mounts, overlay_root

    def _base_run_args(self, kind: str) -> list[str]:
        """Flags shared by every container; ``kind`` is exec | run | bg."""
        args = [
            "--label",
            f"{LABEL}=1",
            "--label",
            f"{_LABEL_KIND}={kind}",
            "--label",
            f"{_LABEL_PID}={os.getpid()}",
            "--network",
            self._network,
            "--user",
            self._user,
            "--memory",
            self._memory,
            "--cpus",
            self._cpus,
            "--pids-limit",
            str(self._pids_limit),
        ]
        if self._read_only:
            args.append("--read-only")
        for mount in self._tmpfs:
            args.extend(["--tmpfs", mount])
        return args

    async def _docker(
        self, args: list[str], *, timeout: float = 60.0
    ) -> tuple[int, str, str]:
        """Run a docker CLI command and return (exit_code, stdout, stderr)."""
        proc = await anyio.open_process(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        try:
            assert proc.stdout is not None
            assert proc.stderr is not None

            async def _read_stdout() -> None:
                assert proc.stdout is not None
                async for chunk in proc.stdout:
                    stdout_chunks.append(chunk)

            async def _read_stderr() -> None:
                assert proc.stderr is not None
                async for chunk in proc.stderr:
                    stderr_chunks.append(chunk)

            with anyio.fail_after(timeout):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(_read_stdout)
                    tg.start_soon(_read_stderr)
                await proc.wait()
        finally:
            await proc.aclose()

        stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        exit_code = proc.returncode if proc.returncode is not None else 0
        return exit_code, stdout, stderr

    async def _ensure_container(self, cwd: Path) -> None:
        """Create the long-lived container for ``mode='exec'`` on first use."""
        if self._container_id is not None:
            return
        await _reap_once()
        name = self._container_name or f"paf-sandbox-{uuid.uuid4().hex[:8]}"
        workspace_args, overlay_root = self._workspace_args(cwd)
        args = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            *self._base_run_args("exec"),
            *workspace_args,
            self._image,
            "sleep",
            "infinity",
        ]
        code, out, err = await self._docker(args)
        if code != 0:
            if overlay_root is not None:
                shutil.rmtree(overlay_root, ignore_errors=True)
            raise RuntimeError(
                f"Failed to start sandbox container: {err.strip() or out.strip()}"
            )
        self._container_id = out.strip()
        self._container_name = name
        # The long-lived container keeps its overlays mounted for as long as it
        # lives, so the temp dir is removed by `close()`, not here.
        self._exec_overlay_root = overlay_root
        _owned.add(name)

    async def close(self) -> None:
        """Remove this executor's long-lived (exec-mode) container, if any.

        Safe to call more than once and when nothing was started. Per-command
        (``run``) containers remove themselves; background ones are removed by
        ``stop``.
        """
        name = self._container_name if self._container_id is not None else None
        self._container_id = None
        if name is not None:
            await self._docker(["docker", "rm", "-f", name], timeout=30.0)
            _owned.discard(name)
        if self._exec_overlay_root is not None:
            shutil.rmtree(self._exec_overlay_root, ignore_errors=True)
            self._exec_overlay_root = None

    async def run(
        self,
        command: str,
        *,
        cwd: Path,
        env: dict[str, str] | None,
        timeout: float,
    ) -> ExecResult:
        overlays: list[Path] = []
        try:
            return await self._run(
                command, cwd=cwd, env=env, timeout=timeout, overlays=overlays
            )
        finally:
            for root in overlays:
                shutil.rmtree(root, ignore_errors=True)

    async def _run(
        self,
        command: str,
        *,
        cwd: Path,
        env: dict[str, str] | None,
        timeout: float,
        overlays: list[Path],
    ) -> ExecResult:
        run_name: str | None = None
        await _reap_once()
        if self._mode == "exec":
            await self._ensure_container(cwd)
            args = ["docker", "exec", self._container_id, "sh", "-c", command]
        else:
            run_name = f"paf-run-{uuid.uuid4().hex[:12]}"
            workspace_args, overlay_root = self._workspace_args(cwd)
            if overlay_root is not None:
                overlays.append(overlay_root)
            args = [
                "docker",
                "run",
                "--rm",
                "--name",
                run_name,
                *self._base_run_args("run"),
                *workspace_args,
                self._image,
                "sh",
                "-c",
                command,
            ]
        # Watch the host-visible writable paths for the whole call; the cleanup
        # afterwards also runs for calls that stayed under the cap, to drop
        # symlinks/FIFOs planted by the sandboxed code.
        roots = [cwd / rel for rel in self._writable]
        started = time.time()
        capped = {"hit": False}

        async def watch() -> None:
            assert run_name is not None and self._max_writable_bytes is not None
            while True:
                await anyio.sleep(0.25)
                size = await run_sync(tree_bytes, roots)
                if size > self._max_writable_bytes:
                    capped["hit"] = True
                    await self._docker(["docker", "rm", "-f", run_name], timeout=30.0)
                    return

        timed_out = False
        code, out, err = 124, "", ""
        async with anyio.create_task_group() as tg:
            if roots and run_name is not None and self._max_writable_bytes:
                tg.start_soon(watch)
            try:
                code, out, err = await self._docker(args, timeout=timeout)
            except TimeoutError:
                # Killing the docker CLI client does not stop the container, so a
                # per-command container is removed explicitly. (exec mode shares
                # one long-lived container; callers that need a hard in-container
                # kill wrap the command in ``timeout``.)
                timed_out = True
                if run_name is not None:
                    await self._docker(["docker", "rm", "-f", run_name], timeout=30.0)
            finally:
                tg.cancel_scope.cancel()
        notes = await run_sync(
            lambda: enforce_writable_policy(
                roots, max_bytes=self._max_writable_bytes, started=started
            )
        )
        if capped["hit"]:
            notes.insert(
                0,
                f"sandbox: writable-path size cap of {self._max_writable_bytes} "
                "bytes exceeded; the run was killed",
            )
        if notes:
            err = (err + "\n" if err and not err.endswith("\n") else err) + "".join(
                f"[{n}]\n" for n in notes
            )
        if timed_out:
            return ExecResult(stdout="", stderr=err, exit_code=124, timed_out=True)
        return ExecResult(stdout=out, stderr=err, exit_code=code)

    async def start(
        self,
        command: str,
        *,
        cwd: Path,
        env: dict[str, str] | None,
        output_path: Path | None = None,
    ) -> BackgroundHandle:
        await _reap_once()
        command_id = uuid.uuid4().hex[:12]
        name = f"paf-bg-{command_id}"
        workspace_args, overlay_root = self._workspace_args(cwd)
        args = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            *self._base_run_args("bg"),
            *workspace_args,
            self._image,
            "sh",
            "-c",
            command,
        ]
        code, out, err = await self._docker(args)
        if code != 0:
            if overlay_root is not None:
                shutil.rmtree(overlay_root, ignore_errors=True)
            raise RuntimeError(
                f"Failed to start background container: {err.strip() or out.strip()}"
            )
        state = _DockerBackgroundState(
            container_id=name, output_path=output_path, overlay_root=overlay_root
        )
        return BackgroundHandle(command_id=command_id, state=state)

    async def check(self, handle: BackgroundHandle) -> BackgroundStatus:
        state: _DockerBackgroundState = handle.state
        _, running_out, _ = await self._docker(
            ["docker", "inspect", "-f", "{{.State.Running}}", state.container_id]
        )
        is_running = running_out.strip() == "true"
        _, logs_out, logs_err = await self._docker(
            ["docker", "logs", state.container_id]
        )
        status = "running" if is_running else "finished"
        exit_code: int | None = None
        if not is_running:
            _, code_out, _ = await self._docker(
                ["docker", "inspect", "-f", "{{.State.ExitCode}}", state.container_id]
            )
            try:
                exit_code = int(code_out.strip())
            except ValueError:
                exit_code = None
        logs_out, logs_err = _save_logs(state, logs_out, logs_err)
        return BackgroundStatus(
            status=status, stdout=logs_out, stderr=logs_err, exit_code=exit_code
        )

    async def stop(self, handle: BackgroundHandle) -> BackgroundStatus:
        state: _DockerBackgroundState = handle.state
        await self._docker(["docker", "stop", state.container_id])
        _, logs_out, logs_err = await self._docker(
            ["docker", "logs", state.container_id]
        )
        _, code_out, _ = await self._docker(
            ["docker", "inspect", "-f", "{{.State.ExitCode}}", state.container_id]
        )
        await self._docker(["docker", "rm", state.container_id])
        if state.overlay_root is not None:
            shutil.rmtree(state.overlay_root, ignore_errors=True)
        try:
            exit_code = int(code_out.strip())
        except ValueError:
            exit_code = None
        logs_out, logs_err = _save_logs(state, logs_out, logs_err)
        return BackgroundStatus(
            status="finished", stdout=logs_out, stderr=logs_err, exit_code=exit_code
        )
