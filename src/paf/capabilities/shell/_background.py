"""Standalone background queue that runs shell ``start_command`` processes.

A generic :class:`BackgroundQueueService` subclass (NOT the memory-specific
``BackgroundCoordinator``) that backgrounds *any* shell command. ``start_command``
enqueues a :class:`ShellRun` and returns its ``run_id`` immediately — the worker
task starts at once (``enqueue`` spawns it), no turn-end / cadence / ``after_turn``.

Execution model:

- The worker calls :meth:`SandboxedExecutor.start` to spawn a DETACHED subprocess
  (own session / process-group, so it survives its parent turn and loop).
- It records the ``pid`` in the RICH ``ShellCommandStore`` record and a BASIC
  ``status`` record in the coordinator's ``RunEventStore`` (``runs.jsonl`` — the
  one the UI watches). ``run_id`` is the join key between the two.
- While the session's event loop is alive the worker polls :meth:`check` and
  publishes ``running -> done|error`` transitions. When the loop closes the
  worker task dies, but the DETACHED subprocess keeps running; a later session /
  fresh toolset re-attaches by ``pid`` from the durable store.

``notify_on_complete`` reuses the existing ``BackgroundTask`` -> ``BackgroundEvent``
(``done``) feed machinery via an optional ``on_complete`` callback.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from paf.background import BackgroundQueueService, QueueState
from paf.background_processes._events import (
    ProcessEvent,
    ProcessEventType,
    get_agent_emitter,
)
from paf.capabilities._sandbox import BackgroundHandle, SandboxedExecutor
from paf.capabilities.shell._store import ShellCommandStore

log = logging.getLogger(__name__)

#: Seconds between status polls while a background process is running.
_POLL_S: float = 0.25


@dataclass(frozen=True)
class ShellRun:
    """A request to run one shell command in the background."""

    run_id: str
    command: str
    conversation_id: str | None = None
    cwd: str | None = None  # explicit cwd override; default = queue's cwd
    env: dict[str, str] | None = None
    notify_on_completion: bool = False


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _pid_of(handle: BackgroundHandle) -> int | None:
    """Best-effort pid from a background handle (opaque to callers).

    The handle's backend-specific ``state`` carries the process; we read the
    pid without coupling to any one backend. Returns ``None`` when unavailable
    (caller then relies on the tempfile prefix / run_id to re-attach).
    """
    state = getattr(handle, "state", None)
    proc = getattr(state, "proc", None)
    pid = getattr(proc, "pid", None)
    return int(pid) if pid is not None else None


class ShellBackgroundQueue(BackgroundQueueService[ShellRun]):
    """FIFO queue that runs shell commands as detached background subprocesses.

    Mirrors :class:`paf.capabilities.memory._background.MemoryConsolidationQueue`
    structurally: a generic queue whose :meth:`process_job` does the real work.
    The executor is injected so the execution boundary stays swappable (local
    now, Docker later) exactly as the shell toolset's sandbox seam allows.
    """

    name = "shell-background"

    def __init__(
        self,
        *,
        executor: SandboxedExecutor,
        cwd: str | Path,
        agent: str,
        shell_cmds,  # ShellCommandStore
        on_complete=None,  # Callable[[str, str, bool, str], Awaitable[None]] (run_id, conversation_id, ok, detail)
        name: str = "shell-background",
    ) -> None:
        super().__init__()
        self.executor = executor
        self.cwd = Path(cwd).resolve() if cwd else None
        self.agent = agent
        self.shell_cmds = shell_cmds
        self.on_complete = on_complete
        self.name = name
        # The per-agent live event emitter (shared with the coordinator's task
        # bus and the SSE feed). Shell emits a ProcessEvent on each transition.
        self._emitter = get_agent_emitter(agent)

    def _record_rich(self, job: ShellRun, *, status: str, **fields: object) -> None:
        """Append one rich shell journal line (join key = run_id).

        ``status`` is REQUIRED and stamped per line with a single ``timestamp``;
        never mutate a prior line. Each line is self-contained — ``command`` /
        ``agent`` / ``conversation_id`` are repeated on every line so a later
        line stands alone regardless of what preceded it.
        """
        self.shell_cmds.append(
            {
                "run_id": job.run_id,
                "command": job.command,
                "agent": self.agent,
                "conversation_id": job.conversation_id,
                "timestamp": _now_iso(),
                "status": status,
                **fields,
            }
        )

    async def _emit(
        self, run_id: str, status: str, label: str, detail: str = ""
    ) -> None:
        """Push one live process event to the per-agent emitter.

        Shell processes are conversation-scoped by origin, so the event carries
        ``lane="conversation"`` and the run's conversation id as ``scope``; the
        SSE feed filters on these to route the row to the right panel.
        """
        await self._emitter.emit_process(
            ProcessEvent(
                id=run_id,
                label=label,
                status=ProcessEventType(status),
                detail=detail,
                lane="conversation",
                scope=self._job_cid.get(run_id),
            )
        )

    _job_cid: dict[str, str | None] = {}
    """run_id -> conversation_id for emitted events (the emitter has no job object)."""

    async def shutdown(self) -> None:
        """Drop per-session queue state; the worker keeps running in the runner.

        The worker task now lives in the process-level :class:`BackgroundRunner`
        (spawned by ``enqueue``), so it is NOT cancelled here — it keeps
        supervising the detached process to completion and is only stopped on
        server exit via the runner. We just release the per-session state map.
        """
        self._states.clear()

    async def _notify(self, job: ShellRun, ok: bool, detail: str) -> None:
        """Fire the ``on_complete`` callback best-effort (never breaks the worker).

        The callback runs a full agent turn (the notify-on-completion message),
        so a failure there must not mask the terminal-status recording or the
        original exception in :meth:`process_job`.
        """
        if self.on_complete is None:
            return
        try:
            await self.on_complete(job.run_id, job.conversation_id, ok, detail)
        except Exception:  # noqa: BLE001 - notify is best-effort
            log.exception("notify_on_complete failed for %s", job.run_id)

    async def process_job(
        self,
        session_id: str,
        job: ShellRun,
        state: QueueState[ShellRun],
    ) -> None:
        del state
        try:
            self._job_cid[job.run_id] = job.conversation_id
            # The basic "running" row is already published synchronously by
            # start_command (with a richer "started: <cmd>" detail); don't
            # append a second one here.
            cwd = Path(job.cwd).resolve() if job.cwd else self.cwd or Path(".")
            env = job.env
            handle = await self.executor.start(job.command, cwd=cwd, env=env)
            pid = _pid_of(handle)
            self._record_rich(
                job,
                status="running",
                pid=pid,
                cwd=str(cwd),
            )

            # Supervise while this session's loop is alive. A long-running
            # detached process keeps going even if the loop closes; a later
            # session re-attaches by pid.
            while True:
                await asyncio.sleep(_POLL_S)
                status = await self.executor.check(handle)
                if status.status == "finished":
                    self._record_rich(
                        job,
                        status="done" if status.exit_code == 0 else "error",
                        exit_code=status.exit_code,
                    )
                    await self._emit(
                        job.run_id,
                        "done" if status.exit_code == 0 else "error",
                        f"shell: {job.command}",
                        _detail(status.stdout, status.stderr, status.exit_code),
                    )
                    if job.notify_on_completion and self.on_complete is not None:
                        await self._notify(
                            job, status.exit_code == 0, f"exit {status.exit_code}"
                        )
                    return
        except Exception as exc:
            self._record_rich(job, status="error")
            await self._emit(job.run_id, "error", f"shell: {job.command}", str(exc))
            if job.notify_on_completion and self.on_complete is not None:
                await self._notify(job, False, str(exc))
            raise


def _detail(stdout: str, stderr: str, exit_code: int | None) -> str:
    """Short feed detail: exit code plus a tail of any output."""
    snippet = (stdout or stderr).strip().splitlines()
    tail = snippet[-1][:80] if snippet else ""
    if exit_code is None:
        return "finished"
    return f"exit {exit_code}" + (f": {tail}" if tail else "")


# --- Reconnect restore -------------------------------------------------------
# Shell processes live in the durable shell store (``shell_cmds.jsonl``), NOT the
# coordinator's ``task_history``, so the SSE process feed's on-connect restore
# must read them here to re-show them after a refresh / conversation switch.
# Restore active (running) rows plus terminal rows finished within a short
# window; older terminal rows are dropped so the panel never accumulates every
# shell process a conversation has ever run.

_RESTORE_RECENT_WINDOW_S = 15 * 60  # terminal rows restored if finished within this


def conversation_shell_frames(
    processes_root: str | Path,
    agent: str,
    scoped_cid: str,
    *,
    now: datetime | None = None,
    recent_window_s: int = _RESTORE_RECENT_WINDOW_S,
) -> list[dict[str, object]]:
    """Current shell background processes for a conversation, as task frames.

    Reads the durable shell store (``shell_cmds.jsonl``) for ``scoped_cid`` and
    returns one row per process in the same shape the live feed publishes
    (``id`` / ``label`` / ``status`` / ``progress`` / ``detail``), so a reconnect
    can re-show them. A process's current state is the LAST journal line for its
    ``run_id`` (append order is chronological). Active ``running`` rows are
    always returned; terminal rows (``done``/``error``) are returned only when
    finished within ``recent_window_s`` (default 15 min) so old runs fall off.
    """
    now = now or datetime.now(UTC)
    store = ShellCommandStore.open(
        str(processes_root), agent, conversation_id=scoped_cid
    )
    latest: dict[str, dict[str, object]] = {}
    for rec in store.read_all():
        rid = rec.get("run_id")
        if not rid:
            continue
        latest[rid] = rec  # last line wins (append order = chronological)

    frames: list[dict[str, object]] = []
    for rid, rec in latest.items():
        status = rec.get("status")
        command = str(rec.get("command") or rid)
        if status == "running":
            frames.append(
                {
                    "id": rid,
                    "label": f"shell: {command}",
                    "status": "running",
                    "progress": None,
                    "detail": f"started: {command}",
                }
            )
        elif status in ("done", "error"):
            if _finished_within(rec, now, recent_window_s):
                exit_code = rec.get("exit_code")
                detail = f"exit {exit_code}" if exit_code is not None else str(status)
                frames.append(
                    {
                        "id": rid,
                        "label": f"shell: {command}",
                        "status": str(status),
                        "progress": None,
                        "detail": detail,
                    }
                )
    return frames


def _finished_within(rec: dict[str, object], now: datetime, window_s: int) -> bool:
    """True when the record's ``timestamp`` falls inside the recency window."""
    ts = rec.get("timestamp")
    if not ts:
        return False
    try:
        finished = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return False
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=UTC)
    return 0 <= (now - finished).total_seconds() <= window_s
