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
from paf.capabilities._sandbox import BackgroundHandle, SandboxedExecutor
from paf.capabilities.shell._events import ProcessEvent, ProcessEventType
from paf.capabilities.shell._registry import get_process_emitter

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
        on_complete=None,  # Callable[[str, bool, str], None] (session_id, ok, detail)
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
        self._emitter = get_process_emitter(agent)

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
                event_type=ProcessEventType(status),
                id=run_id,
                label=label,
                status=status,
                detail=detail,
                lane="conversation",
                scope=self._job_cid.get(run_id),
                updated_at=_now_iso(),
            )
        )

    _job_cid: dict[str, str | None] = {}
    """run_id -> conversation_id for emitted events (the emitter has no job object)."""

    async def shutdown(self) -> None:
        """Cancel all live worker tasks (detached processes keep running).

        Called on toolset/loop teardown so lingering supervisor tasks don't
        hold the subprocess transports past loop close (which otherwise raises
        ``RuntimeError: Event loop is closed`` on GC). The DETACHED OS process
        is unaffected — it keeps running and a later session re-attaches by pid.
        """
        pending: list[asyncio.Task] = []
        for session_id, state in list(self._states.items()):
            task = getattr(state, "worker_task", None)
            if task is not None and not task.done():
                task.cancel()
                pending.append(task)
            if session_id in self._states:
                del self._states[session_id]
        # Await the cancellations so each worker fully unwinds (releasing its
        # subprocess transport) before the caller closes the loop.
        for task in pending:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

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
                    if self.on_complete is not None:
                        self.on_complete(
                            session_id,
                            status.exit_code == 0,
                            f"exit {status.exit_code}",
                        )
                    return
        except Exception as exc:
            self._record_rich(job, status="error")
            await self._emit(job.run_id, "error", f"shell: {job.command}", str(exc))
            if self.on_complete is not None:
                self.on_complete(session_id, False, str(exc))
            raise


def _detail(stdout: str, stderr: str, exit_code: int | None) -> str:
    """Short feed detail: exit code plus a tail of any output."""
    snippet = (stdout or stderr).strip().splitlines()
    tail = snippet[-1][:80] if snippet else ""
    if exit_code is None:
        return "finished"
    return f"exit {exit_code}" + (f": {tail}" if tail else "")
