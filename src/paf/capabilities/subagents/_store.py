"""Durable subagent task records + markdown output files.

Every delegation (sync or background) gets a task record in an append-only
JSONL, scoped per conversation exactly like shell's ``ShellCommandStore``:
``<processes_root>/<agent>/conversations/<cid>`` (hashed into one segment by the
shared ``FlatJsonlStore``) ``/subagent_tasks.jsonl``. Each line is a complete,
self-contained snapshot of the task; the LAST line per ``task_id`` is its
current state. The JSONL is the only source of truth for status.

A finished task's output is written to ``subagent_outputs/<task_id>.md`` next to
the JSONL, with a small YAML frontmatter (task id, agent, conversation, trace
id, the task text) so a file opened on its own is self-describing. The
frontmatter deliberately has no ``status``. Failed / cancelled / interrupted
tasks get no file — their record carries the ``detail``.

Background agents in paf persist results, not transcripts: the child run's
internals live only in traces (the record keeps the ``trace_id``).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from paf.capabilities._storage import FlatJsonlStore, write_atomic
from paf.capabilities._storage._flat_file import _safe_key_segment

KIND = "subagent_tasks"
OUTPUTS_DIR = "subagent_outputs"

ACTIVE = frozenset({"queued", "running"})
TERMINAL = frozenset({"done", "error", "cancelled", "interrupted"})

# Store paths already reconciled in THIS process. Reconciliation flips rows left
# active by a previous process; doing it once per path per process means it can
# never touch a task this process started (that task used the store after).
_reconciled: set[Path] = set()


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _safe_filename(task_id: str) -> str:
    """A filename-safe form of ``task_id`` (the cid part may contain ``:``)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", task_id)


class SubagentTaskStore:
    """Task records + output files for one agent's conversation."""

    def __init__(
        self,
        processes_root: str | Path,
        agent: str,
        *,
        conversation_id: str | None = None,
    ) -> None:
        # Lazy import: background_processes._store re-enters paf.capabilities.
        from paf.background_processes._store import _safe_cid

        self.agent = agent
        self.conversation_id = conversation_id
        key = (
            f"{agent}/conversations/{_safe_cid(conversation_id)}"
            if conversation_id is not None
            else agent
        )
        self._key = key
        self._jsonl = FlatJsonlStore(processes_root)
        self.dir = Path(processes_root) / _safe_key_segment(key)
        self.outputs_dir = self.dir / OUTPUTS_DIR
        self._reconcile_once()

    # -- reads -----------------------------------------------------------------

    def _lines(self) -> list[dict[str, Any]]:
        return self._jsonl.read_all(KIND, self._key)

    def latest_all(self) -> dict[str, dict[str, Any]]:
        """Current state of every task, keyed by ``task_id`` (last line wins)."""
        latest: dict[str, dict[str, Any]] = {}
        for rec in self._lines():
            tid = rec.get("task_id")
            if tid:
                latest[tid] = rec
        return latest

    def get(self, task_id: str) -> dict[str, Any] | None:
        """The current record for ``task_id``, or ``None`` if unknown."""
        return self.latest_all().get(task_id)

    def read_output(self, task_id: str) -> str | None:
        """The task's output (frontmatter stripped), or ``None`` if no file."""
        rec = self.get(task_id)
        path = rec.get("output_path") if rec else None
        if not path or not Path(path).exists():
            return None
        return _strip_frontmatter(Path(path).read_text(encoding="utf-8"))

    # -- writes ----------------------------------------------------------------

    def next_task_id(self) -> str:
        """Next ``subagent.<cid>.<seq>`` id, unique across turns and restarts."""
        cid = self.conversation_id or "global"
        prefix = f"subagent.{cid}."
        seq = 0
        for tid in self.latest_all():
            tail = tid[len(prefix) :] if tid.startswith(prefix) else ""
            if tail.isdigit():
                seq = max(seq, int(tail))
        return f"{prefix}{seq + 1}"

    def create(
        self,
        *,
        agent_name: str,
        task: str,
        mode: str,
        parent_run_id: str | None,
        status: str = "queued",
    ) -> dict[str, Any]:
        """Append the first line for a new task and return it."""
        rec = {
            "task_id": self.next_task_id(),
            "agent": agent_name,
            "mode": mode,
            "task": task,
            "parent_run_id": parent_run_id,
            "conversation_id": self.conversation_id,
            "status": status,
            "detail": "",
            "trace_id": None,
            "usage": None,
            "output_path": None,
            "output_chars": None,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        self._jsonl.append(KIND, self._key, rec)
        return rec

    def update(self, task_id: str, **fields: Any) -> dict[str, Any]:
        """Append a new full snapshot of ``task_id`` with ``fields`` changed."""
        current = self.get(task_id)
        if current is None:
            raise KeyError(task_id)
        rec = {**current, **fields, "updated_at": _now_iso()}
        self._jsonl.append(KIND, self._key, rec)
        return rec

    def mark_done(
        self, task_id: str, *, output: str, usage: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Write the output file, then record the task as ``done``."""
        current = self.get(task_id)
        if current is None:
            raise KeyError(task_id)
        path = self.outputs_dir / f"{_safe_filename(task_id)}.md"
        write_atomic(path, _render_output(current, output))
        return self.update(
            task_id,
            status="done",
            detail="",
            usage=usage,
            output_path=str(path),
            output_chars=len(output),
        )

    def _reconcile_once(self) -> None:
        """Mark tasks left active by a previous process as ``interrupted``.

        Background children are in-process asyncio tasks, so a server restart
        kills them mid-run. Their last line still says ``queued``/``running``;
        flip it (append-only) so they never look alive forever. No re-run:
        a delegation is not idempotent.
        """
        path = self.dir / f"{KIND}.jsonl"
        if path in _reconciled:
            return
        _reconciled.add(path)
        for tid, rec in self.latest_all().items():
            if rec.get("status") in ACTIVE:
                self.update(tid, status="interrupted", detail="server restarted")


def _render_output(rec: dict[str, Any], output: str) -> str:
    """Frontmatter (no status) + the leaf's output verbatim."""
    meta = {
        "task_id": rec.get("task_id"),
        "agent": rec.get("agent"),
        "mode": rec.get("mode"),
        "created": rec.get("created_at"),
        "conversation_id": rec.get("conversation_id"),
        "parent_run_id": rec.get("parent_run_id"),
        "trace_id": rec.get("trace_id"),
        "task": rec.get("task"),
    }
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=100)
    return f"---\n{front}---\n\n{output}"


def _strip_frontmatter(text: str) -> str:
    """Drop a leading ``---`` YAML block (and the blank line after it)."""
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    return text[end + len("\n---\n") :].removeprefix("\n")
