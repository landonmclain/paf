"""Scheduled-task store: master list (``scheduled_jobs.json``) + run history
(``scheduled_runs.jsonl``).

A thin typed wrapper over the shared storage seams. The master list rides
:class:`StorageService` (whole-artifact JSON, atomic writes) and the run history
rides :class:`AppendOnlyStore` (append-only JSONL) — the two opposite persistence
shapes the spec calls for (§4.8). Both are scoped per-agent by ``key`` (one flat
table per agent, like a database); the job/run records carry ``agent`` +
``conversation_id`` as columns.

The store is **synchronous**, like the other paf stores (``ShellCommandStore``,
the memory journal, the background-process store): it is a thin wrapper over the
sync storage seams, and the async layer lives at the call site (the scheduler /
toolset), not here.

Layout::

    <root>/<agent>/scheduled_jobs.json     # master list — CRUD, soft-delete
    <root>/<agent>/scheduled_runs.jsonl    # run history — append-only

An optional ``on_change`` callback is invoked after every mutation, so the
scheduler can wake on a store change instead of polling.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from paf.capabilities._storage._flat_file import FlatFileStorage
from paf.capabilities._storage._jsonl_file import FlatJsonlStore
from paf.capabilities._storage._protocol import AppendOnlyStore, StorageService

from ._types import NotifyMode, RunRecord, ScheduledJob

_JOBS_KIND = "scheduled_jobs"
_RUNS_KIND = "scheduled_runs"


class ScheduledTaskStore:
    """Master list + run history for one agent's scheduled tasks.

    Composes a :class:`StorageService` (master list) and an
    :class:`AppendOnlyStore` (run history) under the same ``key`` (the agent
    name). Synchronous, like the other paf stores.
    """

    def __init__(
        self,
        root: str | Path = "data/scheduled_tasks",
        *,
        agent: str = "default",
        storage: StorageService | None = None,
        runs: AppendOnlyStore | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._key = agent
        # Default backends are the flat-file seams; alternates can be dropped in
        # (tests / future SQLite) via `storage` / `runs`.
        self._storage = storage if storage is not None else FlatFileStorage(Path(root))
        self._runs = runs if runs is not None else FlatJsonlStore(Path(root))
        self._on_change = on_change

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ScheduledTaskStore(key={self._key!r})"

    def _notify_change(self) -> None:
        if self._on_change is not None:
            self._on_change()

    # --- master list (scheduled_jobs.json) --------------------------------

    def _load_jobs(self) -> list[ScheduledJob]:
        payload = self._storage.read(_JOBS_KIND, self._key)
        if payload is None:
            return []
        rows = (
            payload.get("jobs", [])
            if isinstance(payload, dict)
            else cast(list[Any], payload)
        )
        return [ScheduledJob.model_validate(row) for row in rows]

    def _save_jobs(self, jobs: list[ScheduledJob]) -> None:
        self._storage.write(
            _JOBS_KIND,
            self._key,
            {"jobs": [job.model_dump(mode="json") for job in jobs]},
        )

    def create_job(self, job: ScheduledJob) -> ScheduledJob:
        """Add `job`; returns it. Raises if a job with the same id exists."""
        jobs = self._load_jobs()
        if any(existing.id == job.id for existing in jobs):
            raise ValueError(f"A job with id {job.id!r} is already scheduled.")
        jobs.append(job)
        self._save_jobs(jobs)
        self._notify_change()
        return job

    def get_job(self, job_id: str) -> ScheduledJob | None:
        """Return the job with `job_id`, or `None`."""
        return next((j for j in self._load_jobs() if j.id == job_id), None)

    def list_jobs(self, *, include_archived: bool = False) -> list[ScheduledJob]:
        """Every job in insertion order, archived ones excluded by default."""
        jobs = self._load_jobs()
        if not include_archived:
            jobs = [j for j in jobs if not j.archived]
        return jobs

    def update_job(
        self,
        job_id: str,
        *,
        schedule: str | None = None,
        name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        script: str | None = None,
        notify: NotifyMode | None = None,
        delivery: str | None = None,
        enabled: bool | None = None,
        last_run: datetime | None = None,
    ) -> ScheduledJob | None:
        """Apply the non-`None` fields to `job_id`; return the updated job or `None`.

        Re-validates the result (so e.g. setting `instructions` on a script job
        without clearing `script` raises the exactly-one-of error). `None` means
        "leave unchanged", so switching a job between `instructions`/`script`
        modes is a toolset concern, not handled here.
        """
        jobs = self._load_jobs()
        for index, existing in enumerate(jobs):
            if existing.id != job_id:
                continue
            data = existing.model_dump()
            if schedule is not None:
                data["schedule"] = schedule
            if name is not None:
                data["name"] = name
            if description is not None:
                data["description"] = description
            if instructions is not None:
                data["instructions"] = instructions
            if script is not None:
                data["script"] = script
            if notify is not None:
                data["notify"] = notify
            if delivery is not None:
                data["delivery"] = delivery
            if enabled is not None:
                data["enabled"] = enabled
            if last_run is not None:
                data["last_run"] = last_run
            updated = ScheduledJob.model_validate(data)
            jobs[index] = updated
            self._save_jobs(jobs)
            self._notify_change()
            return updated
        return None

    def set_definition(
        self,
        job_id: str,
        *,
        instructions: str | None = None,
        script: str | None = None,
    ) -> ScheduledJob | None:
        """Set the task definition, clearing the other mode; return the job or None.

        ``update_job`` treats ``None`` as "leave unchanged", so it cannot switch a
        job between ``instructions``/``script`` modes (setting one would leave the
        other set and fail the exactly-one-of validation). This method sets exactly
        the given definition, clearing the other field — the toolset uses
        it for mode switches.
        """
        jobs = self._load_jobs()
        for index, existing in enumerate(jobs):
            if existing.id != job_id:
                continue
            data = existing.model_dump()
            data["instructions"] = instructions
            data["script"] = script
            updated = ScheduledJob.model_validate(data)
            jobs[index] = updated
            self._save_jobs(jobs)
            self._notify_change()
            return updated
        return None

    def remove_job(self, job_id: str) -> bool:
        """Soft-delete `job_id` (set `archived=true`); return whether it existed."""
        jobs = self._load_jobs()
        for index, existing in enumerate(jobs):
            if existing.id == job_id:
                jobs[index] = existing.model_copy(update={"archived": True})
                self._save_jobs(jobs)
                self._notify_change()
                return True
        return False

    # --- run history (scheduled_runs.jsonl) -------------------------------

    def append_run(self, record: RunRecord) -> RunRecord:
        """Append `record` to the run history; returns it."""
        self._runs.append(_RUNS_KIND, self._key, record.model_dump(mode="json"))
        return record

    def read_runs(self) -> list[RunRecord]:
        """All runs in append order."""
        rows = self._runs.read_all(_RUNS_KIND, self._key)
        return [RunRecord.model_validate(row) for row in rows]
