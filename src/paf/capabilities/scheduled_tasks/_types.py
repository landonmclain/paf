"""Scheduled-task data models: :class:`NotifyMode`, :class:`ScheduledJob`, :class:`RunRecord`.

The data model for the scheduled-tasks capability (see the vault spec
``Scheduled-Tasks-Cron-Spec.md``). ``ScheduledJob`` is the master-list record
(``scheduled_jobs.json``); ``RunRecord`` is one append-only run-history row
(``scheduled_runs.jsonl``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class NotifyMode(str, Enum):  # noqa: UP042 - str-Enum keeps values JSON-serializable as plain strings
    """How often the agent is in the loop for a script-executed job.

    The graduated trust ramp (§4.6): a script task starts in ``always`` (the
    agent reviews every run and builds confidence), then flips to ``on_trigger``
    once it trusts the script.
    """

    always = "always"
    on_trigger = "on_trigger"


class ScheduledJob(BaseModel):
    """One scheduled task.

    Exactly one of ``instructions`` / ``script`` is set:

    - ``instructions`` — agent-executed (action tasks): the agent follows the
      instruction block and does the work. The agent is the decision point.
    - ``script`` — script-executed (check/alert tasks): a script runs code-only,
      skipping the agent unless X, with ``notify`` controlling how often the
      agent is in the loop.

    ``id`` and ``created_at`` are **stamped in code** (determinism rule, #5676):
    they are set by the model's default factories at creation, never copied by
    the caller.
    """

    id: str = Field(
        default_factory=lambda: uuid4().hex[:8],
        description="Stable id, stamped in code at creation.",
    )
    agent: str = Field(
        default="",
        description="Which agent owns the job (scoping column; redundant with the folder but explicit).",
    )
    conversation_id: str = Field(
        default="global",
        description="The origin conversation the job runs in (routing column).",
    )
    name: str = Field(
        description="Human-readable name for the task (shown in the UI).",
    )
    description: str = Field(
        description="What the task does / when it runs (shown in the UI).",
    )
    schedule: str = Field(
        description="Cron expression. Parsed by the scheduler via croniter."
    )
    instructions: str | None = Field(
        default=None,
        description="The instruction block; the whole task definition (agent-executed).",
    )
    script: str | None = Field(
        default=None,
        description="A script to run code-only, skipping the agent unless X (script-executed).",
    )
    notify: NotifyMode = Field(
        default=NotifyMode.always,
        description="Graduated trust ramp for script tasks: always (notify every run) "
        "or on_trigger (notify only when X).",
    )
    delivery: str = Field(
        default="file",
        description="Delivery target (default `file`).",
    )
    enabled: bool = Field(
        default=True,
        description="Whether the job is active.",
    )
    archived: bool = Field(
        default=False,
        description="Soft-delete/archive flag; `remove` sets this, not a hard delete.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Stamped in code at creation.",
    )
    last_run: datetime | None = Field(
        default=None,
        description="The only persisted watermark (restart safety); set by the scheduler.",
    )

    @model_validator(mode="after")
    def _exactly_one_of_instructions_script(self) -> ScheduledJob:
        if (self.instructions is None) == (self.script is None):
            raise ValueError("exactly one of `instructions` or `script` must be set")
        return self


class RunRecord(BaseModel):
    """One scheduled run, appended to the run history (``scheduled_runs.jsonl``).

    Mirrors the background-process RunJournal shape (run_id, job_id, agent,
    conversation_id, status, detail, timestamp) so it slots into the existing
    Processes feed without a new schema. The run-time fields (``verdict``,
    ``skip_reason``, ``cost``, ``finished_at``) are not recorded here; the store
    only appends and reads these records.
    """

    run_id: str = Field(
        default_factory=lambda: uuid4().hex[:8],
        description="Unique id for this run (one per fire).",
    )
    job_id: str = Field(description="The job that ran.")
    agent: str = Field(
        default="",
        description="Which agent owns the job (routing column).",
    )
    conversation_id: str = Field(
        default="global",
        description="The origin conversation the run belongs to (routing column).",
    )
    status: str = Field(
        default="running",
        description="running / done / error / skipped.",
    )
    detail: str = Field(
        default="",
        description="Free-form detail (what it did / why it errored / exit code).",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the run was recorded.",
    )
