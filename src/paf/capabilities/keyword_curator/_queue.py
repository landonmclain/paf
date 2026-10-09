"""The keyword-curation queue: runs one curator pass off the turn path.

A single-flight FIFO worker (the same ``BackgroundQueueService`` the memory and
thread-summarizer workers use). The coordinator's post-turn trigger decides *when*
(something changed, cooldown elapsed, curation opted in) and enqueues a
:class:`KeywordJob`; this worker does the scan + curate + store write. It is the
sole writer of the keyword store.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paf.background import BackgroundQueueService, QueueState

from ._background import PassResult, project_seeds, run_keyword_pass
from ._sources import scan_sources

log = logging.getLogger(__name__)

PassFn = Callable[..., Awaitable[PassResult]]
"""``(directory, sources, *, limit, seeds) -> PassResult`` — the worker's seam."""


@dataclass(frozen=True)
class KeywordJob:
    """One curation request for an agent's keyword store."""

    agent_name: str
    directory: str
    sources_config: dict[str, Any] = field(default_factory=dict)
    """``scan_sources`` kwargs (skill libraries, vault root/allow/deny)."""
    limit: int | None = None
    """Cap on changed sources curated this pass (a backlog is worked off over
    several passes instead of one large, expensive one)."""
    seed_projects: bool = True


def summarize_pass(result: PassResult) -> str:
    """One human line for the Processes row."""
    if result.skipped:
        return f"skipped — {result.skipped}"
    parts = [f"{len(result.keyed)} curated"]
    if result.failed:
        parts.append(f"{len(result.failed)} failed")
    if result.removed:
        parts.append(f"{len(result.removed)} dropped")
    parts.append(f"{result.entities} entities")
    return ", ".join(parts)


class KeywordCuratorQueue(BackgroundQueueService[KeywordJob]):
    """Single-flight worker that owns an agent's keyword store."""

    name = "keyword-curator"

    def __init__(
        self,
        *,
        pass_fn: PassFn | None = None,
        on_complete: Callable[[str, bool, str], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__()
        self.pass_fn = pass_fn or run_keyword_pass
        self.on_complete = on_complete

    def job_trace(self, session_id: str, job: KeywordJob) -> tuple[str, dict]:
        """Each pass is its own trace, grouped by agent (not nested in a turn)."""
        del session_id
        return (f"paf.keyword_curate.{job.agent_name}", {})

    async def process_job(
        self,
        session_id: str,
        job: KeywordJob,
        state: QueueState[KeywordJob],
    ) -> None:
        del state
        ok, detail = False, ""
        try:
            sources = await asyncio.to_thread(scan_sources, **job.sources_config)
            seeds = project_seeds() if job.seed_projects else []
            result = await self.pass_fn(
                Path(job.directory), sources, limit=job.limit, seeds=seeds
            )
            ok, detail = result.ok, summarize_pass(result)
            if result.failed:
                detail += f" — retry next pass: {', '.join(sorted(result.failed)[:3])}"
        except Exception as exc:  # noqa: BLE001 - a failed pass must not kill the worker
            log.exception("keyword curation pass failed agent=%s", job.agent_name)
            detail = f"{type(exc).__name__}: {exc}"
        if self.on_complete is not None:
            await self.on_complete(session_id, ok, detail)
