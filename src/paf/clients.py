"""Helpers shared by paf's front ends (the chat CLI and the SSE server).

Both clients drive the same ``Session``; this module holds the pieces they
would otherwise each re-implement: the wording of a background-completion
notify, reading a conversation's plan and task rows from the stores the
capabilities write, and turning uploaded images into prompt text.
"""

from __future__ import annotations

import logging
from typing import Any

from paf.capabilities.planning import FilePlanStore, PlanItem
from paf.capabilities.visual import (
    Describer,
    UploadMeta,
    UploadStore,
    with_image_blocks,
)
from paf.config import PLANS_DIR, PROCESSES_DIR

log = logging.getLogger(__name__)

DESCRIBE_TIMEOUT_SECONDS = 180.0


def notify_message(run_id: str, detail: str) -> str:
    """The text a background-completion notify delivers to the agent.

    Subagent tasks (``subagent.<cid>.<n>``) and shell commands
    (``shell.<cid>.<n>``) hand over a ready-made message (the output, or an
    excerpt plus a ``check_task`` / ``check_command`` pointer), delivered as-is.
    Anything else is wrapped here.
    """
    if run_id.startswith(("subagent.", "shell.")):
        return detail
    return f"[system] Background process {run_id} finished: {detail}"


async def plan_items(scoped_cid: str) -> list[PlanItem]:
    """A conversation's current plan, from the store the Planning capability writes."""
    return await FilePlanStore(root=PLANS_DIR, conversation_id=scoped_cid).get_items()


def task_frames(agent: str, scoped_cid: str) -> list[dict[str, Any]]:
    """Shell-command and subagent rows for a conversation (running or recent).

    These live in their own durable stores, not the background coordinator's
    task history, so a client must read them to show them after a restart.
    Each source is best-effort: a broken store logs and contributes nothing.
    """
    from paf.capabilities.shell._background import conversation_shell_frames
    from paf.capabilities.subagents._store import conversation_subagent_frames

    frames: list[dict[str, Any]] = []
    for read in (conversation_shell_frames, conversation_subagent_frames):
        try:
            frames += read(PROCESSES_DIR, agent, scoped_cid)
        except Exception:  # noqa: BLE001 - status rows are advisory
            log.warning("%s failed", read.__name__, exc_info=True)
    return frames


class UnknownImageError(LookupError):
    """An image id that is not in the upload store."""


async def prompt_with_images(
    prompt: str,
    image_ids: list[str],
    *,
    model: str | None,
    model_settings: dict[str, Any] | None = None,
    uploads: UploadStore,
    describer: Describer,
    timeout: float = DESCRIBE_TIMEOUT_SECONDS,
) -> str:
    """``prompt`` plus one described ``<image>`` block per uploaded image.

    ``model`` is the agent's vision model (``visual.model`` in its spec); each
    description is awaited, starting any not yet begun. With no vision model
    the images are attached with whatever description they already have.
    """
    metas: list[UploadMeta] = []
    for upload_id in image_ids:
        meta = (
            await describer.wait(upload_id, model, model_settings, timeout=timeout)
            if model
            else uploads.meta(upload_id)
        )
        if meta is None:
            raise UnknownImageError(upload_id)
        metas.append(meta)
    return with_image_blocks(prompt, metas)
