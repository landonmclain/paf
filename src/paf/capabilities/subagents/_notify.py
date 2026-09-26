"""Completion-notify text for subagent tasks (full output or a truncated head).

A background task's result reaches the parent as a ``[system]`` notify message,
which lands in the parent conversation's history — so it is bounded. Output up
to :data:`NOTIFY_MAX_CHARS` rides the notify whole (the parent needs no
``check_task`` call); longer output is cut back to the last paragraph or line
boundary inside the limit and ends with a pointer to ``check_task`` for the
rest. Leaves are asked to open with a TL;DR, so the head is usually the answer.
"""

from __future__ import annotations

NOTIFY_MAX_CHARS = 1500
"""Output at or under this many characters is delivered whole."""


def excerpt(text: str, task_id: str, limit: int = NOTIFY_MAX_CHARS) -> str:
    """Return ``text`` whole if it fits, else a boundary-cut head plus a pointer.

    The cut prefers the last paragraph break (``\\n\\n``), then the last line
    break, as long as it keeps at least half the limit; otherwise it hard-cuts at
    ``limit`` so a single huge paragraph still yields a useful head.
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = -1
    for sep in ("\n\n", "\n"):
        idx = head.rfind(sep)
        if idx >= limit // 2:
            cut = idx
            break
    kept = (head[:cut] if cut != -1 else head).rstrip()
    remaining = len(text) - len(kept)
    return (
        f"{kept}\n\n… [truncated — {remaining} more chars; "
        f"check_task('{task_id}') for the full output]"
    )


def notify_text(agent: str, task_id: str, ok: bool, body: str) -> str:
    """The ``[system]`` message delivered to the parent when a task finishes.

    ``body`` is the leaf's output on success (excerpted here) or the error
    detail on failure (delivered as-is — error details are short).
    """
    if ok:
        return (
            f"[system] Subagent {agent} task {task_id} finished (ok):\n\n"
            f"{excerpt(body, task_id)}"
        )
    return f"[system] Subagent {agent} task {task_id} finished (error): {body}"
