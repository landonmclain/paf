"""Shared helpers for model-visible tool output."""


def truncate_tail(text: str, max_chars: int) -> str:
    """Limit text to `max_chars`, keeping the tail.

    The truncation marker counts toward the cap and reports the exact number
    of retained chars. A cap too small to fit any marker returns a bare tail.
    """
    if max_chars <= 0:  # pragma: no cover - callers validate the cap at construction
        return ""
    if len(text) <= max_chars:
        return text

    def marker(tail_chars: int) -> str:
        return f"[... output truncated, showing last {tail_chars} chars]\n"

    tail_chars = max(0, max_chars - len(marker(max_chars)))
    if tail_chars + 1 + len(marker(tail_chars + 1)) <= max_chars:
        tail_chars += 1
    if tail_chars == 0:
        return text[-max_chars:]
    return marker(tail_chars) + text[-tail_chars:]


# --- background command output (durable file, paged like subagent output) ------

NOTIFY_MAX_CHARS = 1500
"""Output at or under this many characters rides the completion notify whole."""

CHECK_MAX_CHARS = 8000
"""Default page size for ``check_command``."""


def tail_excerpt(text: str, run_id: str, limit: int = NOTIFY_MAX_CHARS) -> str:
    """``text`` whole if it fits, else its END cut at a line boundary plus a
    pointer to the start. Shell output's answer is usually at the end (the
    summary, the error), unlike a subagent's TL;DR-first output."""
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    idx = tail.find("\n")
    if 0 <= idx <= limit // 2:
        tail = tail[idx + 1 :]
    skipped = len(text) - len(tail)
    return (
        f"[… {skipped} earlier chars; check_command('{run_id}', offset=0) "
        f"for the start]\n{tail.lstrip(chr(10))}"
    )


def notify_text(run_id: str, command: str, exit_code: int | None, output: str) -> str:
    """The ``[system]`` message delivered when a background command finishes."""
    status = "finished" if exit_code is None else f"exit {exit_code}"
    head = f"[system] Background command {run_id} ({status}): {command}"
    body = output.strip()
    if not body:
        return f"{head}\n\n(no output)"
    return f"{head}\n\n{tail_excerpt(body, run_id)}"


def page(text: str, run_id: str, offset: int | None, max_chars: int) -> str:
    """One page of output: from ``offset``, or the last ``max_chars`` when
    ``offset`` is None, with pointers to the rest."""
    total = len(text)
    size = max(1, max_chars)
    start = max(0, total - size) if offset is None else min(max(0, offset), total)
    end = min(total, start + size)
    notes = []
    if start > 0:
        notes.append(
            f"[… {start} earlier chars; check_command('{run_id}', "
            f"offset={max(0, start - size)})]"
        )
    body = text[start:end]
    if end < total:
        body += (
            f"\n[… {total - end} more chars; check_command('{run_id}', offset={end})]"
        )
    head = f"[output chars {start}-{end} of {total}]"
    return "\n".join([head, *notes, body])
