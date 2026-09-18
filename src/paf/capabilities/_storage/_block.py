"""``BlockRef`` / ``Span`` + ``read_block`` — the query-the-pointer half of the seam.

The pointer model:

- A retrieval hit never carries heavy source text. It collapses to a
  lightweight :class:`BlockRef` — a source path plus an optional byte/line
  :class:`Span`.
- :func:`read_block` materializes **just the named range**, on demand, bounded
  to ``limit_bytes``. In the flat-file backend that's a byte-slice of the file;
  in a future SQLite/Postgres/vector backend it is the equivalent bounded fetch.
  So the search layer can query *any* backend through one call — this is what
  keeps the seam medium-agnostic rather than flat-file-only.

``BlockRef.file`` is expected to be an absolute path by the time it crosses the
seam — the tool layer resolves a repo-relative pointer against its scan root
before calling ``read_block``, so a co-located backend needs no extra lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Span:
    """Byte-exact and/or line span of a region (1-indexed inclusive lines).

    Byte offsets may be absent (``None``) when a region was produced without a
    precise byte source (e.g. a line-only fallback) — ``read_block`` then falls
    back to a line read.
    """

    start_byte: int | None = None
    end_byte: int | None = None
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True)
class BlockRef:
    """A pointer to a source region (the lightweight handle search hands out)."""

    file: str
    span: Span | None = None


def read_block(
    path: str | Path,
    span: Span | None = None,
    *,
    limit_bytes: int = 65536,
) -> str | None:
    """Return the source text for ``span`` of ``path``, bounded to ``limit_bytes``.

    - With byte offsets present: exact ``[start_byte:end_byte]`` slice (smallest read).
    - With line numbers (no byte offsets): those lines only.
    - With no span: whole file, capped at ``limit_bytes``.

    ``None`` when the file doesn't exist. Never returns more than ``limit_bytes``
    so a single huge file cannot blow a tool-result budget.
    """
    p = Path(path)
    if not p.exists():
        return None

    if span is not None and span.start_byte is not None and span.end_byte is not None:
        with p.open("rb") as fh:
            fh.seek(span.start_byte)
            data = fh.read(span.end_byte - span.start_byte)
        return data.decode("utf-8", errors="replace")

    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return None

    if span is not None and span.start_line is not None and span.end_line is not None:
        lines = text.splitlines()
        text = "\n".join(lines[span.start_line - 1 : span.end_line])

    if len(text) > limit_bytes:
        text = text[:limit_bytes]
    return text
