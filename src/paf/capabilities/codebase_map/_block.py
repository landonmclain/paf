"""BlockRef + read_block — the pointer-read half of the seam.

A search hit never carries the heavy source text; it collapses to a ``BlockRef`` (a
repo-relative path + byte span, see ``_schema.py``). ``read_block`` is the bounded read
that materializes only the exact range a pointer names, on demand. This is the
"query-the-pointer" model: the model asks for the lines it actually needs, not whole
files.

``read_block`` here works on an **absolute** file path plus a ``Span``. The storage seam
(``FlatFileStorage.read_block``) receives a ``BlockRef`` whose ``file`` the caller has
already resolved to an absolute path (the ``read_block`` tool resolves the repo-relative
pointer against ``scan_root`` before crossing the seam) — or callers pass an absolute
path directly.

``Span``/``BlockRef`` are codebase-map's own schema (its gate tests require them to be
surfaced as symbols here); the shared ``paf.capabilities._storage._block`` carries the
same pointer model for other capabilities. Only the atomic-write helpers are conformed
to the shared seam (see ``_store.py``).
"""

from __future__ import annotations

from pathlib import Path

from ._schema import BlockRef, Span


def read_block(
    path: str | Path, span: Span | None = None, *, limit_bytes: int = 65536
) -> str | None:
    """Return the source text for ``span`` of ``path``, bounded to ``limit_bytes``.

    - With a ``Span`` carrying byte offsets: exact ``[start_byte:end_byte]`` slice.
    - With byte offsets absent but line numbers present: read only those lines.
    - With no span at all: return the whole file, capped at ``limit_bytes``.

    Returns ``None`` when the file doesn't exist. Never returns more than ``limit_bytes``
    of content, so a single huge file can't blow a tool-result budget.
    """
    p = Path(path)
    if not p.exists():
        return None

    if span is not None and span.start_byte is not None and span.end_byte is not None:
        # exact byte slice — smallest possible read
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
        lines = lines[span.start_line - 1 : span.end_line]
        text = "\n".join(lines)

    if len(text) > limit_bytes:
        text = text[:limit_bytes]
    return text


__all__ = ["BlockRef", "Span", "read_block"]
