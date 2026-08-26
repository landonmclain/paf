"""Isolated view reader: journal-rendered MEMORY.md inserted into context.

The M2 three-layer design (see `_journal.py` / `_render.py`) makes the
append-only JSONL journal the *source of truth* and `MEMORY.md` a *pure
presentation view* rendered from it. The recall/answerer is READ-ONLY: it
never touches the journal and never writes memory. Its job is simply to have
the current rendered notebook inserted into `<memory>` on every request.

That insertion goes through this view reader on the *rendered output only*:
read() returns the top-level ``MEMORY.md`` file content verbatim -- the exact md
that ``write_render`` produced -- and the capability
injects that text in full. No scope/agent
pathing, no protocol indirection: read the file, insert the text.

Isolated by design: this reader does not implement write/delete/search and
cannot mutate anything. It exists purely as the read seam for the injected
notebook.
"""

from __future__ import annotations

from pathlib import Path

from ._store import MemoryFile


class JournalViewStore:
    """Read-only view of a journal-rendered ``MEMORY.md`` file.

    Implements just the read side of the :class:`MemoryStore` protocol that
    injection needs:

    - ``read(path, max_chars)`` returns the top-level ``MEMORY.md`` content in
      this root directory, *regardless of the scoped path* the capability asks
      for (it asks ``assistant/MEMORY.md``; we always answer with the root
      render).
    - ``list_paths(...)`` returns ``[]`` -- a pure journal view has no topic
      subfiles.
    - ``write``/``delete``/``get_operation`` are read-only stubs that raise.

    ``MEMORY.md`` must already exist (the render step writes it before recall).
    Missing file -> ``read`` returns ``None`` (the capability falls back to its
    empty-memory block), never raises.
    """

    def __init__(self, root: str | Path) -> None:
        self._main = Path(root) / "MEMORY.md"

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        try:
            content = self._main.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        truncated = len(content) > max_chars
        content = content[:max_chars]
        return MemoryFile(
            content=content,
            version="view",
            operation_id=None,
            truncated=truncated,
        )

    async def get_operation(self, operation):
        return None

    async def write(self, path, content, *, expected_version, operation=None):
        raise NotImplementedError("JournalViewStore is read-only")

    async def delete(self, path, *, expected_version, operation=None):
        raise NotImplementedError("JournalViewStore is read-only")

    async def list_paths(self, prefix: str = "", *, limit: int) -> list[str]:
        return []
