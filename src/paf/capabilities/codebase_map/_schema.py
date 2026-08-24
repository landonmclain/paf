"""Typed storage/search records for the codebase-map capability.

Every artifact in the storage seam (§3 of the spec) maps to a record here, plus the
one universal **search result shape** (`SearchHit` + `BlockRef`) that every
`RetrievalBackend` returns. Records live here (not in ``_store.py`` / ``_retrieval.py``)
so the result type sits next to the records it references and both modules import it
without a cross-module cycle.

All records are frozen dataclasses (lean, dependency-light). ``BlockRef`` is the spine
of the whole design: a search hit collapses to a *pointer* (repo-relative file + byte
span), and the heavy source text is only read on demand via ``read_block`` — never
eagerly embedded in a hit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Span:
    """Byte-exact + line span of a symbol or region (maps 1:1 to tree-sitter Span).

    Byte offsets may be absent (``None``) for spans produced without tree-sitter (e.g.
    a line-only ``read_block`` fallback) — ``read_block`` then falls back to a line
    read. Line numbers are 1-indexed inclusive.
    """

    start_byte: int | None = None
    end_byte: int | None = None  # exclusive
    start_line: int | None = None  # 1-indexed inclusive
    end_line: int | None = None  # 1-indexed inclusive


@dataclass(frozen=True)
class BlockRef:
    """A pointer to a source region. The only heavy-read handles we hand out.

    ``file`` is repo-relative (relative to ``scan_root``). ``span`` carries byte offsets
    so the storage seam can do an exact bounded slice read.
    """

    file: str
    span: Span


@dataclass(frozen=True)
class SymbolInfo:
    """One normalized code symbol (our layer over tree-sitter's StructureItem/SymbolInfo)."""

    name: str
    qualified: str | None = None  # dotted path when derivable
    kind: str = "OTHER"  # StructureKind/SymbolKind string, e.g. "CLASS"/"FUNCTION"
    signature: str | None = None
    doc: str | None = None  # our docstring extraction (tree-sitter's is empty)
    body: str | None = None  # symbol body source text (weight-1 field in §3.7.2)
    span: Span | None = None
    visibility: str | None = None  # pub/public/private
    children: list[SymbolInfo] = field(default_factory=list)
    parent: str | None = None


@dataclass(frozen=True)
class FileInfo:
    """Per-file head record (one row of index.json — the HEAD layer)."""

    path: str  # repo-relative
    language: str | None = None
    doc: str | None = None  # module-level docstring (describes the whole file)
    symbols: list[str] = field(default_factory=list)  # name list
    imports: list[str] = field(default_factory=list)
    references_count: int = 0
    searchable_symbols: list[SymbolInfo] = field(default_factory=list)
    # True when the file is in-scope (part of the codebase) but no extractor could
    # produce symbols/imports (unknown/binary/oversized/unparseable). It is still
    # recorded so the corpus is complete and the file is stable, not "changed".
    unparsed: bool = False


# The on-disk `index.json` HEAD is a LEAN serialization of a FileInfo — metadata +
# module-docstring head + symbol NAME LISTS only (spec §3.2). The rich per-symbol
# detail (body/signature/doc/span) lives in `manifest.json` (detail) and is
# tokenized into `tokens.json` (search). `role`/`tags` were dropped from the design
# (overview role_guess taxonomy removed); they are never populated.
# We keep `FileInfo` rich in-memory (used by _analyze/_rank/_graph/tests) and emit
# the lean head only at serialization time via `head_record()` (in _scan.py).


@dataclass(frozen=True)
class SearchHit:
    """ONE universal result shape — every RetrievalBackend returns these.

    A hit collapses to a ``BlockRef`` pointer; the heavy source text is deferred until
    ``read_block``. ``score`` is backend-specific relevance.
    """

    file: str  # repo-relative path
    start_line: int  # 1-indexed inclusive
    end_line: int  # 1-indexed inclusive
    symbol: str | None = None  # owning symbol name (heads-up display)
    snippet: str = ""  # bounded context text
    score: float = 0.0
    block: BlockRef | None = None  # collapse EVERY hit to a pointer


@dataclass(frozen=True)
class DistilledEntry:
    """One ranked entry of the distilled map (aider --map-tokens-style)."""

    rank: int
    file: str
    symbol: str
    signature: str | None = None
    references_count: int = 0
    ref_weight: float = 0.0


def path_rel_to_scan_root(path: str | Path, scan_root: str | Path) -> str:
    """Return ``path`` relative to ``scan_root`` as a POSIX-style repo-relative str.

    Used when normalizing tree-sitter results / building BlockRefs so every record
    stores a stable, portable repo-relative path, not an absolute one.
    """
    rel = Path(path).resolve().relative_to(Path(scan_root).resolve())
    return rel.as_posix()
