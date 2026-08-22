"""Tree-sitter → normalized records (STUB — Phase 0 layout only).

Planned (per Codebase-Map-Capability-Spec § `_analyze.py`): the "own the value" layer.
Delegates parse to `tree-sitter-language-pack.process()`; maps `StructureItem` /
`SymbolInfo` / `ImportInfo` / `Span` onto our own `SymbolInfo` / `FileInfo` records
(`_schema.py`). Houses the docstring + import-normalization layer:
- Own docstring extraction: symbol `span` → `BlockRef` → read the leading docstring of
  the common forms (tree-sitter's `res.docstrings` is empty on 1.14.3 — verified).
- PHP comment extraction is a separate gap (PHP returns 0 comments).
- Per-field `searchable_symbols` capture (§3.7.2) for later BM25 field-weighting.

This is **Phase 1 — the deliverable**. Phase 0 is storage + schema + eval shell only;
this module is a stub so the layout is complete.
"""

from __future__ import annotations

from ._schema import FileInfo, SymbolInfo


def analyze_file(path: str) -> FileInfo:
    """STUB — extract symbols/imports/docstrings from a file (Phase 1)."""
    raise NotImplementedError("analyze_file is not built until Phase 1")


def extract_docstring(source: str, symbol: SymbolInfo) -> str | None:
    """STUB — own docstring extraction via span → read (Phase 1)."""
    raise NotImplementedError("extract_docstring is not built until Phase 1")
