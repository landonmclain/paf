"""Regex fallback for unparseable/niche files (STUB).

Planned: a lightweight regex fallback
for files tree-sitter can't parse (config, markdown, unusual DSLs, generated code).
Runs when `_analyze.py` yields nothing/near-nothing for a file, so every likely-indexable
file still contributes *some* symbol/name signal to search.

Stub for layout.
"""

from __future__ import annotations

from ._schema import FileInfo


def fallback_analyze(path: str) -> FileInfo:
    """STUB — regex-based fallback extraction for unparseable files."""
    raise NotImplementedError("fallback_analyze is not built until a later phase")
