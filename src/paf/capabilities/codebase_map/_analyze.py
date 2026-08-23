"""Tree-sitter → normalized records (the "own the value" layer, Phase 1).

Delegates parsing to ``tree-sitter-language-pack.process()`` (for structure,
symbols, imports, comments, metrics) and maps those onto our own ``SymbolInfo`` /
``FileInfo`` records in ``_schema.py``. Fills the two fields ``process()`` leaves
empty on 1.14.3 — ``signature`` and ``doc`` — via the verified custom-query recovery
in ``_queries.py`` (see the ``tree-sitter-language-pack`` skill, FILL THE BLANKS).

Honest gaps handled here (all documented in the Archive note + skill):
- ``process()`` never fills ``signature``/``visibility``/``docstrings``; we recover
  signature + doc per symbol via QueryCursor queries instead.
- C: ``process()`` yields an anonymous ``Function`` (name=None) — skipped (no usable
  symbol span from the parser); same for the TSX extra anonymous Function item.
- Import normalization: ``ImportInfo.source/items/alias/is_wildcard`` → canonical fields.

Swap-ready: when a wheel with working native ``signature``/``docstrings`` ships, flip
to those fields — the ``SignatureInfo``/``FileInfo`` schema keeps ``signature``/``doc``
regardless (one-line switch). Keeps ``signature``/``doc`` populated today via recovery.
"""

from __future__ import annotations

from pathlib import Path

from tree_sitter_language_pack import (
    ProcessConfig,
    process,
)

from ._queries import QueryRecovery
from ._schema import FileInfo, Span, SymbolInfo, path_rel_to_scan_root

# DDOS/runaway guard for third-party repos: files over this size are skipped.
# (1.14.3 ProcessConfig has no max_source_bytes/parse_timeout_ms — 1.15-docs fields —
#  so we enforce the bound ourselves by reading + pre-checking size.)
_MAX_SOURCE_BYTES = 1_000_000  # 1 MB per file

# Language names paf passes to tree-sitter-language-pack — the "available_languages"
# id set (12 in the wheel). We analyze only code files; config/data files (json/toml)
# flow through data_extraction later, not here.
_SUPPORTED = {
    "python", "typescript", "tsx", "javascript", "php", "go", "rust", "java", "c",
}


def _to_span(ts_span) -> Span | None:
    """Map a tree-sitter Span onto our 1-indexed Span (ts lines are 0-indexed)."""
    if ts_span is None:
        return None
    return Span(
        start_byte=ts_span.start_byte,
        end_byte=ts_span.end_byte,
        start_line=(ts_span.start_line + 1) if ts_span.start_line is not None else None,
        end_line=(ts_span.end_line + 1) if ts_span.end_line is not None else None,
    )


def _kind_value(kind) -> str:
    """Normalize a StructureKind/SymbolKind enum to its string value (e.g. 'Function')."""
    v = getattr(kind, "value", kind)
    return str(v)


# Kinds that carry a signature line (functions + methods). Everything else (classes,
# modules, etc.) has signature=None per the documented/tested contract
# (test_class_signature_is_none).
_FUNCTION_LIKE_KINDS = {"Function", "Method"}


def _native_signature(it) -> str | None:
    """Native 1.15.7 ``structure[].signature`` normalized to our bare-signature contract.

    1.15.7 populates ``structure[].signature`` natively (1.14.3 left it empty, which is
    why ``_queries.py`` ``QueryRecovery.signature_for`` exists). The native field emits
    the full header (``def f(a) -> b:`` / ``async def f(a):``), but the documented and
    tested contract (``test_signature_*``) is the **bare** ``f(a) -> b`` form, and
    classes carry no signature. So: adopt native only for function-like kinds, and strip
    the leading keyword + trailing colon back down to the bare form. Returns ``None``
    when there is nothing to adopt (class, or native left it empty).
    """
    if _kind_value(it.kind) not in _FUNCTION_LIKE_KINDS:
        return None
    sig = (getattr(it, "signature", None) or "").strip()
    if not sig:
        return None
    if sig.endswith(":"):
        sig = sig[:-1].rstrip()  # drop the trailing colon the header always has
    for kw in ("async def ", "def "):  # strip the language keyword(s)
        if sig.startswith(kw):
            sig = sig[len(kw):]
            break
    return sig or None


def analyze_file(
    path: str | Path,
    scan_root: str | Path,
    source: str | None = None,
    language: str | None = None,
) -> FileInfo | None:
    """Extract symbols/imports/docstrings from one source file.

    Returns ``None`` when the file should be skipped: unsupported language, oversized
    (over ``_MAX_SOURCE_BYTES``), unparseable, or no symbols/imports worth indexing.
    ``source`` is read from ``path`` if not supplied. ``language`` is resolved from
    the file extension if not supplied.
    """
    rel = path_rel_to_scan_root(path, scan_root)
    if language is None:
        language = _detect_language(Path(path).name)
    if language is None or language not in _SUPPORTED:
        return None

    if source is None:
        source = _read_source(path)
    if source is None or len(source.encode()) > _MAX_SOURCE_BYTES:
        return None

    recovery = QueryRecovery(language=language, source=source)
    try:
        # NOTE: 1.14.3 ProcessConfig has NO max_source_bytes/parse_timeout_ms (1.15-docs
        # fields) — the pre-read size guard in `analyze_file` covers runaway input instead.
        res = process(
            source,
            ProcessConfig(
                language=language,
                structure=True,
                imports=True,
                comments=True,
                symbols=True,
                diagnostics=False,
            ),
        )
    except Exception:  # noqa: BLE001 — unparseable/odd file -> skip, not fatal
        return None

    symbols, searchable = _walk_with_children(res.structure, recovery)

    imports = _map_imports(res.imports)

    if not symbols and not imports:
        return None

    return FileInfo(
        path=rel,
        language=language,
        doc=recovery.module_docstring(),
        symbols=[s.name for s in symbols if s.name],
        imports=imports,
        searchable_symbols=searchable,
    )


def _read_source(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _detect_language(filename: str) -> str | None:
    ext = Path(filename).suffix.lstrip(".").lower()
    _EXT = {
        "py": "python", "ts": "typescript", "tsx": "tsx", "js": "javascript",
        "jsx": "javascript", "php": "php", "go": "go", "rs": "rust",
        "java": "java", "c": "c", "h": "c",
    }
    return _EXT.get(ext)


def detect_language(filename: str) -> str | None:
    """Public wrapper over extension → language id ('' or None when unknown)."""
    return _detect_language(filename)


def _walk_with_children(items, recovery: QueryRecovery):
    """Recursive builder that attaches children to frozen SymbolInfo via __setattr__."""
    all_searchable: list[SymbolInfo] = []

    def build(items, parent_name: str | None) -> list[SymbolInfo]:
        built: list[SymbolInfo] = []
        for it in items:
            if not it.name:
                continue
            span = _to_span(it.span)
            body_span = _to_span(it.body_span)
            qualified = f"{parent_name}.{it.name}" if parent_name else it.name
            # Prefer a normalized native signature (1.15.7 populates
            # structure[].signature); fall back to our hand recovery (QueryRecovery)
            # only when native has nothing (class, or a grammar 1.15.7 leaves blank).
            sig = _native_signature(it) or recovery.signature_for(span, body_span)
            doc = recovery.docstring_for(span, body_span)
            symbol = SymbolInfo(
                name=it.name,
                qualified=qualified,
                kind=_kind_value(it.kind),
                signature=sig,
                doc=doc,
                body=recovery.body_for(body_span),
                span=span,
                parent=parent_name,
            )
            if it.children:
                child_built = build(it.children, qualified)
                object.__setattr__(symbol, "children", child_built)
            built.append(symbol)
            all_searchable.append(symbol)
        return built

    root = build(items, None)
    return root, all_searchable


def _map_imports(imports) -> list[str]:
    """Normalize ImportInfo list to canonical import strings (module or item source)."""
    out: list[str] = []
    for imp in imports:
        src = (imp.source or "").strip()
        items = getattr(imp, "items", None) or []
        if src:
            # Go emits a duplicate bare-string import node; dedupe by text later.
            token = src if not src.startswith(('"', "'")) else src
            out.append(token)
        elif items:
            out.extend(items)
    # dedupe preserving order (Go quirk: `import "fmt"` -> source AND bare node)
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq
