"""Tree-sitter → normalized records (the "own the value" layer).

Delegates parsing to ``tree-sitter-language-pack.process()`` (for structure,
symbols, imports, comments, metrics) and maps those onto our own ``SymbolInfo`` /
``FileInfo`` records in ``_schema.py``.  On 1.15.7 (the version paf declares + runs),
``process()`` fills ``structure[].signature`` natively; we prefer it (normalized bare
by ``_native_signature``) and fall back to the custom-query recovery in ``_queries.py``
(see the ``tree-sitter-language-pack`` skill) only where native is
empty (classes → ``signature=None``; grammars 1.15.7 still blanks).

Honest gaps handled here:
- ``process()`` never fills ``docstrings`` (``doc``/``doc_comment``) even on 1.15.7 —
  we recover ``doc`` per symbol via QueryCursor queries. ``signature`` *is* native now
  (1.15.7); recovery is the fallback only.
- ``visibility``/``decorators`` still empty on 1.15.7.
- C: ``process()`` yields an anonymous ``Function`` (name=None) — skipped (no usable
  symbol span from the parser); same for the TSX extra anonymous Function item.
- Import normalization: ``ImportInfo.source/items/alias/is_wildcard`` → canonical fields.
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
# We enforce the bound ourselves by reading + pre-checking size so the skip is
# explicit before we spend a parse.
_MAX_SOURCE_BYTES = 1_000_000  # 1 MB per file

# Language names paf passes to tree-sitter-language-pack — the "available_languages"
# id set (12 in the wheel). We analyze only code files; config/data files (json/toml)
# flow through data_extraction later, not here.
_SUPPORTED = {
    "python",
    "typescript",
    "tsx",
    "javascript",
    "php",
    "go",
    "rust",
    "java",
    "c",
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
    """Native ``structure[].signature`` normalized to our bare-signature contract.

    The native field emits the full header (``def f(a) -> b:`` / ``async def f(a):``),
    but the documented and tested contract (``test_signature_*``) is the **bare**
    ``f(a) -> b`` form, and classes carry no signature. So: adopt native only for
    function-like kinds, and strip the leading keyword + trailing colon back down to
    the bare form. Returns ``None`` when there is nothing to adopt (class, or native
    left it empty).
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
            sig = sig[len(kw) :]
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
        # The pre-read size guard in `analyze_file` covers runaway input with an
        # explicit, parse-free skip (see comment at `_MAX_SOURCE_BYTES`).
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

    # Module-level named assignments (NAME = ... / NAME: Type = ...) — a whole
    # class of symbols tree-sitter's `structure` never emits (registry.py's
    # `CAPABILITY_TYPES` is the poster child: an 8KB file whose only substance is
    # one module-level assignment). Captured as Constant/Variable symbols so they
    # are searchable.
    assign_searchable = _module_assignments(recovery)
    symbols = symbols + assign_searchable

    imports = _map_imports(res.imports)

    if not symbols and not imports:
        return None

    return FileInfo(
        path=rel,
        language=language,
        doc=recovery.module_docstring(),
        symbols=[s.name for s in symbols if s.name],
        imports=imports,
        import_names=_map_import_names(res.imports),
        searchable_symbols=assign_searchable + searchable,
    )


def _read_source(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _detect_language(filename: str) -> str | None:
    ext = Path(filename).suffix.lstrip(".").lower()
    _EXT = {
        "py": "python",
        "ts": "typescript",
        "tsx": "tsx",
        "js": "javascript",
        "jsx": "javascript",
        "php": "php",
        "go": "go",
        "rs": "rust",
        "java": "java",
        "c": "c",
        "h": "c",
    }
    return _EXT.get(ext)


def detect_language(filename: str) -> str | None:
    """Public wrapper over extension → language id ('' or None when unknown)."""
    return _detect_language(filename)


def _module_assignments(recovery: QueryRecovery) -> list[SymbolInfo]:
    """Build Constant/Variable SymbolInfo records for module-level named assignments.

    Wraps ``QueryRecovery.module_assignments()`` (python only) into full
    ``SymbolInfo`` records: kind is ``Constant`` for ALL-CAPS names else ``Variable``;
    ``signature`` is the type annotation when present; ``doc`` is the preceding
    ``#`` comment line when one is attached (reuses the comment-association
    machinery); ``body`` is the assignment source (weight-1 field). Returns [] for
    languages where module-assignment capture isn't implemented yet (python,
    typescript, tsx, javascript today; others deferred per "one language at a time").
    """
    out: list[SymbolInfo] = []
    for name, start, end, type_ann in recovery.module_assignments():
        span = Span(start_byte=start, end_byte=end)
        kind = "Constant" if name.upper() == name and "_" in name else "Variable"
        signature = f"{name}: {type_ann}" if type_ann else None
        doc = recovery._preceding_comment_block(start)
        out.append(
            SymbolInfo(
                name=name,
                qualified=name,  # module-level: no parent dotted path
                kind=kind,
                signature=signature,
                doc=doc,
                body=recovery.body_for(span),
                span=span,
            )
        )
    return out


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
            # Prefer a normalized native signature (structure[].signature); fall back
            # to our hand recovery (QueryRecovery) only when native has nothing (class,
            # or a grammar leaves it blank).
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


def _map_import_names(imports) -> list[str]:
    """The imported SYMBOL names (structured), for graph reference resolution.

    Unlike ``_map_imports`` (which keeps raw statement strings for display), this
    returns the actual *things bound into scope* by each import — the items from
    ``from m import A, B`` (A, B), and the alias from ``import m as A`` (A). A bare
    ``import m`` brings a MODULE into scope (``m``), not a symbol — and resolving
    a module name globally is exactly the false-collision trap (``. _store`` vs the
    ``_store`` module var), so it is DELIBERATELY excluded here: module references
    are the graph's import-resolution job, not name-matched symbol references.

    This is language-agnostic because tree-sitter fills ``ImportInfo.items`` for
    every language (Python + JS/TS confirmed; other languages yield items only when
    the grammar surfaces them — exactly what the reference resolver can trust).
    """
    out: list[str] = []
    for imp in imports:
        items = list(getattr(imp, "items", None) or [])
        alias = getattr(imp, "alias", None)
        stmt = (imp.source or "").strip()
        if stmt.startswith("import ") and alias and not items:
            # `import m as A` -> A is the local symbol binding
            out.append(alias)
        else:
            # `from m import A, B` / `import { A } from "..."` -> the items
            out.extend(it for it in items if it and it != "*")
    # dedupe preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq
