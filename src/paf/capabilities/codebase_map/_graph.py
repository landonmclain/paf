"""Call graph + references — Phase 2 (count first; edges come next).

Implemented now (per Landon, 2026-08-23):
  - ``reference_counts``  — inbound-reference counts per FILE, computed at scan time
    and persisted into each ``FileInfo.references_count`` in the index (head). At
    full-scan scale (~114 files, sha256 full rewalk) this is free and never stale;
    recomputed-at-scan beats §5.x's "derive on read" (which was written for a
    file-watcher/incremental future we don't have).

Deferred (key deliverable, comes next):
  - ``build_edges``  — caller→callee edges (the real graph). The count here is a
    *projection of the graph*: an incoming count is not independently maintainable
    without the edge set (which files reference X). When edges land, the count
    falls out of them trivially.

Reference definition (file granularity, honest + cheap): a FILE ``A`` references
FILE ``B`` when ``A`` mentions — in its imports or any of its own symbol
bodies/signatures/docs — a symbol name that ``B`` *defines*. ``B``'s
``references_count`` = the number of DISTINCT other files that so reference it.
Self-references are excluded (a file doesn't count as its own referencer).
"""

from __future__ import annotations

import re

from ._schema import FileInfo

# Names that occur in this set are "references" a file makes to another file's
# defined symbols. Cheap derived-string tokenizer: pull identifier runs so
# `from x import Foo` contributes `Foo` and `Settings.consol_model` contributes
# `Settings` + `consol_model` (the latter won't resolve to a defined symbol, so
# it harmlessly no-ops).
_REF_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
# Python/TS import-statement keywords that are never symbol references. Used only by
# the backward-compat fallback when a hand-built FileInfo lacks structured import_names.
_IMPORT_KEYWORDS = {
    "from", "import", "as", "export", "default", "type", "return", "def", "class",
}


def _defined_name_map(files: list[FileInfo]) -> dict[str, str]:
    """Map every defined symbol name → its defining file path.

    First-writer wins (a name defined in multiple files resolves to the first
    alphabetical file that defines it — an honest approximation for the count;
    true disambiguation is the graph phase's job).
    """
    defined_by: dict[str, str] = {}
    for f in files:
        for name in f.symbols:
            # skip dunder/module-internal noise that would mis-attribute
            # references to the wrong file (e.g. `_private` used everywhere)
            if name.startswith("__") and name.endswith("__"):
                continue
            defined_by.setdefault(name, f.path)
    return defined_by


def _file_referenced_names(f: FileInfo) -> set[str]:
    """Distinct identifier names a file references (imports + its own symbol text).

    Imports come from the STRUCTURED ``import_names`` (the imported symbol names,
    populated from tree-sitter ``ImportInfo.items``/``alias`` at analyze time) — NOT a
    regex over the raw import string. Regex over the string leaks module-path basenames
    (``. _store``, ``./lib``) as if they were symbols, which then false-collide with any
    file that happens to define a symbol of that name (Phase-1.1 bug, fixed here). Module
    references proper are the import-resolution job; this name-match layer only claims
    *symbol* references, so it stays language-agnostic and honest.

    The file's own symbol bodies/signatures/docs may *use* other files' symbols; capture
    those identifiers too (name tokens from body text, an acceptable approximation).
    """
    names: set[str] = set(f.import_names)
    # Backward-compat fallback: hand-built FileInfo (e.g. older tests) populate
    # `imports` with bare referenced names but no `import_names`. Only then derive
    # from the import strings — stripping path segments/keywords/attribute tails so
    # we never re-introduce the basename-leak bug. Analyzed files always carry
    # `import_names`, so this branch is dead in production.
    if not f.import_names:
        for imp in f.imports:
            for m in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", imp):
                tok = m.group()
                if tok in _IMPORT_KEYWORDS:
                    continue
                if m.start() > 0 and imp[m.start() - 1] == ".":
                    continue  # dotted module path segment
                names.add(tok)
    for sym in f.searchable_symbols:
        for field in (sym.signature, sym.body, sym.doc):
            if not field:
                continue
            # A name is a *symbol reference* only when it does NOT sit on the right
            # of an attribute access (`self._store`, `x.shared`, `Settings.MODE`).
            # The RHS token is a member/attribute of some owner — resolving it as a
            # standalone module-level name false-collides (e.g. `self._store` matching
            # visual's `_store` module var). Tokenize WITH the preceding char so we can
            # drop `.X` tails.
            for m in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", field):
                if m.start() > 0 and field[m.start() - 1] == ".":
                    continue  # right-hand side of attribute access -> not a symbol ref
                names.add(m.group())
    return names


def reference_counts(files: list[FileInfo]) -> dict[str, int]:
    """Inbound-reference counts per file path (in-degree at file granularity).

    ``A`` references ``B`` when A mentions (imports / bodies) a symbol B defines.
    Returns ``{path: count}`` where count = number of DISTINCT other files that
    reference at least one symbol defined by that path. Self-reference excluded.
    """
    defined_by = _defined_name_map(files)
    # Every file starts at 0; only those actually referenced by others go positive.
    counts: dict[str, int] = {f.path: 0 for f in files}
    for f in files:
        for name in _file_referenced_names(f):
            defining_file = defined_by.get(name)
            if defining_file is not None and defining_file != f.path:
                counts[defining_file] = counts.get(defining_file, 0) + 1
    return counts


def apply_references(files: list[dict]) -> None:
    """Populate ``references_count`` on the manifest records IN PLACE.

    ``files`` is the raw list of dict records as written by ``_scan.py`` for
    ``manifest.json`` (each carries full ``symbols`` as asdict(SymbolInfo) plus
    ``imports``). Reads each record's ``symbols``/``imports`` to compute inbound
    counts, then sets ``record['references_count']`` on the *defining* record.
    Returns ``None`` (mutates the list so the caller's artifacts already carry the
    counts). The scan driver then mirrors the counts onto the lean index HEAD rows
    by path — reference counting needs symbol text, which the head deliberately
    does not carry.
    """
    infos = [_fileinfo_from_record(r) for r in files]
    counts = reference_counts(infos)
    for record in files:
        record["references_count"] = counts.get(record["path"], 0)


def _fileinfo_from_record(record: dict) -> FileInfo:
    """Build a ``FileInfo`` from a raw manifest record dict (reverse of asdict).

    Rehydrates ``symbols`` (stored as full dicts) back into ``SymbolInfo`` objects so
    ``reference_counts`` can walk their fields uniformly, and keeps the ``span`` as a
    ``Span`` when present. ``children``/``parent`` are dropped for the count pass
    (references walk top-level field text, not the tree).
    """
    from ._schema import Span, SymbolInfo

    syms: list[SymbolInfo] = []
    for s in record.get("symbols") or []:
        span = s.get("span")
        if isinstance(span, dict):
            span = Span(**{k: v for k, v in span.items() if k in ("start_byte", "end_byte", "start_line", "end_line")})
        syms.append(
            SymbolInfo(
                name=s.get("name", ""),
                qualified=s.get("qualified"),
                kind=s.get("kind", "OTHER"),
                signature=s.get("signature"),
                doc=s.get("doc"),
                body=s.get("body"),
                span=span,
                visibility=s.get("visibility"),
                children=[],
                parent=s.get("parent"),
            )
        )
    return FileInfo(
        path=record.get("path", ""),
        language=record.get("language"),
        doc=record.get("doc"),
        symbols=[s.name for s in syms if s.name],
        imports=record.get("imports") or [],
        import_names=record.get("import_names") or [],
        references_count=record.get("references_count", 0),
        searchable_symbols=syms,
        unparsed=record.get("unparsed", False),
    )


def build_edges(
    files: list[FileInfo],
    sources: dict[str, str] | None = None,
) -> list[dict]:
    """Symbol-bearing edges (file→file), reference- and/or call-level.

    For every file ``A``, resolve each identifier ``A`` references (imports +
    its own symbol bodies/signatures/docs) to the file that *defines* it; emit
    one edge ``{source, target, symbols}`` per distinct (``A`` → defining-file)
    pair, where ``symbols`` is the set of names carried across that edge.
    Self-edges (``A`` → ``A``) are excluded, as are dunder names.

    Call refinement (Phase 1.2): when ``sources`` (repo-relative path → raw
    source) is supplied, ``_calls.call_edge_set`` re-parses each file and finds
    which edges are *provable* call reachability (a call site resolved through
    an import to a defining symbol). Those edges are marked ``kind: "call"``
    with ``call_symbols`` = the subset of ``symbols`` that are verified calls.
    Edges that resolve—or files whose source is absent—stay ``kind: "reference"``
    (``call_symbols`` empty). This is the honest B1 carve: never guess a call;
    uncertain resolution remains reference.

    With no ``sources``, every edge is ``kind: "reference"`` and the output is
    byte-identical to the pre-Phase-1.2 build_edges (graceful degradation).

    Determinism: iteration order is the input file order; per-file names are
    sorted at emission. Callers must sort the result for a stable global order
    (tests do).
    """
    from ._calls import call_edge_set

    defined_by = _defined_name_map(files)
    edges: list[dict] = []
    seen: set[tuple[str, str]] = set()

    # call reachability, keyed by (source, target) -> {called symbols}
    call_map: dict[tuple[str, str], set[str]] = {}
    if sources:
        call_map = call_edge_set(files, sources)

    for f in files:
        by_target: dict[str, set[str]] = {}
        for name in _file_referenced_names(f):
            defining_file = defined_by.get(name)
            if defining_file is None or defining_file == f.path:
                continue
            by_target.setdefault(defining_file, set()).add(name)
        for target, names in by_target.items():
            key = (f.path, target)
            if key in seen:
                continue
            seen.add(key)
            called = call_map.get(key, set())
            kind = "call" if called else "reference"
            edges.append(
                {
                    "source": f.path,
                    "target": target,
                    "kind": kind,
                    "symbols": sorted(names),
                    "call_symbols": sorted(called),
                }
            )
    return edges
