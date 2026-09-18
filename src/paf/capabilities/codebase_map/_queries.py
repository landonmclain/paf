"""Per-language tree-sitter query recovery for what ``process()`` leaves empty.

``tree-sitter-language-pack.process()`` never fills ``docstrings`` (``doc``/
``doc_comment``) even on 1.15.7 (the version paf declares + runs) — and until the
1.15.7 upgrade also left ``signature`` empty (on 1.14.3, the only installable wheel back
then) — see the ``tree-sitter-language-pack`` skill (FILL THE BLANKS) and the Archive
note *Tree-Sitter-Empty-Fields-and-Custom-Query-Recovery*. We verified (2026-08-22)
that these ARE recoverable with ``tree_sitter.Query``+``QueryCursor`` against the
REAL bundled grammar. This module is that thin per-language recovery layer: ONE small
query per language per piece (node vocab differs per language), NOT a general walker.

Since the 1.15.7 upgrade (2026-08-23) ``signature``/``module-signature`` recovery here is
FALLBACK ONLY — ``_analyze.py`` prefers native ``StructureItem.signature``; this recovery
kicks in where native is empty (classes → ``signature=None``, grammars 1.15.7 still blanks).
Docstring + body recovery remain the primary path (native still leaves them empty).

Read-model philosophy (memory): captured nodes are cheap byte-offset pointers into
``source``; value cleaning (strip quotes/``*``) and comment→declaration association we
do here in Python, not via ``#strip!``/``#select-adjacent!`` predicates (unsupported
in tree-sitter 0.26). Swap-ready: when a wheel with working native
``process().signature`` / ``.docstrings`` ships, ``_analyze.py`` flips to the native
fields and this module's per-language tables become unnecessary (schema keeps the
``signature``/``doc`` fields regardless).

Verified gotchas (both tripped us — see skill):
- ``Query.captures`` does NOT exist in tree-sitter 0.26.0; use ``QueryCursor(q).captures``.
- ``QueryCursor(q).captures(root)`` returns a **dict** ``{capture_name: [node,...]}``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tree_sitter import Query, QueryCursor, Tree
from tree_sitter_language_pack import get_language, get_parser

# ---------------------------------------------------------------------------
# Per-language grammar vocabulary (VERIFIED against bundled grammar, 2026-08-22)
# ---------------------------------------------------------------------------
# Each entry: a `header` query that captures `@name` and `@params` for ONE function
# declaration (matched, then filtered to the symbol's header byte-region), a `ret`
# query capturing `@ret` (or "" when the language has no separable return node —
# e.g. JS is untyped, C nests name/params in a declarator we handle via header),
# and the `retsep` used to append the return type. Node vocab differs per language
# (Python `parameters` vs JS/TS `formal_parameters`; PHP `name` field; C name inside
# `function_declarator`) — this table IS the "thin per-language query set".

_LANG = {
    "python": {
        "header": "(function_definition name: (identifier) @name parameters: (parameters) @params)",
        "ret": "(function_definition return_type: (type) @ret)",
        "retsep": " -> ",
    },
    "typescript": {
        "header": "(function_declaration name: (identifier) @name parameters: (formal_parameters) @params)",
        "ret": "(function_declaration return_type: (type_annotation) @ret)",
        "retsep": ": ",
    },
    "tsx": {
        "header": "(function_declaration name: (identifier) @name parameters: (formal_parameters) @params)",
        "ret": "(function_declaration return_type: (type_annotation) @ret)",
        "retsep": ": ",
    },
    "javascript": {
        "header": "(function_declaration name: (identifier) @name parameters: (formal_parameters) @params)",
        "ret": "",  # JS is untyped — no return type node
        "retsep": ": ",
    },
    "php": {
        "header": "(function_definition name: (name) @name parameters: (formal_parameters) @params)",
        "ret": "(function_definition return_type: (_) @ret)",
        "retsep": ": ",
    },
    "go": {
        "header": "(function_declaration name: (identifier) @name parameters: (parameter_list) @params)",
        "ret": "(function_declaration result: (_) @ret)",
        "retsep": " ",
    },
    "rust": {
        "header": "(function_item name: (identifier) @name parameters: (parameters) @params)",
        "ret": "(function_item return_type: (_) @ret)",
        "retsep": " -> ",
    },
    "java": {
        "header": "(method_declaration name: (identifier) @name parameters: (formal_parameters) @params)",
        # Java return type is inline before the name (no `return_type` field, like C).
        # Capture it would need a fragile leading-type query; omit for v1 (honest).
        "ret": "",
        "retsep": " ",
    },
    "c": {
        # C's name + params live inside a `function_declarator`, not direct fields.
        "header": "(function_definition declarator: (function_declarator name: (identifier) @name parameter_list: (parameter_list) @params))",
        "ret": "",  # leading return type is a separate `type:` field; we skip for v1
        "retsep": " ",
    },
}

# Python's docstring is a `string` node (old grammar) direct child of the symbol's
# `block` (the skill-verified recipe). One disjunction covers function/method AND
# class bodies (class docstrings are often the richest description — always capture).
# For every other language doc comments are comment nodes — handled by Python-side
# association in `docstring_for`.
_PYTHON_DOCSTRING = (
    "[(function_definition body: (block . (string) @doc))"
    " (class_definition body: (block . (string) @doc))]"
)

# Module-level docstring: the leading `string` node that is a direct child of the
# `module` root (old grammar — no `expression_statement` wrapper, verified 2026-08-22).
_MODULE_DOCSTRING = "(module . (string) @doc)"

# Module-level named assignments (NAME = ... / NAME: Type = ...). Enumerate every
# `assignment` node that is a DIRECT child of `module` (nested fns/class attr
# assignments live under block/class_definition, so they're excluded). The
# per-assignment name/type is recovered by a scoped sub-query (see
# `module_assignments`) — robust against the `@type` list not aligning with
# `@name` when annotated and unannotated assignments are mixed.
_MODULE_ASSIGN = "(module (assignment) @assign)"
_MODULE_ASSIGN_SUB = "(assignment left: (identifier) @name (type (_) @type)?)"

# TS/TSX/JS module-level named assignments: `const x = ...` / `let x = ...` /
# `export const x = ...`. Tree shapes differ from Python (lexical_declaration +
# variable_declarator, not bare `assignment`); and `export const x = ...` is an
# `export_statement` WRAPPING a `lexical_declaration` (export is an anonymous child,
# not a named field — using a field name here is invalid, see the probe).
#
# CRITICAL — name must be captured IN THE SAME match as the decl, anchored to the
# lexical_declaration's OWN direct `variable_declarator`. A separate byte-scoped
# sub-query is WRONG here: a `const X = () => { ...long body... }` arrow component
# spans thousands of bytes and the scoped sub-query matches EVERY nested
# `variable_declarator` inside that body (isUser, widthClass, …), returning a
# nondeterministic one (tree-sitter gives children in hash order). Anchoring the
# `variable_declarator name:` to the lexical_declaration root is deterministic
# (verified 15x stable across fresh recoveries on a big TSX component).
#
# The enumerator is ANCHORED TO `program` (both disjuncts) so ONLY module-scope
# declarations match — without the anchor, `(lexical_declaration)` also matches
# locals inside function/arrow bodies. `export default defineConfig(...)` has no
# `variable_declarator`/identifier -> correctly excluded (not named). `export
# type X = ...` is a `type_alias_declaration` -> excluded (v2 concern). NOTE: with
# tree-sitter 0.26 QueryCursor, a node type captured in multiple disjuncts must be
# in a single combined pattern (not separate queries) or it dedupes to nothing.
_MODULE_ASSIGN_TS = (
    "[(program (lexical_declaration (variable_declarator name: (identifier) @name) "
    " (type_annotation)? @type) @decl)"
    " (program (export_statement (lexical_declaration "
    " (variable_declarator name: (identifier) @name) (type_annotation)? @type)) @decl)]"
)
# JS grammar has NO `type_annotation` node at all (JS is untyped) — same enumerator
# minus the type capture (the `(type_annotation)?` is invalid node type in JS).
_MODULE_ASSIGN_JS = (
    "[(program (lexical_declaration (variable_declarator name: (identifier) @name)) @decl)"
    " (program (export_statement (lexical_declaration "
    " (variable_declarator name: (identifier) @name))) @decl)]"
)

# Comment-node type(s) per language (VERIFIED: rust/java use line_comment/block_comment,
# not `comment`; the rest use `comment`).
_COMMENT_NODE = {
    "python": "comment",
    "typescript": "comment",
    "tsx": "comment",
    "javascript": "comment",
    "php": "comment",
    "go": "comment",
    "c": "comment",
    "rust": "line_comment",
    "java": "block_comment",
}

# Line-comment syntax per language (used to strip markers for the doc text).
_LINE_COMMENT_MARKERS = {
    "python": "#",
    "typescript": "//",
    "tsx": "//",
    "javascript": "//",
    "php": "// ",
    "go": "//",
    "rust": "//",
    "java": "//",
    "c": "//",
}


@dataclass
class QueryRecovery:
    """Per-file recovery object: parse once, answer signature/docstring per symbol span.

    One instance per analyzed file. ``source`` is the file text; constructs a fresh
    parser + QueryCursor per language lazily so ``_analyze.py`` merely calls
    ``signature_for(span, body_span)`` / ``docstring_for(span, body_span)``.
    """

    language: str
    source: str
    _parsed: Tree | None = field(init=False, repr=False, default=None)
    _sig_cache: dict = field(init=False, repr=False, default_factory=dict)
    _doc_cache: dict = field(init=False, repr=False, default_factory=dict)
    _bsrc: bytes = field(init=False, repr=False)  # BYTES copy of source (see _slice)

    def __post_init__(self) -> None:
        # Byte offsets from tree-sitter MUST be sliced against a bytes copy, never the
        # decoded str (char-indexed) — any non-ASCII char before the offset drifts the
        # slice. See skill Pitfall #8 / vault History "BYTE OFFSETS vs CHAR INDICES".
        self._bsrc = self.source.encode("utf-8")

    def _slice(self, start_byte: int, end_byte: int) -> str:
        """Slice source bytes by TREE-SITTER BYTE offsets and decode to str."""
        return self._bsrc[start_byte:end_byte].decode("utf-8", errors="replace")

    def body_for(self, body_span, *, max_bytes: int = 16_384) -> str | None:
        """Slice a symbol's body source bytes (weight-1 searchable field, §3.7.2).

        ``body_span`` is our ``Span`` (byte offsets from ``process().structure``;
        sliced against ``self._bsrc`` bytes for the byte-offset-vs-char rule). Bodies
        are capped at ``max_bytes`` so a giant function can't balloon the manifest —
        a body token is weight-1 search signal, not a full-text archive.
        """
        if not body_span or body_span.start_byte is None or body_span.end_byte is None:
            return None
        start = body_span.start_byte
        end = min(body_span.end_byte, start + max_bytes)
        if end <= start:
            return None
        return self._slice(start, end)

    def _tree(self) -> Tree:
        if self._parsed is None:
            self._parsed = get_parser(self.language).parse(self._bsrc)
        tree = self._parsed
        assert tree is not None
        return tree

    def _captures(self, query: str, region: tuple[int, int] | None = None) -> dict:
        """Run ``query`` on the file; return {name: [(start,end),...]} inside `region`."""
        tree = self._tree()
        q = Query(get_language(self.language), query)
        out: dict[str, list[tuple[int, int]]] = {}
        lo, hi = region if region else (0, 10**18)
        for name, nodes in QueryCursor(q).captures(tree.root_node).items():
            out[name] = [
                (n.start_byte, n.end_byte)
                for n in nodes
                if n.end_byte > lo and n.start_byte < hi
            ]
        return out

    def signature_for(self, span, body_span) -> str | None:
        """Assemble a signature line (``name(params) -> ret``) for the symbol's span.

        ``span``/``body_span`` carry BYTE offsets from ``process().structure``. We
        capture a function header whose name+params fall inside the symbol's header
        byte-region (span.start → body start) and reassemble ``name(params)
        [retsep ret]``. All byte offsets are sliced against ``self._bsrc`` (a bytes
        copy) — NEVER against the decoded str (see skill Pitfall #8 / vault History
        note: byte offsets ≠ char indices). Uses the per-language ``header``/``ret``
        queries in ``_LANG``. FALLBACK ONLY since the 1.15.7 upgrade — ``_analyze.py``
        prefers native ``StructureItem.signature`` (normalized bare by ``_native_signature``)
        and calls this only for symbols/grammars where native is empty (e.g. classes,
        which stay ``signature=None``, and languages 1.15.7 still blanks).
        """
        cfg = _LANG.get(self.language)
        if not cfg or not span or span.start_byte is None:
            return None
        header_end = (
            body_span.start_byte
            if body_span and body_span.start_byte
            else (span.end_byte or span.start_byte)
        )
        region = (span.start_byte, header_end)
        if header_end <= span.start_byte:  # degenerate — nothing between span and body
            return None

        caps = self._captures(cfg["header"], region)
        name_spans = caps.get("name") or []
        param_spans = caps.get("params") or []
        if not name_spans:
            return None
        s, e = name_spans[0]
        name = self._slice(s, e)

        params_txt = ""
        if param_spans:
            ps, pe = param_spans[0]
            params_txt = self._slice(ps, pe)

        sig = f"{name}{params_txt}"
        retq = cfg.get("ret") or ""
        if retq:
            rcaps = self._captures(retq, region)
            for rs, re_ in rcaps.get("ret", []):
                ret_txt = self._slice(rs, re_).strip(": ").strip()
                if ret_txt:
                    sig += f"{cfg['retsep']}{ret_txt}"
                break
        return sig

    def docstring_for(self, span, body_span) -> str | None:
        """Recover a leading docstring for the symbol span.

        Python: the verified ``(block . (string) @doc)`` query, restricted to the
        symbol's body region. Other languages: the nearest comment node immediately
        preceding the symbol's start byte (Python-side association — more robust than
        the unsupported #select-adjacent! predicate).
        """
        if not span or span.start_byte is None:
            return None
        # --- C-style comment-based languages (JS/TS/TSX/Java/Rust/Go/PHP/C) ---
        if self.language != "python":
            return self._preceding_comment(span.start_byte)
        # --- Python: string-as-direct-child-of-block in the body region ---
        if not body_span or body_span.start_byte is None:
            return None
        caps = self._captures(
            _PYTHON_DOCSTRING, (body_span.start_byte, body_span.end_byte)
        )
        doc_spans = caps.get("doc", [])
        if not doc_spans:
            return None
        s, e = doc_spans[0]
        return _dedent(_strip_quotes(self._slice(s, e)))

    def module_docstring(self) -> str | None:
        """Recover the module-level docstring (python) or None.

        Python: the leading ``string`` direct child of ``module``. Non-python: no
        module-level doc extraction in v1 (the module doc is carried on the file via
        ``FileInfo.doc`` where supported).
        """
        if self.language != "python":
            return None
        caps = self._captures(_MODULE_DOCSTRING)
        doc_spans = caps.get("doc", [])
        if not doc_spans:
            return None
        s, e = doc_spans[0]
        return _dedent(_strip_quotes(self._slice(s, e)))

    def module_assignments(self) -> list[tuple[str, int, int, str | None]]:
        """Recover module-level named assignments as ``(name, start, end, type)``.

        ``start``/``end`` are the whole assignment node's byte span (so ``_analyze.py``
        can build a ``Span`` for ``read_block`` and slice the assignment source as the
        weight-1 ``body``). ``type`` is the type annotation text (``NAME: TYPE = ...``)
        or ``None`` for unannotated assignments. Nested assignments (inside functions /
        class bodies) are excluded — only direct children of ``module`` / ``program``
        match. Dunder names (``__x__``) and the throwaway ``_`` are skipped, but
        leading-single-underscore names (``_MAX_SOURCE_BYTES``) ARE kept (they're real
        constants).

        Python: ``assignment`` nodes that are direct children of ``module``. Typescript
        / tsx / javascript: ``lexical_declaration`` nodes at ``program`` scope, bare or
        wrapped in ``export_statement`` — they carry ``variable_declarator`` names.
        """
        if self.language == "python":
            return self._python_module_assignments()
        if self.language in ("typescript", "tsx", "javascript"):
            return self._ts_module_assignments()
        return []

    def _python_module_assignments(self) -> list[tuple[str, int, int, str | None]]:
        caps = self._captures(_MODULE_ASSIGN)
        out: list[tuple[str, int, int, str | None]] = []
        for as_, ae in caps.get("assign", []):
            # scoped sub-query: name + optional annotation within THIS assignment.
            # Scoping avoids list-alignment drift when annotated/unannotated mix.
            sub = self._captures(_MODULE_ASSIGN_SUB, (as_, ae))
            nm = sub.get("name")
            if not nm:
                continue
            ns, ne = nm[0]
            name = self._slice(ns, ne)
            if name == "_" or (name.startswith("__") and name.endswith("__")):
                continue
            typ = sub.get("type")
            type_ann = self._slice(typ[0][0], typ[0][1]).strip() if typ else None
            out.append((name, as_, ae, type_ann))
        return out

    def _ts_module_assignments(self) -> list[tuple[str, int, int, str | None]]:
        # Combined enumerator emits @decl/@name/(@type) captures in source order,
        # ONE of each per match (a lexical_declaration has a single direct
        # variable_declarator). So index i of @decl ↔ index i of @name (verified:
        # each name[i] is byte-inside decl[i]). Type annotations are mapped to their
        # containing decl by byte span (they follow the name inside the decl).
        query = (
            _MODULE_ASSIGN_JS if self.language == "javascript" else _MODULE_ASSIGN_TS
        )
        caps = self._captures(query)
        decls = caps.get("decl", [])
        names = caps.get("name", [])
        types = caps.get("type", [])
        # map each type_annotation to the decl whose span contains it (one per decl)
        type_by_decl: dict[int, tuple[int, int]] = {}
        for ts_, te in types:
            containing = [ds for (ds, de) in decls if ds <= ts_ < de]
            if containing:
                type_by_decl[min(containing)] = (ts_, te)

        out: list[tuple[str, int, int, str | None]] = []
        for i, (ds, de) in enumerate(decls):
            if i >= len(names):
                continue
            name = self._slice(names[i][0], names[i][1])
            if name == "_" or (name.startswith("__") and name.endswith("__")):
                continue
            type_ann = None
            typ = type_by_decl.get(ds)
            if typ:
                t = self._slice(typ[0], typ[1]).strip()
                if t.startswith(":"):
                    t = t[1:].strip()
                type_ann = t or None
            out.append((name, ds, de, type_ann))
        return out

    def _preceding_comment(self, start_byte: int) -> str | None:
        """Nearest comment node that ends immediately above ``start_byte``.

        Uses the language's comment-node type(s) (``_COMMENT_NODE``). Association is
        done here in Python (simpler + more robust than the unsupported
        ``#select-adjacent!`` predicate).
        """
        node_type = _COMMENT_NODE.get(self.language, "comment")
        caps = self._captures(f"({node_type}) @c")
        best: tuple[int, int] | None = None
        for cs, ce in caps.get("c", []):
            if ce <= start_byte and (best is None or ce > best[1]):
                best = (cs, ce)
        if best is None:
            return None
        gap = start_byte - best[1]
        if gap > 400:  # too far above the symbol — likely unrelated
            return None
        return _strip_comment_markers(self._slice(best[0], best[1]))

    def _preceding_comment_block(self, start_byte: int) -> str | None:
        """Aggregate the contiguous comment lines immediately above ``start_byte``.

        Multi-line ``#`` / ``//`` comment runs are one `comment` node PER LINE in
        python (and most grammars), so ``_preceding_comment`` (single closest node)
        returns only the last line. This walks backwards over comment nodes that are
        byte-adjacent (no code between them) and joins them — so the rich doc comment
        above a symbol (e.g. registry.py's ``CAPABILITY_TYPES`` block) is captured
        whole. Returns None when no comment is within a sane distance above the symbol.
        """
        node_type = _COMMENT_NODE.get(self.language, "comment")
        caps = self._captures(f"({node_type}) @c")
        comments = sorted(
            (cs, ce) for (cs, ce) in caps.get("c", []) if ce <= start_byte
        )
        if not comments:
            return None
        # start from the nearest comment; walk backward joining byte-adjacent comment
        # lines. Two consecutive comment LINES are adjacent when the next line's START
        # sits a byte (the newline) after the previous line's END — i.e. `cs - prev_ce <= 1`.
        best: list[tuple[int, int]] = [comments[-1]]
        cursor = comments[-1][0]  # START of nearest comment
        for cs, ce in reversed(comments[:-1]):
            if (
                cursor - ce <= 1
            ):  # this comment line ends right where the nearer one begins
                best.append((cs, ce))
                cursor = cs
            else:
                break
        gap = start_byte - best[0][1]
        if gap > 400:
            return None
        # best is reversed (farthest->nearest); join in source order
        parts = [_strip_comment_markers(self._slice(s, e)) for (s, e) in reversed(best)]
        return " ".join(p for p in parts if p) or None


def _strip_quotes(raw: str) -> str:
    """Strip a Python string literal to its content (handles triple+single quotes)."""
    txt = raw.strip()
    for q in ('"""', "'''", '"', "'"):
        if txt.startswith(q) and txt.endswith(q) and len(txt) >= 2 * len(q):
            return txt[len(q) : -len(q)].strip()
    return txt


def _dedent(raw: str) -> str:
    """Clean a captured docstring: dedent continuation lines + trim blank lines.

    ``textwrap.dedent`` alone won't work — the first line is at column 0 while
    continuation lines carry the body's indentation, so the common prefix is 0.
    ``inspect.cleandoc`` handles exactly this (dedents by the first non-blank line,
    strips leading/trailing blank lines). ``raw`` is already quote-stripped.
    """
    from inspect import cleandoc

    return cleandoc(raw)


def _strip_comment_markers(raw: str) -> str:
    """Best-effort strip of ``///`` / ``//`` / ``*`` / ``#`` / ``/* */`` comment markers.

    Handles JSDoc/phodoc blocks (``/** ... */``), Rust ``///`` doc-lines, Go/JS ``//``,
    and Python ``#``. Leading AND trailing block markers are removed; a leading ``*``
    on each line (DocBlock asterisk column) is dropped.
    """
    lines = []
    for line in raw.splitlines():
        s = line.strip()
        # trailing block-close / any trailing asterisk run (``*/`` or lone ``*`` col)
        if s.endswith("*/"):
            s = s[:-2].strip()
        s = s.rstrip("*/").strip() if s.endswith("*") else s
        # opening block marker
        if s.startswith("/*"):
            s = s[2:].strip()
        # DocBlock leading-asterisk column
        while s.startswith("*"):
            s = s[1:].strip()
        # line markers, greedy so "///" fully consumed (incl. residual single "/")
        for marker in ("//", "#", "--"):
            while s.startswith(marker):
                s = s[len(marker) :].strip()
        while s.startswith("/"):
            s = s[1:].strip()
        lines.append(s)
    return " ".join(x for x in lines if x).strip()
