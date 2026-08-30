"""Call-site extraction + conservative call-edge resolution (Phase 1.2).

Upgrades ``build_edges`` from the honest-but-coarse *reference* level ("file A
*mentions* a symbol B defines") to true *call* edges ("A *invokes* symbol ``s``
defined in B") — but only where the callee resolves **unambiguously** to a
single defining symbol. See the B1 scope decision (Landon, 2026-08-29): resolve
calls whose callee maps through a *module import* to a defining symbol; leave
instance/param/self-method calls and unresolved/ambiguous names as
``kind: "reference"`` (never downgrade certainty, never guess a call).

Package-root note: paf scans with ``src/paf`` as ``scan_root``, so the dotted
module ``paf.capabilities.memory._watermark`` is the code under ``capabilities/
memory/_watermark.py`` — the ``paf.`` package prefix is *above* the scan root and
is stripped before mapping dots->slashes. ``module_to_path`` handles that plus
relative imports (``from ._schema import ...`` inside ``capabilities/codebase_map/``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from tree_sitter_language_pack import ProcessConfig, process

from ._queries import QueryRecovery
from ._schema import FileInfo

# --- per-language call-site queries (VERIFIED against bundled grammar) ---------
# Python: `call` has a `function:` field; `(_) @callee` captures the whole callee
# expression (bare identifier for `foo(...)` / `Klass(...)`, dotted for
# `B.build_it(...)`). TypeScript/JS use `call_expression`; `new X()` is a separate
# `new_expression` whose constructor identifier we also capture as a call to X's
# constructor.
_CALL_QUERIES = {
    "python": "(call function: (_) @callee)",
    "typescript": "(call_expression function: (_) @callee)",
    "tsx": "(call_expression function: (_) @callee)",
    "javascript": "(call_expression function: (_) @callee)",
}
_CONSTRUCT_QUERIES = {
    "python": "",
    "typescript": "(new_expression constructor: (_) @callee)",
    "tsx": "(new_expression constructor: (_) @callee)",
    "javascript": "(new_expression constructor: (_) @callee)",
}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Language ids we can run call extraction for. Languages without a grammar entry
# here (php/go/rust/java/c) are reference-only for v1 — honest, no guessing.
_CALL_LANGS = set(_CALL_QUERIES)


@dataclass
class CallExtractor:
    """Per-file call/import extraction (parse once, answer many questions)."""

    language: str
    source: str
    # import maps, populated in __post_init__:
    _module_aliases: dict[str, str] = field(init=False, repr=False)
    _direct_items: dict[str, str] = field(init=False, repr=False)
    _plain_modules: list[str] = field(init=False, repr=False)
    _rec: QueryRecovery = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rec = QueryRecovery(language=self.language, source=self.source)
        self._module_aliases, self._direct_items = self._build_import_maps()
        self._plain_modules = self._build_plain_modules()

    # -- imports ---------------------------------------------------------------
    def _build_import_maps(self) -> tuple[dict[str, str], dict[str, str]]:
        """Recover ``{alias: module}`` (import x as A) and ``{name: module}``
        (from m import name). Uses tree-sitter ``process().imports`` which keeps
        statement-level ``alias``/``items`` but drops per-item aliases (the
        honest-fallback boundary for v1: an aliased local that isn't otherwise
        defined resolves to nothing and stays ``reference``).
        """
        module_aliases: dict[str, str] = {}
        direct_items: dict[str, str] = {}
        try:
            res = process(
                self.source,
                ProcessConfig(language=self.language, structure=False, imports=True),
            )
        except Exception:  # noqa: BLE001 — parse noise -> no imports (safe)
            return module_aliases, direct_items
        for imp in res.imports:
            stmt = (imp.source or "").strip()
            items = list(getattr(imp, "items", None) or [])
            alias = getattr(imp, "alias", None)
            if not stmt:
                continue
            if stmt.startswith("import ") and " as " in stmt and alias:
                # import pkg.mod as A  ->  A -> pkg.mod
                mod = stmt[len("import "):].split(" as ")[0].strip()
                module_aliases[alias] = mod
            elif stmt.startswith("from "):
                mod = stmt[len("from "):].split(" ")[0].strip()  # may be relative
                for it in items:
                    if it != "*" and it != "as":
                        direct_items[it] = mod
        return module_aliases, direct_items

    def _build_plain_modules(self) -> list[str]:
        """Recover ``import pkg.mod`` (no alias) module paths — used only as a
        last resort for dotted-callee root resolution; rare in paf."""
        try:
            res = process(
                self.source,
                ProcessConfig(language=self.language, structure=False, imports=True),
            )
        except Exception:  # noqa: BLE001
            return []
        out: list[str] = []
        for imp in res.imports:
            stmt = (imp.source or "").strip()
            if stmt.startswith("import ") and getattr(imp, "alias", None) is None:
                out.extend(it for it in (getattr(imp, "items", None) or []) if it != "*")
        return out

    # -- call sites ------------------------------------------------------------
    def call_callees(self) -> tuple[list[str], list[str]]:
        """Return ``(direct_callees, qualified_callees)`` across the file.

        ``direct``: bare identifiers invoked (``foo(...)``, ``Klass(...)``).
        ``qualified``: dotted callee texts (``B.build_it(...)``, ``a.b.deep(...)``).
        """
        direct: list[str] = []
        qualified: list[str] = []
        qry = _CALL_QUERIES.get(self.language, "")
        if qry:
            caps = self._rec._captures(qry)
            for (s, e) in caps.get("callee", []):
                text = self._rec._slice(s, e)
                if "." in text:
                    qualified.append(text)
                elif _IDENT_RE.fullmatch(text):
                    direct.append(text)
        cq = _CONSTRUCT_QUERIES.get(self.language, "")
        if cq:
            caps = self._rec._captures(cq)
            for (s, e) in caps.get("callee", []):
                text = self._rec._slice(s, e)
                if _IDENT_RE.fullmatch(text):
                    direct.append(text)
        return direct, qualified

    # -- resolution ------------------------------------------------------------
    def resolve_callee(
        self,
        callee: str,
        defined_by: dict[str, str],
        module_to_path,
        importer_path: str,
        file_set: set[str],
    ) -> str | None:
        """Resolve one callee to the repo-relative path that defines it, or None.

        Bare ``foo``: ``foo`` must map through a direct-item import to a module
        that defines a symbol named ``foo``.
        Dotted ``root.attr``: ``root`` resolves (module alias / direct item) to a
        module whose file defines ``attr``.
        Never guesses: ambiguous/unresolved -> None (caller keeps ``reference``).
        """
        if "." in callee:
            parts = _IDENT_RE.findall(callee)
            if len(parts) < 2:
                return None
            root, attr = parts[0], parts[-1]
            mod = self._mapped_module(root)
            if mod is None:
                return None
            path = module_to_path(mod, importer_path, file_set)
            if path is not None and defined_by.get(attr) == path:
                return path
            return None
        mod = self._mapped_module(callee)
        if mod is None:
            return None
        path = module_to_path(mod, importer_path, file_set)
        if path is not None and defined_by.get(callee) == path:
            return path
        return None

    def _mapped_module(self, name: str) -> str | None:
        """Dotted/relative module a local ``name`` refers to, or None."""
        if name in self._module_aliases:
            return self._module_aliases[name]
        if name in self._direct_items:
            return self._direct_items[name]
        return None


def module_to_path(dotted: str, importer_path: str, file_set: set[str]) -> str | None:
    """Map a dotted (or relative) module reference to a repo-relative file path.

    Rules (conservative — None when ambiguous, never guess):
      - strip a leading ``paf.`` package prefix (the plugin's code root == scan root)
      - relative imports: ``.foo`` / ``..foo`` resolved against the importer's dir
      - ``a.b.c`` -> try ``a/b/c.py`` then ``a/b/__init__.py`` then ``a/b.py`` (``c``
        may be a symbol in ``a/b`` rather than a submodule — callers verify the
        symbol exists in the returned file)
      - only an *existing* repo-relative path (present in ``file_set``) is returned;
        the first candidate that exists wins, else ``None`` (module is stdlib/third
        party or ambiguous)
    """
    if not dotted:
        return None
    rel = dotted
    if rel.startswith("paf."):
        rel = rel[len("paf."):]
    elif rel == "paf":
        rel = ""
    # relative imports: leading dots offset against the importer's directory
    leading_dots = len(rel) - len(rel.lstrip("."))
    if leading_dots:
        base = PurePosixPath(importer_path).parent
        for _ in range(leading_dots - 1):  # `.` = same dir, `..` = one up, ...
            base = base.parent
        rel = rel.lstrip(".")
        if rel:
            base = base / PurePosixPath(*rel.split("."))
        rel = base.as_posix()
    elif rel:
        rel = "/".join(rel.split("."))  # a.b.c -> a/b/c
    else:
        return None
    if not rel:
        return None
    candidates = [f"{rel}.py", f"{rel}/__init__.py", f"{rel.rsplit('/', 1)[0]}.py"] \
        if "/" in rel and rel.rsplit("/", 1)[0] else [f"{rel}.py", f"{rel}/__init__.py"]
    for cand in candidates:
        if cand in file_set:
            return cand
    return None


def call_edge_set(
    files: list[FileInfo],
    sources: dict[str, str],
) -> dict[tuple[str, str], set[str]]:
    """Emit ``(source, target) -> {called symbols}`` for *provable* call reachability.

    ``files``: all ``FileInfo`` (with ``path`` repo-relative and ``symbols`` set).
    ``sources``: map of repo-relative path -> raw source (required; call detection
    needs the parse tree, which the lean ``FileInfo`` does not carry).

    Only pairs where a call site resolved unambiguously to a defining symbol are
    returned; everything else is *not* a candidate for a call edge and stays
    ``reference`` in ``build_edges``.
    """
    defined_by: dict[str, str] = {}
    for f in files:
        for name in f.symbols:
            if name.startswith("__") and name.endswith("__"):
                continue
            defined_by.setdefault(name, f.path)
    # set of real file paths (for module_to_path existence validation)
    file_set = {f.path for f in files}

    result: dict[tuple[str, str], set[str]] = {}
    for f in files:
        src = sources.get(f.path)
        if not src or f.language not in _CALL_LANGS:
            continue
        try:
            ex = CallExtractor(language=f.language or "python", source=src)
        except Exception:  # noqa: BLE001 — a bad file shouldn't kill the whole graph
            continue
        direct, qualified = ex.call_callees()
        for callee in direct + qualified:
            target = ex.resolve_callee(callee, defined_by, module_to_path, f.path, file_set)
            if target is None or target not in file_set or target == f.path:
                continue
            # symbol is the invoked name (or the callee root's attr / bare name)
            symbol = callee.split(".")[-1] if "." in callee else callee
            if symbol not in defined_by or defined_by[symbol] != target:
                continue
            result.setdefault((f.path, target), set()).add(symbol)
    return result
