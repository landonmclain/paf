"""CodebaseMapToolset — the navigation-first 5-tool ladder (LOCKED read-model).

The model climbs a ladder instead of brute-forcing keyword scans:

    get_overview() → lookup_file(path, symbol, detail) → read_block(block_ref)
                                    ↕                        ↑
                            search_codebase(query)  (escape hatch for known words)

- ``get_overview()``            — orient to the top; the ``overview.json`` tree
- ``lookup_file(...)``          — the bridge; 4 shapes head ↔ detail
- ``read_block(block_ref)``     — the whole-file/region rung; the ONLY path to raw source
- ``search_codebase(query)``    — the thin escape hatch over ``tokens.json``; NO mode menu
- ``get_repo_map(...)``         — orient map (Phase 4; stub)
- ``run_codebase_map()``        — (re)build the index artifacts for the rooted repo

``callers``/``with_graph`` defer to the graph phase (YAGNI). A future ``semantic``
backend is an alternative ``RetrievalBackend``, never a ``mode``. All hits collapse to
the universal ``SearchHit``/``BlockRef`` shape.

Tools are **agent-facing** — every method returns a bounded, readable ``str`` (matching
the git/fs/shell toolset convention), never raw dicts. The heavy source text is only
materialized on ``read_block`` (query-the-pointer); ``lookup_file`` and ``search``
present bounded heads/blocks.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai.toolsets import FunctionToolset

from ._block import BlockRef
from ._overview import build_overview, load_overview
from ._retrieval import LexicalRetrieval
from ._schema import Span
from ._store import ArtifactKind, FlatFileStorage, StorageService

_MAX_SEARCH_HEAD = 200          # chars of module-doc head shown per search hit row
_MAX_SNIPPET_BYTES = 65536      # read_block hard bound (matches _block limit_bytes)
_MAX_LOOKUP_SYMBOLS = 120       # cap symbol name lists so a big head stays bounded


class CodebaseMapToolset(FunctionToolset):
    """Navigation-first five-tool codebase browser rooted at a scan root.

    All reads go through the storage seam (default ``FlatFileStorage``); path/symbol
    lookups are confined to the scan root. Tools are read-only — nothing here writes
    artifacts.
    """

    def __init__(
        self,
        *,
        scan_root: str | Path,
        storage: StorageService | None = None,
        retrieval: LexicalRetrieval | None = None,
    ) -> None:
        super().__init__()
        self._scan_root = Path(scan_root).resolve()
        self._storage = storage or FlatFileStorage()
        self._retrieval = retrieval or LexicalRetrieval(
            storage=self._storage, scan_root=self._scan_root
        )

        self.add_function(self.get_overview, name="get_overview")
        self.add_function(self.lookup_file, name="lookup_file")
        self.add_function(self.read_block, name="read_block")
        self.add_function(self.search_codebase, name="search_codebase")
        self.add_function(self.get_repo_map, name="get_repo_map")
        self.add_function(self.run_codebase_map, name="run_codebase_map")

    # -- presentation helpers -------------------------------------------------

    def _head_rows_by_path(self) -> dict[str, dict]:
        payload = self._storage.read(ArtifactKind.INDEX, self._scan_root)
        rows = (payload or {}).get("files", []) if isinstance(payload, dict) else []
        return {r["path"]: r for r in rows if isinstance(r, dict) and isinstance(r.get("path"), str)}

    def _manifest_paths(self) -> dict[str, dict]:
        payload = self._storage.read(ArtifactKind.MANIFEST, self._scan_root)
        rows = (payload or {}).get("files", {}) if isinstance(payload, dict) else {}
        return rows if isinstance(rows, dict) else {}

    # -- tools ----------------------------------------------------------------

    def get_overview(self) -> str:
        """Orient to the top of the codebase: return the navigation tree.

        Use this FIRST, before any other tool, to see how the codebase is laid out
        (folders → files with counts). This is the starting point / orientation —
        a tree you climb down, not a search. Once you know where you are, drill with
        ``lookup_file``.

        Returns a tree of repo-relative ``path`` entries (dirs with ``file_count``,
        files with ``line_count``), shallowest depth first, plus a top summary.
        """
        overview = load_overview(self._storage, self._scan_root)
        if overview is None:
            # Not indexed yet — derive the tree on demand from the same walker.
            overview = build_overview(self._scan_root)
        tree = overview.get("tree", [])
        dirs = [e for e in tree if e.get("type") == "dir"]
        files = [e for e in tree if e.get("type") == "file"]
        lines = [
            (
                f"{'  ' * (e.get('depth', 1) - 1)}{e['path']}/"
                f"  ({e.get('file_count', 0)} files)"
            )
            for e in dirs
        ] + [
            (
                f"{'  ' * (e.get('depth', 1) - 1)}{e['path']}"
                f"  ({e.get('line_count', 0)} lines)"
            )
            for e in files
        ]
        head = (
            f"Overview of {overview.get('scan_root', self._scan_root)}\n"
            f"{len(dirs)} dirs, {len(files)} files\n"
        )
        return head + "\n".join(lines)


    def lookup_file(self, path: str, symbol: str | None = None, detail: bool = False) -> str | None:
        """Look up a file's head or detail (the bridge between overview and source).

        Four forms, least → most detail:
        - ``lookup_file(path)``                    → the file's HEAD: metadata, module
          docstring, and its symbol NAME list (no bodies).
        - ``lookup_file(path, symbol=X)``          → just that symbol's head
          (name/kind/span) from the manifest.
        - ``lookup_file(path, detail=True)``       → the file's full manifest block:
          every symbol's docstring/signature/span.
        - ``lookup_file(path, symbol=X, detail=True)`` → that symbol's full manifest
          entry (docstring/signature/body/span).

        ``path`` is repo-relative. Returns ``None`` for an unknown path/symbol.
        This is a *jump* to a known location — prefer ``get_overview`` to discover
        where things are. For the actual source, follow up with ``read_block``.
        """
        heads = self._head_rows_by_path()
        head = heads.get(path)
        if head is None:
            return None

        manifests = self._manifest_paths()
        manifest = manifests.get(path)

        if detail:
            if symbol is not None:
                return self._format_symbol_manifest(manifest, symbol, path)
            return self._format_file_detail(manifest, head, path)
        if symbol is not None:
            return self._format_symbol_head(manifest, symbol, path)
        return self._format_head(head, path)

    def read_block(self, block_ref: str) -> str | None:
        """Read raw source for a file or region by a ``block_ref`` pointer.

        ``block_ref`` is the compact pointer form ``path`` or ``path:L1-L2``
        (e.g. ``src/paf/config.py`` or ``src/paf/config.py:40-80``), resolved
        relative to the scan root. This is the ONLY tool that returns actual source
        text — use it after ``lookup_file``/``search_codebase`` point you at a file
        or symbol. Bounded: never returns more than 64 KB.

        Returns ``None`` if the file does not exist.
        """
        file_part, _, line_part = block_ref.partition(":")
        span: Span = Span()
        if line_part:
            try:
                a, _, b = line_part.partition("-")
                span = Span(
                    start_byte=None, end_byte=None,
                    start_line=int(a), end_line=int(b) if b else int(a),
                )
            except ValueError:
                span = Span()  # unparseable range -> whole-file read
        # resolve the repo-relative path against the scan root (with an escape guard),
        # then cross the storage seam: the backend reads the exact source bytes.
        full = (self._scan_root / file_part).resolve()
        try:
            full.relative_to(self._scan_root)
        except ValueError:
            return None
        ref = BlockRef(file=str(full), span=span)
        return self._storage.read_block(ref)

    def search_codebase(self, query: str, limit: int = 20) -> str:
        """Search the codebase for ``query`` and return the top hits.

        The thin escape hatch — use when you know a word/name but not where it lives.
        Scores over the symbol-scoped search index (name > signature > doc) and returns
        ranked hits. **Each hit includes a directly-usable ``read_block`` ref**
        (``read: path:L1-L2``) — paste it straight into ``read_block`` to jump to the
        exact matched symbol's source; no need to re-type the path. Also shows the
        matched ``symbol``, ``manifest_ref`` line range, and module-doc head snippet.
        There is no mode menu — one fused search call.

        Args:
            query: The words to search for (identifiers, docstring terms, etc.).
            limit: Max hits to return (default 20).
        """
        hits = self._retrieval.search(query, limit=max(1, limit))
        if not hits:
            return f"No results for {query!r}."
        head = self._head_rows_by_path()
        lines = [f"Top {len(hits)} hits for {query!r}:"]
        for h in hits:
            doc = (head.get(h.file, {}).get("doc") or "")[: _MAX_SEARCH_HEAD]
            sym = f"#{h.symbol}" if h.symbol else ""
            # The whole point of the index: the hit collapses to a pointer. Hand it
            # over as a paste-able read_block ref (prefer the resolved BlockRef span,
            # fall back to the hit's own line fields).
            ref = self._read_block_ref(h)
            lines.append(
                f"- {h.file}{sym}  lines {h.start_line}-{h.end_line} "
                f"(score {h.score:.2f})"
            )
            if ref:
                lines.append(f"    read: {ref}")
            if doc:
                lines.append(f"    {doc}")
        return "\n".join(lines)

    @staticmethod
    def _read_block_ref(h: object) -> str | None:
        """Compact ``path:L1-L2`` ref for a hit's ``BlockRef`` (the pointer)."""
        file = getattr(h, "file", None)
        if not file:
            return None
        block = getattr(h, "block", None)
        span = getattr(block, "span", None) if block is not None else None
        if span is not None and getattr(span, "start_line", None) and getattr(span, "end_line", None):
            return f"{file}:{span.start_line}-{span.end_line}"
        sl, el = getattr(h, "start_line", None), getattr(h, "end_line", None)
        if sl and el:
            return f"{file}:{sl}-{el}"
        return f"{file}"

    def get_repo_map(self, map_tokens: int | None = None, focus: str | None = None) -> str:
        """[NOT BUILT — Phase 4] Return a distilled orientation map of the codebase.

        **Intended shape (LOCKED 2026-08-24): a synthetic, zero-LLM orientation map.**
        Bound by ``map_tokens`` (default 1000), optionally ``focus``-biased. Returns the
        most load-bearing files/symbols as compact ``file.py: symbol(args) -> T`` lines,
        ordered by ``references_count`` (in-degree) desc, with each file's module-docstring
        head. Deliberately NO LLM summarization — everything shown is **real** data pulled
        from existing artifacts (`overview.json` tree, `index.json` head + symbol name lists,
        ``manifest`` signatures, ``references_count``), so it cannot hallucinate a summary or
        freeze understanding at build time. It is the "here's the whole repo in N tokens"
        view — the direct analog of a really good README. Computed on demand from artifacts
        (optionally cached in ``distilled_map.json``). This is a mild superset of
        ``get_overview`` (tree + rank + signature one-liners); the schema `DistilledEntry`
        already matches. Not implemented yet.
        """
        raise NotImplementedError("get_repo_map is not built until Phase 4")

    def run_codebase_map(self) -> str:
        """Re-scan the codebase root and (re)build the index artifacts.

        Use this when the index is missing, stale, or you just edited many files and
        want the map refreshed. Re-extracts every in-scope file and rewrites
        ``overview.json`` / ``index.json`` / ``manifest.json`` / ``tokens.json`` /
        ``snapshot.json`` for the rooted repo (``scan_root``). Returns a bounded
        summary (files found/extracted, symbol count, unparsed) — not the index
        contents (use ``get_overview``/``lookup_file``/``search_codebase`` to read).
        """
        from ._scan import scan_and_write

        res = scan_and_write(self._scan_root, self._storage)
        return (
            f"Re-indexed {res['scan_root']}\n"
            f"{res['extracted']} files extracted, {res['total_symbols']} symbols\n"
            f"{len(res['unparsed'])} unparsed, {len(res['skipped'])} skipped\n"
            f"index at {res['index_dir']}"
        )

    # -- format helpers -------------------------------------------------------

    def _format_head(self, head: dict, path: str) -> str:
        doc = (head.get("doc") or "").splitlines()[0] if head.get("doc") else ""
        names = self._name_lists(head)
        out = [f"# {path}", f"language: {head.get('language') or '?'}"]
        if doc:
            out.append(f"doc: {doc}")
        if head.get("imports"):
            out.append(f"imports: {', '.join(head['imports'][:_MAX_LOOKUP_SYMBOLS])}")
        if names["classes"]:
            out.append(f"classes: {', '.join(names['classes'])}")
        if names["functions"]:
            out.append(f"functions: {', '.join(names['functions'])}")
        if names["other"]:
            out.append(f"symbols: {', '.join(names['other'])}")
        return "\n".join(out)

    def _format_file_detail(self, manifest: dict | None, head: dict, path: str) -> str:
        out = [self._format_head(head, path), "", "## Detail"]
        if manifest is None:
            out.append("(no manifest detail available)")
            return "\n".join(out)
        syms = manifest.get("symbols") or []
        if not syms:
            out.append("(no symbols)")
        for s in syms[: _MAX_LOOKUP_SYMBOLS]:
            out.append(self._symbol_block(s))
        if len(syms) > _MAX_LOOKUP_SYMBOLS:
            out.append(f"... {len(syms) - _MAX_LOOKUP_SYMBOLS} more symbols")
        return "\n".join(out)

    def _format_symbol_head(self, manifest: dict | None, symbol: str, path: str) -> str | None:
        if manifest is None:
            return None
        sym = self._find_symbol(manifest.get("symbols") or [], symbol)
        if sym is None:
            return None
        span = self._span_str(sym.get("span"))
        return f"{path}#{symbol}  ({sym.get('kind', '?')}{span})"

    def _format_symbol_manifest(self, manifest: dict | None, symbol: str, path: str) -> str | None:
        if manifest is None:
            return None
        sym = self._find_symbol(manifest.get("symbols") or [], symbol)
        if sym is None:
            return None
        out = [f"# {path}#{symbol}"]
        if sym.get("kind"):
            out.append(f"kind: {sym['kind']}")
        span = self._span_str(sym.get("span"))
        if span:
            out.append(f"span: {span}")
        if sym.get("signature"):
            out.append(f"signature: {sym['signature']}")
        if sym.get("doc"):
            out.append(f"docstring: {sym['doc']}")
        return "\n".join(out)

    def _symbol_block(self, s: dict) -> str:
        name = s.get("name", "?")
        kind = s.get("kind", "?")
        span = self._span_str(s.get("span"))
        body = s.get("body")
        block = [f"## {name} ({kind}{span})"]
        if s.get("signature"):
            block.append(f"  {s['signature']}")
        if s.get("doc"):
            block.append(f"  {s['doc']}")
        if body:
            block.append(body)
        return "\n".join(block)

    @staticmethod
    def _find_symbol(symbols: list[dict], name: str) -> dict | None:
        for s in symbols:
            if s.get("name") == name:
                return s
        return None

    @staticmethod
    def _name_lists(head: dict) -> dict[str, list[str]]:
        return {
            "classes": list(head.get("classes") or [])[: _MAX_LOOKUP_SYMBOLS],
            "functions": list(head.get("functions") or [])[: _MAX_LOOKUP_SYMBOLS],
            "other": list(head.get("symbols") or [])[: _MAX_LOOKUP_SYMBOLS],
        }

    @staticmethod
    def _span_str(span: object | None) -> str:
        if not isinstance(span, dict):
            return ""
        sl, el = span.get("start_line"), span.get("end_line")
        if sl is not None and el is not None:
            return f" [{sl}-{el}]"
        return ""
