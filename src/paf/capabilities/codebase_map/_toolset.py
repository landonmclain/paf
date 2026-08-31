"""CodebaseMapToolset — the navigation-first 5-tool ladder (LOCKED read-model).

The model climbs a ladder instead of brute-forcing keyword scans:

    get_overview() → lookup_file(path, symbol, detail) → read_block(block_ref)
                                    ↕                        ↑
                            search_codebase(query)  (escape hatch for known words)

- ``get_overview()``            — orient to the top; the ``overview.json`` tree
- ``lookup_file(...)``          — the bridge; 4 shapes head ↔ detail
- ``read_block(block_ref)``     — the whole-file/region rung; the ONLY path to raw source
- ``search_codebase(query)``    — the thin escape hatch over ``tokens.json``; NO mode menu
- ``run_codebase_map()``        — (re)build the index artifacts for the rooted repo

``get_repo_map`` (ranked zero-LLM orient map) is a registered tool — see below.
``callers``/``with_graph`` defer to the graph phase (YAGNI). A
future ``semantic`` backend is an alternative ``RetrievalBackend``, never a ``mode``.
All hits collapse to the universal ``SearchHit``/``BlockRef`` shape.

Tools are **agent-facing** — every method returns a bounded, readable ``str`` (matching
the git/fs/shell toolset convention), never raw dicts. The heavy source text is only
materialized on ``read_block`` (query-the-pointer); ``lookup_file`` and ``search``
present bounded heads/blocks.
"""

from __future__ import annotations

import posixpath
from dataclasses import asdict
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


def _parent_dir(path: str) -> str:
    """Parent dir (``""`` at the root) of a repo-relative POSIX path.

    E.g. ``capabilities/memory/_organize.py`` -> ``capabilities/memory``;
    ``config.py`` -> ``""`` (a top-level entry, child of the root)."""
    return posixpath.dirname(path)


def _dir_importance(
    tree: list[dict], imp: dict[str, float]
) -> dict[str, tuple[float, float]]:
    """Map each directory path to (mean, max) importance of its DESCENDANT files.

    A directory's importance is the mean and max of `imp` taken over every file
    rooted under it (through nested subdirs), e.g. `capabilities/` covers all 62
    files, including `capabilities/codebase_map/*`. This is the module-level
    complement to per-file `imp`: it says how importance-DENSE a folder is (mean,
    corrected for file count -- so a small folder with a couple of hot files ranks
    fairly) and which single file dominates it (max).

    Only directories that appear as `type == "dir"` entries get a key; a dir with
    no scored files is omitted so the renderer shows no annotation for it.
    """
    dirs = {e["path"] for e in tree if e.get("type") == "dir"}
    acc: dict[str, list[float]] = {}
    for path, val in imp.items():
        if "/" not in path:
            continue
        parts = path.split("/")
        for i in range(1, len(parts)):
            d = "/".join(parts[:i])
            if d in dirs:
                acc.setdefault(d, []).append(val)
    out: dict[str, tuple[float, float]] = {}
    for d, vals in acc.items():
        if vals:
            hi = max(vals)
            out[d] = (sum(vals) / len(vals), hi)
    return out


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

    def get_overview(self, max_depth: int | None = None) -> str:
        """Orient to the top of the codebase: return the navigation tree.

        Use this FIRST, before any other tool, to see how the codebase is laid out
        (folders → files with counts). This is the starting point / orientation —
        a tree you climb down, not a search. Once you know where you are, drill with
        ``lookup_file``.

        Returns a **nested tree** (children indented under their parent directory),
        repo-relative: dirs show ``file_count``, files show ``line_count``.
        ``max_depth`` caps how deep the tree opens (default: open fully; pass e.g.
        ``max_depth=2`` on a big repo to see just the top directories + their files,
        then drill deeper with ``lookup_file``).

        When a call graph has been persisted at scan time, each file is annotated
        with a dependency-importance score (``imp=N``) computed from the RESOLVED
        graph edges — NOT raw occurrence counts. ``imp`` = a file's cross-repo
        coupling: every incoming dependence edge and every outgoing edge adds
        weight (a proven ``kind: \"call\"`` edge = 3.0, a reference/type edge = 1.0).
        This captures both kinds of importance: **hubs** (many things depend on the
        file — high in-degree, e.g. ``config.py``) and **orchestrators** (the file
        reaches into much of the codebase — high out-degree, e.g. ``runtime.py``).
        The count is derived from graph edges carrying RESOLVED symbol names, so it
        is immune to the word-collision inflation that a raw name-occurrence count
        suffers (the latter is NOT used here).
        """
        overview = load_overview(self._storage, self._scan_root)
        if overview is None:
            # Not indexed yet — derive the tree on demand from the same walker.
            overview = build_overview(self._scan_root)
        tree = overview.get("tree", [])
        imp = self._importance_by_path()
        dir_imp = _dir_importance(tree, imp)
        head = (
            f"Overview of {overview.get('scan_root', self._scan_root)}\n"
            f"{sum(1 for e in tree if e.get('type') == 'dir')} dirs, "
            f"{sum(1 for e in tree if e.get('type') == 'file')} files\n"
        )
        head += "file `imp=N` = dependency importance (in+out, call=3, ref=1); "
        head += "dir line: `imp: mean=N max=M` over its files\n"
        rendered = self._render_tree(tree, max_depth=max_depth, imp=imp, dir_imp=dir_imp)
        return head + rendered if rendered else head + "(empty)"

    def _importance_by_path(self) -> dict[str, float]:
        """Map repo-relative path → dependency-importance score from the call graph.

        ``imp(file) = sum over every graph edge touching it of (3.0 if the edge is a
        proven call else 1.0)`` — over BOTH in-edges (things that depend on it) and
        out-edges (things it reaches into). Resolved graph edges only, so this is
        immune to the generic-word-collision inflation that a name-occurrence count
        suffers. Returns ``{}`` when no call_graph.json has been persisted.
        """
        out: dict[str, float] = {}
        payload = self._storage.read(ArtifactKind.GRAPH, self._scan_root)
        edges = payload.get("edges", []) if isinstance(payload, dict) else []
        for e in edges:
            if not isinstance(e, dict):
                continue
            w = 3.0 if e.get("kind") == "call" else 1.0
            for side in ("source", "target"):
                p = e.get(side)
                if isinstance(p, str) and p:
                    out[p] = out.get(p, 0.0) + w
        return out

    @staticmethod
    def _render_tree(
        tree: list[dict],
        max_depth: int | None = None,
        imp: dict[str, float] | None = None,
        dir_imp: dict[str, tuple[float, float]] | None = None,
    ) -> str:
        """Render the flat ``(depth, path)``-sorted entry list as a real nested tree.

        Children are emitted indented under their parent directory (dirs before
        files, alphabetical within each), instead of the prior flat "all dirs then
        all files" dump with fully-requalified paths repeated on every line.

        Optional ``imp`` ({path: dependency-importance}) annotates each file line
        with ``(imp=N)``; optional ``dir_imp`` ({dir: (mean, max)}) annotates each
        dir line with ``imp: mean=M max=N``. Degrades to the plain tree when empty.
        """
        if not tree:
            return ""
        imp = imp or {}
        dir_imp = dir_imp or {}
        dirs: dict[str, list[dict]] = {}  # dir path -> its child entries (files + subdirs)
        for e in tree:
            path = e.get("path", "")
            parent = _parent_dir(path)
            dirs.setdefault(parent, []).append(e)
        lines: list[str] = []

        def _open_dir(dir_path: str, prefix: str) -> None:
            children = sorted(
                dirs.get(dir_path, []),
                key=lambda e: (0 if e.get("type") == "dir" else 1, e.get("path", "")),
            )
            for e in children:
                path = e.get("path", "")
                depth = e.get("depth", 0)
                if max_depth is not None and depth > max_depth:
                    # Drill only into dirs above the cap; files deeper than max_depth
                    # are folded away (a later `get_overview`/`lookup_file` reveals them).
                    continue
                name = path.rsplit("/", 1)[-1] if "/" in path else path
                if e.get("type") == "dir":
                    di = dir_imp.get(path)
                    d_suffix = (
                        f"  imp: mean={di[0]:.1f} max={di[1]:.0f}" if di else ""
                    )
                    lines.append(f"{prefix}{name}/  ({e.get('file_count', 0)} files){d_suffix}")
                    _open_dir(path, prefix + "  ")
                else:
                    r = imp.get(path)
                    suffix = f"  imp={r:g}" if r is not None and r > 0 else ""
                    lines.append(f"{prefix}{name}  ({e.get('line_count', 0)} lines){suffix}")

        # The root's children are the top-level entries (depth 1); start there.
        _open_dir("", "")
        return "\n".join(lines)


    def lookup_file(self, path: str, symbol: str | None = None, detail: bool = False) -> str | None:
        """Look up a file's head or detail (the bridge between overview and source).

        Four forms, least → most detail:
        - ``lookup_file(path)``                    → the file's HEAD: metadata, module
          docstring, and its symbol NAME list (no bodies).
        - ``lookup_file(path, symbol=X)``          → just that symbol's head
          (name/kind/span) from the manifest.
        - ``lookup_file(path, detail=True)``       → the file's manifest block as
          METADATA: every symbol's kind/signature/docstring/span — **no bodies**, so
          it stays bounded no matter how big the file is. Use ``read_block`` or the
          symbol form for actual source/body text.
        - ``lookup_file(path, symbol=X, detail=True)`` → that symbol's full manifest
          entry **including its body** (docstring/signature/body/span) — the bounded
          "read one symbol" form.

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
        text = self._storage.read_block(ref)
        if text is None:
            return None
        # Present line numbers to the model so it can report "the line range".
        # The storage seam returns RAW source (a swappable backend shouldn't
        # format); numbering is a tool-layer presentation concern. Only annotate
        # when the caller asked for a line span (`path:L1-L2`), so whole-file or
        # byte-slice reads stay un-suffixed.
        if span.start_line is not None and span.end_line is not None:
            numbered = []
            for offset, line in enumerate(text.splitlines()):
                numbered.append(f"{span.start_line + offset}: {line}")
            # Preserve a single trailing newline so the reply renders cleanly.
            suffix = "\n" if text.endswith("\n") else ""
            text = "\n".join(numbered) + suffix
        return text

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
        """Orient to the whole codebase in one shot: ranked, zero-LLM distilled map.

        Use this when you need the "here's the whole repo" picture — the direct analog
        of a really good README — rather than climbing the tree one rung at a time.
        Higher-level than ``get_overview`` (which is a bare navigation tree): this ranks
        the most load-bearing files/symbols and gives each a one-line signature.

        Ranking is **call-weighted** (Phase 1.2): files that other code *provably
        invokes* (orchestrators, helpers, constructors) outrank files that are merely
        imported as types. Everything rendered is **real** data pulled from the manifest
        (signatures) and the persisted graph (in-degree) — never LLM prose — so it cannot
        hallucinate a summary.

        Bound by ``map_tokens`` (default 1000; ~one signature per token). Pass
        ``focus=capabilities/memory`` (a repo-relative path prefix) to orient to just
        that subtree.
        """
        budget = int(map_tokens) if map_tokens is not None else 1000
        if budget < 1:
            budget = 1
        entries = self._retrieval.distilled_map(
            map_tokens=budget,
            focus=(focus if focus and focus.strip() else None),
        )
        if entries is None:
            return (
                f"No call graph for {self._scan_root} — the index predates the graph "
                "artifact. Run `run_codebase_map` to re-index (it now persists "
                "call_graph.json), then call get_repo_map again."
            )
        if not entries:
            return (
                f"No repo map for {self._scan_root} — run `run_codebase_map` to build "
                "the manifest + graph first."
            )
        head = (
            f"Repo map of {self._scan_root} ({len(entries)} symbols, "
            f"{budget}-token budget)\n"
        )
        body_lines: list[str] = []
        cur = None
        for e in entries:
            if e.file != cur:
                body_lines.append(f"{e.file}:")
                cur = e.file
            # manifest signature already embeds the symbol name — render only its
            # first line (a multiline signature's body comment is noise in a
            # budget-bounded one-liner map; full detail is one lookup_file away).
            # Fall back to the bare name for symbols with no signature.
            shown = e.signature if e.signature else e.symbol
            if shown:
                shown = shown.splitlines()[0].rstrip()
            body_lines.append(f"    {shown}")
        rendered = head + "\n".join(body_lines)

        # Persist the map so it exists as a real artifact on disk (repo_map.json) —
        # the structured entries (reusable) + the rendered text + metadata. It is a
        # cheap cache of real artifact data, overwritten on each successful render.
        try:
            self._storage.write(
                ArtifactKind.MAP,
                self._scan_root,
                {
                    "scan_root": str(self._scan_root),
                    "map_tokens": budget,
                    "focus": focus if focus and focus.strip() else None,
                    "entries": [asdict(e) for e in entries],
                    "rendered": rendered,
                },
            )
        except Exception:  # pragma: no cover - cache write must never break the tool
            pass  # the map still returns; persistence is best-effort
        return rendered

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
            out.append(self._symbol_meta(s))
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
        """A single symbol's full manifest entry — INCLUDING its body.

        The bounded "read one symbol" form: only the requested symbol's source/body is
        returned, so it can't blow up context the way a whole-file dump can. Body text
        is the weight-1 search field from the manifest (see _symbol_block)."""
        if manifest is None:
            return None
        sym = self._find_symbol(manifest.get("symbols") or [], symbol)
        if sym is None:
            return None
        return self._symbol_block(sym)

    def _symbol_meta(self, s: dict) -> str:
        """A symbol's manifest METADATA (kind/signature/docstring/span) — **no body**.

        Used for the whole-file ``detail=True`` form, which must stay bounded no
        matter how large the file is. Body/source text is served by ``read_block`` or
        the single-symbol form (``_symbol_block``)."""
        name = s.get("name", "?")
        kind = s.get("kind", "?")
        span = self._span_str(s.get("span"))
        block = [f"## {name} ({kind}{span})"]
        if s.get("signature"):
            block.append(f"  {s['signature']}")
        if s.get("doc"):
            block.append(f"  {s['doc']}")
        return "\n".join(block)

    def _symbol_block(self, s: dict) -> str:
        """A symbol's full manifest entry including its body — the single-symbol form."""
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
