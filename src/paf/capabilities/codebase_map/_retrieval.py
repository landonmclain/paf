"""Retrieval seam — the search-method extension point.

This is the *how queries are answered* seam (vs `_store.py`'s *where artifacts are
stored* seam). ``RetrievalBackend`` is the designated drop point for a future
semantic/vector backend; ``LexicalRetrieval`` is the default implementation.

Under the LOCKED read-model, ``search`` scores over **``tokens.json``**
(the inverted, symbol-scoped search backend — name+sig+doc postings), presents the
**``index`` HEAD** row for each surfaced file (path, language, module docstring,
matched name lists), and dereferences ``manifest`` only on pick — progressive
disclosure. Body text is deliberately not tokenized (grep-on-manifest / ``read_block``
covers in-body). The lean head is the *presentation* layer, not the scoring source.

``LexicalRetrieval`` reads only; it NEVER writes to storage (flat JSON stays the single
source of truth).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ._schema import DistilledEntry, SearchHit
from ._store import ArtifactKind, FlatFileStorage, StorageService
from ._tokens import tokens_rank

# Weightings for the zero-LLM distilled-map ranking (call-weighted in-degree).
# A PROVEN call edge (kind="call") counts for CALL_WEIGHT; a mere
# reference/type-import edge counts for REFERENCE_WEIGHT. Chose 3:1 so files the
# agent actually invokes outrank files it merely annotates — without drowning the
# reference signal entirely (focus on the *hub* files, not the noise).
CALL_WEIGHT = 3.0
REFERENCE_WEIGHT = 1.0


def _split_dir(path: str) -> str:
    """Containing directory of a repo-relative POSIX path (``\"\"`` at the root)."""
    i = path.rfind("/")
    return path[:i] if i >= 0 else ""


def _weighted_in_degree(edges: list[dict]) -> dict[str, float]:
    """Cross-module, call-weighted in-degree per target file.

    Each edge ``{source, target, kind}`` adds ``CALL_WEIGHT`` when ``kind == "call"``
    else ``REFERENCE_WEIGHT`` to its target's score — **but only when source and target
    live in different directories**. Edges *within* the same module are orientation
    noise: a 16-file module that imports its own ``_schema``/``_store`` heavily would
    otherwise inflate those files' in-degree purely via self-reference, drowning the
    true repo-wide hubs (config, the turns anchor, shared rankers). Restricting to
    cross-module edges measures real hub-ness: "files other parts of the codebase
    depend on". Self-edges are already excluded upstream. Returns ``{target: score}``.
    """
    out: dict[str, float] = {}
    for e in edges:
        s = e.get("source")
        t = e.get("target")
        if not isinstance(t, str) or not t:
            continue
        if not isinstance(s, str):
            continue
        if s and _split_dir(s) == _split_dir(t):
            continue  # same-directory coupling is not cross-codebase hub-ness
        w = CALL_WEIGHT if e.get("kind") == "call" else REFERENCE_WEIGHT
        out[t] = out.get(t, 0.0) + w
    return out


class RetrievalBackend(Protocol):
    """Search-method seam. A future semantic backend implements these two methods."""

    def search(
        self,
        query: str,
        *,
        mode: str,
        scope: Any = None,
        limit: int = 20,
    ) -> list[SearchHit]: ...

    def distilled_map(
        self, *, map_tokens: int, focus: Any = None
    ) -> list[DistilledEntry] | None: ...


class LexicalRetrieval:
    """Default: field-weighted search over the ``tokens.json`` search backend.

    Scores tokens (name > signature > doc) → annotates each hit with its ``index``
    HEAD row (module-docstring head + matched name lists) → returns ``SearchHit``s
    whose ``block`` collapses to the best-matched symbol's ``manifest_ref``.
    """

    def __init__(
        self,
        storage: StorageService | None = None,
        scan_root: str | Path | None = None,
    ) -> None:
        self.storage = storage or FlatFileStorage()
        self._scan_root_value = Path(scan_root).resolve() if scan_root else None

    def search(
        self,
        query: str,
        *,
        mode: str = "keywords",
        scope: Any = None,
        limit: int = 20,
    ) -> list[SearchHit]:
        """Token-scored search over the search backend, presented as index heads.

        ``mode`` is accepted for API compatibility; broad token ranking is the
        first pass (exact/`callers`/graph modes are not yet built). Scores
        ``tokens.json`` (the search backend) and presents each
        surfaced file's lean ``index`` HEAD row (module docstring + matched names).
        """
        tokens = self._tokens_payload()
        return tokens_rank(query, tokens, limit=limit, heads=self._heads_by_path())

    def distilled_map(
        self, *, map_tokens: int, focus: Any = None
    ) -> list[DistilledEntry] | None:
        """Zero-LLM synthetic orientation map: the whole repo in `map_tokens` tokens.

        Ranks files by **call-weighted in-degree** from the persisted graph artifact:
        an incoming edge contributes ``CALL_WEIGHT`` (3.0) when it is a provable call
        (``kind: "call"``) and ``REFERENCE_WEIGHT`` (1.0) otherwise. Files the
        agent actually *invokes* (orchestrators, helpers, constructors) therefore outrank
        files that are merely *type-annotated / imported* — no tokens spent on
        embeddings.

        Within the budget it walks files in descending score and emits one
        ``DistilledEntry`` per searchable symbol, pulling the real ``signature`` from the
        manifest. ``focus`` (a repo-relative path prefix) biases candidates to that
        subtree so a caller can orient to just one area. Everything rendered is **real
        artifact data** — never LLM prose — so the map cannot hallucinate or freeze
        understanding at build time.
        """
        graph = self.storage.read(ArtifactKind.GRAPH, self._scan_root()) or {}
        edges = graph.get("edges", []) if isinstance(graph, dict) else []
        if graph is None or not isinstance(graph, dict) or "edges" not in graph:
            # No persisted call graph — the index predates the graph artifact (or was never
            # scanned). Ranking needs edges; refuse to emit a misleading
            # alphabetical "map" instead of the real in-degree order. Caller
            # (get_repo_map) surfaces this as "reindex needed".
            return None
        manifest = self.storage.read(ArtifactKind.MANIFEST, self._scan_root()) or {}
        files = manifest.get("files", {}) if isinstance(manifest, dict) else {}
        coming = _weighted_in_degree(edges)  # path -> weighted score

        def select(path: str) -> bool:
            if not focus:
                return True
            fp = focus.rstrip("/")
            return path == fp or path.startswith(fp + "/")

        scored = sorted(
            (p for p in files if select(p)),
            key=lambda p: (-coming.get(p, 0.0), p),
        )

        entries: list[DistilledEntry] = []
        budget = max(1, int(map_tokens))
        rank = 0
        for path in scored:
            rec = files.get(path) or {}
            sig = rec.get("signature") or ""
            for sym in rec.get("symbols", []) or []:
                if not isinstance(sym, dict):
                    continue
                name = sym.get("name")
                if not name:
                    continue
                rank += 1
                s = sym.get("signature") or sig
                entries.append(
                    DistilledEntry(
                        rank=rank,
                        file=path,
                        symbol=name,
                        signature=s,
                        references_count=rec.get("references_count", 0),
                        ref_weight=coming.get(path, 0.0),
                    )
                )
                budget -= 1
                if budget <= 0:
                    return entries
        return entries

    # ---- reads --------------------------------------------------------------

    def _tokens_payload(self) -> dict:
        payload = self.storage.read(ArtifactKind.TOKENS, self._scan_root())
        return payload if isinstance(payload, dict) else {}

    def _heads_by_path(self) -> dict[str, dict]:
        """Lean ``index`` HEAD rows keyed by path (the ``present the head`` layer)."""
        index = self.storage.read(ArtifactKind.INDEX, self._scan_root())
        rows = (index or {}).get("files", []) if isinstance(index, dict) else []
        heads: dict[str, dict] = {}
        for r in rows:
            if isinstance(r, dict) and isinstance(r.get("path"), str):
                heads[r["path"]] = r
        return heads

    def _scan_root(self) -> Path:
        """The scan root we search. Explicit scan_root wins,
        else derive from the storage override dir, else cwd (tests pass a storage
        with a fixed index_root, so ``list_scan_roots`` reports the override)."""
        if self._scan_root_value is not None:
            return self._scan_root_value
        roots = self.storage.list_scan_roots()
        if roots:
            return Path(roots[0])
        return Path.cwd()
