"""Inverted, symbol-scoped token index — the search backend (spec §3.4).

Builds ``tokens.json`` from the extraction and ranks queries against it. Under the
LOCKED read-model, ``search_codebase`` scores HERE (field-tagged, symbol-scoped
postings), presents the ``index`` HEAD row for surfaced files, and dereferences
``manifest`` on pick (progressive disclosure). Body text is deliberately **not**
tokenized — name + signature + doc only, per Landon 2026-08-23 (grep-on-manifest /
``read_block`` covers in-body searches). The ``field`` tag lets the ranker weight
where a match landed: name > signature > doc.

Invariants:
- **Symbol-scoped**: every posting collapses to a symbol's ``manifest_ref`` block,
  not a bare line number — so a hit lands on a symbol, not a file.
- **Field-tagged**: ``field`` ∈ {name, signature, doc}.
- **Read-only**: ``tokens_rank`` never writes to storage (flat JSON stays the
  single source of truth; tokenization is a derivation).
"""

from __future__ import annotations

import math
from typing import Any

from ._schema import BlockRef, FileInfo, SearchHit, Span

# Weight a query token by how load-bearing the field it hits in is (§3.7.2):
# a plain name match beats a docstring mention.
_FIELD_WEIGHTS: dict[str, float] = {
    "name": 6.0,
    "signature": 3.0,
    "doc": 2.0,
}

# High-frequency English function words that carry no search signal (Landon
# 2026-08-23: tokens were "too verbose" — postings counted 25% stopwords).
# Filtered at tokenize time in ``_tok`` so index BUILD and QUERY see the same
# vocabulary (a stopword-only query must not match the same stopword in the
# index). Curated conservatively: only unambiguous envelope words, never
# meaningful identifiers (e.g. ``set``, ``get``, ``run``, ``call`` stay).
_STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be but by for from in is it of on or that the there this to was will with then than when where while your their its so such etc not if else elif i we you our they them me my his her no yes do does did has have had been being can could should would may might must into over under about up down out off again further once here all any both each few more most other some only own same too very just also however therefore because whereas whether although yet since during before after unless until despite still almost between among within without through above below around behind beside beyond inside outside near opposite across along""".split()
)

# Drop single/double-character tokens (``a``, ``to``, ``in``, ``is``, ``h``,
# ``_``) — never query-worthy, pure index bloat.
_MIN_TOKEN_LEN = 2


def _tok(text: str) -> list[str]:
    from paf.capabilities.search._ranking import tokenize

    return [
        t for t in tokenize(text) if len(t) > _MIN_TOKEN_LEN and t not in _STOPWORDS
    ]


def build_tokens(files: list[FileInfo]) -> dict[str, Any]:
    """Build the inverted ``tokens.json`` payload from rich ``FileInfo`` records.

    Returns ``{"version": 1, "files": {path: {token: [posting, ...]}}}`` where each
    posting is ``{symbol, kind, field, manifest_ref: {start_line, end_line}}``.
    Only fields {name, signature, doc} are tokenized (body is excluded, per the
    locked read-model). Unparsed files contribute nothing (no symbols to search).
    """
    out: dict[str, Any] = {"version": 1, "files": {}}
    for fi in files:
        if fi.unparsed:
            continue
        pmap: dict[str, list[dict[str, Any]]] = {}
        for s in fi.searchable_symbols:
            if not s.name:
                continue
            fields: list[tuple[str, str]] = [("name", s.name)]
            if s.signature:
                fields.append(("signature", s.signature))
            if s.doc:
                fields.append(("doc", s.doc))
            ref = None
            if s.span is not None and (
                s.span.start_line is not None or s.span.end_line is not None
            ):
                ref = {
                    "start_line": s.span.start_line,
                    "end_line": s.span.end_line,
                }
            for field, text in fields:
                for tok in _tok(text):
                    post: dict[str, Any] = {
                        "symbol": s.name,
                        "kind": s.kind,
                        "field": field,
                    }
                    if ref is not None:
                        post["manifest_ref"] = dict(ref)
                    pmap.setdefault(tok, []).append(post)
        if pmap:
            out["files"][fi.path] = pmap
    return out


def tokens_rank(
    query: str,
    tokens: dict[str, Any],
    *,
    limit: int = 20,
    heads: dict[str, dict[str, Any]] | None = None,
) -> list[SearchHit]:
    """Field-weighted relevance rank over the inverted ``tokens.json`` payload.

    Scores files by summed field-weighted idf over matched symbol postings, then
    surfaces the top ``limit`` as ``SearchHit``s. Each hit's ``block`` collapses to
    the best-matched symbol's ``manifest_ref`` (a ``BlockRef`` into ``manifest``).
    ``snippet`` carries the file's ``index`` HEAD row doc head when ``heads`` (a
    path → index head row map) is supplied — the ``present the head`` step of the
    locked read-model — so the caller can dereference ``manifest`` on pick.
    """
    qtokens = _tok(query)
    if not qtokens:
        return []
    files_map = tokens.get("files") or {}
    paths = [p for p in files_map if files_map[p]]
    if not paths:
        return []
    n = len(paths)
    qtoks = list(dict.fromkeys(qtokens))

    # idf per query term over the file corpus (df = files containing the term).
    idf: dict[str, float] = {}
    for q in qtoks:
        df = sum(1 for p in paths if q in files_map[p])
        idf[q] = math.log((n - df + 0.5) / (df + 0.5) + 1.0) if df else 0.0

    scored: dict[str, float] = {}
    best: dict[str, tuple[str | None, Span | None]] = {}  # path -> (symbol, span)

    for p in paths:
        pmap = files_map[p]
        s = 0.0
        top_sym: str | None = None
        top_span: Span | None = None
        top_w = 0.0
        for q in qtoks:
            postings = pmap.get(q)
            if not postings:
                continue
            w = idf[q]
            for post in postings:
                fw = _FIELD_WEIGHTS.get(post.get("field", ""), 1.0)
                contrib = w * fw
                s += contrib
                if contrib > top_w:
                    top_w = contrib
                    top_sym = post.get("symbol")
                    ref = post.get("manifest_ref")
                    if ref:
                        top_span = Span(
                            start_byte=None,
                            end_byte=None,
                            start_line=ref.get("start_line"),
                            end_line=ref.get("end_line"),
                        )
        if s > 0:
            scored[p] = s
            best[p] = (top_sym, top_span)

    ranked = sorted(scored, key=lambda p: scored[p], reverse=True)[:limit]

    hits: list[SearchHit] = []
    for p in ranked:
        sym, span = best.get(p, (None, None))
        head_span = span or Span(start_line=1, end_line=1)
        head = (heads or {}).get(p, {})
        hits.append(
            SearchHit(
                file=p,
                start_line=head_span.start_line or 1,
                end_line=head_span.end_line or 1,
                symbol=sym,
                snippet=(head.get("doc") or "")[:200],
                score=scored[p],
                block=BlockRef(file=p, span=head_span),
            )
        )
    return hits
