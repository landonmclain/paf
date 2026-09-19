"""Ranking: BM25 first-pass search over the manifest.

**BM25 first-pass broad search** — the quick start.

Design:
  - Shares the repo's dependency-free BM25 tokenizer and tuning constants
    (``paf.capabilities.search._ranking`` — reuses ``tokenize`` and ``Bm25Params``),
    but reimplements the idf + scoring loop here (``_idf``, the inline rank loop).
    The codebase-map field-weighting (per-record field tokens, reference tiebreak)
    differs enough from the search module's that shared scoring isn't worth the
    coupling. No retrieval framework (no Rank-BM25 / whoosh / LlamaIndex).
  - Field-weighted: name > qualified > file_path > signature >
    doc > imports > body. Each ``FileInfo`` record is scored as a bag of weighted
    field tokens so a hit in a high-weight field (name/file_path) beats one in a
    low-weight field (body).
  - Reference count is a *harmless weak* signal: added as a tiny
    fractional tiebreak so a widely-referenced symbol edges out an equal-scoring
    obscure one — but it does NOT dominate (broad search runs on text; the count
    "won't move the needle much, but as long as it doesn't hurt").
  - Returns the universal ``SearchHit`` shape (collapses to a ``BlockRef``
    pointer; heavy source deferred to ``read_block``).
"""

from __future__ import annotations

import math

from ._schema import BlockRef, FileInfo, SearchHit, Span

# Weight a query token by how load-bearing the field it hits in is:
# ranking text is more meaningful in an exact field than in prose body text.
_FIELD_WEIGHTS: dict[str, float] = {
    "name": 6.0,
    "qualified": 4.0,
    "file_path": 4.0,
    "signature": 3.0,
    "doc": 2.0,
    "imports": 2.0,
    "body": 1.0,
}

# Reference-count tiebreak: tiny, so it never dominates text relevance.
_REF_TIE_WEIGHT = 0.001


def _tok(text: str) -> list[str]:
    from paf.capabilities.search._ranking import tokenize

    return tokenize(text)


def _field_tokens(rec: FileInfo) -> list[tuple[str, list[str]]]:
    """``(field_name, tokens)`` pairs for each indexed field of one record."""
    pairs: list[tuple[str, list[str]]] = []
    pairs.append(("file_path", _tok(rec.path)))
    pairs.append(("doc", _tok(rec.doc or "")))
    if rec.imports:
        pairs.append(("imports", _tok(" ".join(rec.imports))))
    for name in rec.symbols:
        pairs.append(("name", _tok(name)))
    for sym in rec.searchable_symbols:
        if sym.name:
            pairs.append(("name", _tok(sym.name)))
        if sym.qualified:
            pairs.append(("qualified", _tok(sym.qualified)))
        if sym.signature:
            pairs.append(("signature", _tok(sym.signature)))
        if sym.doc:
            pairs.append(("doc", _tok(sym.doc)))
        if sym.body:
            pairs.append(("body", _tok(sym.body)))
    return pairs


def _weighted_doc_tokens(rec: FileInfo) -> dict[str, float]:
    """Flatten a record's fields into a term→weight map (field-weighted bag).

    A term's weight = sum over every field it appears in of that field's weight
    (so a term in both ``name`` and ``doc`` accumulates both). This is our
    field-weighting layer over the raw BM25 term frequency.
    """
    weighted: dict[str, float] = {}
    for field, tokens in _field_tokens(rec):
        w = _FIELD_WEIGHTS.get(field, 1.0)
        for token in set(tokens):  # dedupe within a field once
            weighted[token] = weighted.get(token, 0.0) + w
    return weighted


def _idf(n: int, df: int) -> float:
    """Standard BM25 idf; only a zero-document-frequency term is non-discriminative (0)."""
    if df == 0:
        return 0.0
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


def rank(
    query: str,
    files: list[FileInfo],
    *,
    limit: int = 20,
    ref_weight: bool = True,
) -> list[SearchHit]:
    """BM25 field-weighted rank of ``files`` against ``query`` → ``SearchHit`` list.

    Returns at most ``limit`` hits with ``score > 0``, best first. ``score`` uses
    BM25 over the field-weighted token bag, plus an optional tiny reference-count
    tiebreak. A hit's ``block`` points at the file (repo-relative) with a line-1
    span (whole-file) — the pointer for ``read_block``.
    """
    from paf.capabilities.search._ranking import Bm25Params

    query_tokens = _tok(query)
    if not query_tokens or not files:
        return []

    docs = [_weighted_doc_tokens(f) for f in files]
    avgdl = sum(sum(d.values()) for d in docs) / len(docs) or 1.0

    params = Bm25Params()
    unique_q = list(dict.fromkeys(query_tokens))

    # idf computed against the weighted token bags (document frequency = how many
    # records contain the term at all).
    doc_key_sets = [set(d) for d in docs]
    n = len(docs)
    idf_scores: dict[str, float] = {}
    for qt in unique_q:
        df = sum(1 for ks in doc_key_sets if qt in ks)
        idf_scores[qt] = _idf(n, df)

    scored: list[tuple[float, FileInfo]] = []
    for rec, doc in zip(files, docs, strict=True):
        dl = sum(doc.values()) or 1.0
        score = 0.0
        for qt in unique_q:
            tf = doc.get(qt, 0.0)
            if tf == 0:
                continue
            idf = idf_scores.get(qt, 0.0)
            if idf == 0:
                continue
            num = tf * (params.k1 + 1.0)
            den = tf + params.k1 * (1.0 - params.b + params.b * dl / avgdl)
            score += idf * num / den
        if ref_weight:
            score += rec.references_count * _REF_TIE_WEIGHT
        if score > 0:
            scored.append((score, rec))

    scored.sort(key=lambda item: item[0], reverse=True)
    scored = scored[:limit]

    hits: list[SearchHit] = []
    for score, rec in scored:
        span = Span(start_byte=None, end_byte=None, start_line=1, end_line=1)
        hits.append(
            SearchHit(
                file=rec.path,
                start_line=1,
                end_line=1,
                symbol=rec.symbols[0] if rec.symbols else None,
                snippet=(rec.doc or "")[:200],
                score=score,
                block=BlockRef(file=rec.path, span=span),
            )
        )
    return hits


def rank_fused(
    query: str,
    files: list[FileInfo],
    *,
    limit: int = 20,
    **_: object,
) -> list[SearchHit]:
    """Alias of ``rank`` — fused (field-weighted BM25 + tiny ref tiebreak)."""
    return rank(query, files, limit=limit)
