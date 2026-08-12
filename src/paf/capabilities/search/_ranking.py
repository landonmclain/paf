"""Dependency-free BM25 ranking core for the paf record-owning full-context search.

This is the ranking heart of ``search_conversation_history``. It is a **paf-local
copy** of the core from `pydantic_ai_harness.conversation_search._toolset`
(MIT licensed), deliberately re-homed here so the whole recall surface -- index
search, read_turn, and wide conversation search -- is owned by paf with a single
marker-strip convention, rather than partially borrowed from the harness.

Only the pure ranking math lives here (`_tokenize`, `_compute_idf`, `_bm25_score`,
`_bm25_rank`). The corpus construction (what text gets searched) lives in the
toolset, so paf's injected-artifact strip applies uniformly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime

_TOKENIZE_RE = re.compile(r"\w+")
"""Tokenizer regex: word runs (unicode letters, digits, underscore)."""


@dataclass(frozen=True)
class Bm25Params:
    """BM25 tuning. Mirrors the harness defaults (Lucene/Elasticsearch-inspired)."""

    k1: float = 1.5
    """Term-frequency saturation, non-negative. Lucene's `BM25Similarity` uses 1.2."""

    b: float = 0.75
    """Length normalization, between 0.0 and 1.0 (Lucene/Elasticsearch default)."""


def tokenize(text: str) -> list[str]:
    """Split `text` into lowercase word tokens."""
    return [match.group().lower() for match in _TOKENIZE_RE.finditer(text)]


def compute_idf(term: str, doc_tokens: list[list[str]]) -> float:
    """Inverse document frequency, standard BM25 form `ln((N - df + 0.5)/(df + 0.5) + 1)`."""
    n = len(doc_tokens)
    df = sum(1 for tokens in doc_tokens if term in set(tokens))
    if df == 0:
        return 0.0
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


def bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf_scores: dict[str, float],
    avgdl: float,
    params: Bm25Params,
) -> float:
    """BM25 score of one document against a set of query tokens."""
    dl = len(doc_tokens)
    if dl == 0 or avgdl == 0:
        return 0.0

    tf_map: dict[str, int] = {}
    for token in doc_tokens:
        tf_map[token] = tf_map.get(token, 0) + 1

    score = 0.0
    for qt in query_tokens:
        tf = tf_map.get(qt, 0)
        if tf == 0:
            continue
        idf = idf_scores.get(qt, 0.0)
        numerator = tf * (params.k1 + 1.0)
        denominator = tf + params.k1 * (1.0 - params.b + params.b * dl / avgdl)
        score += idf * numerator / denominator

    return score


def bm25_rank(
    query: str,
    documents: list[str],
    params: Bm25Params | None = None,
) -> list[tuple[int, float]]:
    """Rank `documents` by BM25 relevance to `query`, descending; only positive scores.

    Returns a list of `(index, score)` pairs. Indices are stable against the input
    `documents` list, so callers can align results back to their corpus.
    """
    if params is None:
        params = Bm25Params()
    query_tokens = tokenize(query)
    if not query_tokens or not documents:
        return []

    doc_tokens = [tokenize(doc) for doc in documents]
    avgdl = sum(len(tokens) for tokens in doc_tokens) / len(doc_tokens)

    unique_query_tokens = list(dict.fromkeys(query_tokens))
    idf_scores = {qt: compute_idf(qt, doc_tokens) for qt in unique_query_tokens}

    results: list[tuple[int, float]] = []
    for index, tokens in enumerate(doc_tokens):
        score = bm25_score(unique_query_tokens, tokens, idf_scores, avgdl, params)
        if score > 0:
            results.append((index, score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results


def recency_boost(ts: str, anchor_ts: str, half_life_days: float) -> float:
    """Freshness multiplier in [0, 1] for one entry, relative to a newer anchor.

    Returns 1.0 for the newest entry (age 0), decaying exponentially with a
    configurable half-life toward 0 as `ts` falls behind `anchor_ts`. This is a
    pure rank-agnostic freshness signal; callers fold it into a relevance score.
    Unparseable timestamps degrade gracefully to 0.0 (treated as stale), and an
    anchor older than the entry is clamped to 1.0.
    """
    try:
        t = datetime.fromisoformat(ts)
        anchor = datetime.fromisoformat(anchor_ts)
    except (ValueError, TypeError):
        return 0.0
    age = (anchor - t).total_seconds() / 86400.0  # days
    if age <= 0:
        return 1.0
    return 0.5 ** (age / half_life_days)
