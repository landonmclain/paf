"""The hot path: match what the user just said against the curator's vocabulary.

Pure and deterministic — no model, no I/O beyond loading the store — so it is safe
to run on every real user turn. The index is built from the journal's *current
record per source*, **not** from the entity map: the map unions aliases across an
entity's sources, which loses which source declared what. Here a source only
matches through terms **its own record** declared (its entities' canonical names
and aliases, its keywords, and its own name), so ``quantconnect``'s thirteen
sources are told apart by the specific thing the user said.

Scoring, per source:

- **Phrase tier.** Every declared term is a token phrase. A phrase found in the
  message at word boundaries scores ``weight x idf x length-boost``: alias /
  canonical weight 3, keyword / name weight 2, longer phrases boosted (more
  specific), and ``idf`` down-weights terms that many sources share (``qc``,
  ``lean``) so a unique term outranks a common one. Contiguous sub-phrases of
  longer terms count at half weight, so "live logs" finds a source whose keyword
  is "live logs read".
- **Token tier.** A source that shares at least two distinct content words with
  the message (beyond the ones its phrases already matched) earns a small extra
  score — the bag-of-words fallback for compound queries nothing declared whole.

A hint must clear an absolute floor and a fraction of the best score, so an
unrelated message returns nothing and a strong match is not drowned by weak ones.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._store import ENTITY_MAP_FILENAME, KeywordJournalStore, normalize_term

_TOKEN = re.compile(r"[a-z0-9]+")

ALIAS_WEIGHT = 3.0
KEYWORD_WEIGHT = 2.0
SUBPHRASE_FACTOR = 0.5
TOKEN_WEIGHT = 0.5
MIN_TOKEN_HITS = 2
MAX_PHRASE_TOKENS = 6

DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 2.5
"""Absolute floor on a hint's score (a lone shared term can't clear it)."""
DEFAULT_RELATIVE = 0.35
"""A hint must also reach this fraction of the best hint's score."""

_STOPWORDS = frozenset(
    """a about after all also am an and any are as at be been being but by can could
    did do does doing done for from get got had has have how i if in into is it its
    just let like make me more my need no not of on one or our out over please
    should so some than that the their them then there these they this to up us
    use want was we were what when where which who why will with would you your
    again able check look see show tell give take try""".split()
)


def tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, lightly stemmed (``logs`` → ``log``)."""
    return [_stem(t) for t in _TOKEN.findall(str(text).casefold())]


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


@dataclass(frozen=True)
class Hint:
    """One source worth pointing the model at."""

    ref: str
    """``skill:<id>`` or ``note:<vault-relative-path>``."""
    label: str
    """The skill id / note path."""
    kind: str
    """``"skill"`` or ``"note"``."""
    score: float
    matched: tuple[str, ...] = ()
    """The declared terms that matched, strongest first (for explainability)."""


@dataclass
class _Source:
    ref: str
    label: str
    kind: str
    tokens: set[str] = field(default_factory=set)


def label_for(ref: str) -> tuple[str, str]:
    """``(kind, label)`` for a source ref."""
    kind, _, rest = ref.partition(":")
    return kind, rest


def name_term(ref: str) -> str:
    """A source's own name as a term: the skill's short name / the note's stem."""
    _, label = label_for(ref)
    leaf = label.rsplit("/", 1)[-1]
    return leaf.removesuffix(".md")


class KeywordIndex:
    """Per-source term index with a deterministic ``match``."""

    def __init__(self) -> None:
        self._sources: dict[str, _Source] = {}
        # phrase tokens -> {ref: (weight, display term)}
        self._phrases: dict[tuple[str, ...], dict[str, tuple[float, str]]] = {}
        self._tok_df: dict[str, int] = defaultdict(int)
        # phrase tokens -> size of the largest entity that declares it as an alias
        self._group: dict[tuple[str, ...], int] = {}
        self._max_len = 1

    @property
    def size(self) -> int:
        return len(self._sources)

    @classmethod
    def from_records(
        cls,
        records: Mapping[str, dict[str, Any]],
        live_refs: Iterable[str] | None = None,
    ) -> KeywordIndex:
        """Build from ``{source_ref: journal record}`` (the store's ``latest()``)."""
        idx = cls()
        live = None if live_refs is None else set(live_refs)
        refs = [r for r in sorted(records) if live is None or r in live]
        # How many sources each entity spans: an alias is as unspecific as the
        # entity it names (``qc`` points at all of quantconnect), however few
        # sources happened to spell it out.
        entity_refs: dict[str, set[str]] = defaultdict(set)
        for ref in refs:
            for ent in records[ref].get("entities") or []:
                entity_refs[normalize_term(ent.get("canonical", ""))].add(ref)
        for ref in refs:
            idx._add(ref, records[ref], {k: len(v) for k, v in entity_refs.items()})
        idx._finish()
        return idx

    @classmethod
    def from_store(cls, directory: str | Path) -> KeywordIndex:
        """Build from a keyword store directory (journal + the map's liveness)."""
        import json

        directory = Path(directory)
        records = KeywordJournalStore(directory).latest()
        live: set[str] | None = None
        try:
            emap = json.loads((directory / ENTITY_MAP_FILENAME).read_text("utf-8"))
            if isinstance(emap, dict):
                live = {
                    s
                    for ent in emap.values()
                    if isinstance(ent, dict)
                    for s in ent.get("sources", [])
                }
        except (OSError, ValueError):
            live = None
        return cls.from_records(records, live)

    # -- build ---------------------------------------------------------------

    def _add(
        self, ref: str, rec: dict[str, Any], entity_size: Mapping[str, int]
    ) -> None:
        kind, label = label_for(ref)
        src = _Source(ref=ref, label=label, kind=kind)
        self._sources[ref] = src
        # (term, weight, size of the entity that declares it; 0 for keywords)
        declared: list[tuple[str, float, int]] = []
        for ent in rec.get("entities") or []:
            canon = ent.get("canonical", "")
            size = entity_size.get(normalize_term(canon), 1)
            declared.append((canon, ALIAS_WEIGHT, size))
            declared += [(a, ALIAS_WEIGHT, size) for a in ent.get("aliases") or []]
        declared += [(k, KEYWORD_WEIGHT, 0) for k in rec.get("keywords") or []]
        declared.append((name_term(ref), KEYWORD_WEIGHT, 0))
        for term, weight, size in declared:
            toks = tuple(tokens(term))
            if not toks or all(t in _STOPWORDS for t in toks):
                continue
            src.tokens.update(t for t in toks if t not in _STOPWORDS)
            # Only a bare name stands for the whole entity; a multi-word alias
            # ("QC live logs") is specific on its own, however big the entity is.
            group = size if len(toks) == 1 else 0
            self._put(ref, toks, weight, term, group)
            if len(toks) >= 3:  # contiguous sub-phrases of long terms
                for n in range(2, len(toks)):
                    for i in range(len(toks) - n + 1):
                        sub = toks[i : i + n]
                        if all(t in _STOPWORDS for t in sub):
                            continue
                        self._put(ref, sub, weight * SUBPHRASE_FACTOR, term)

    def _put(
        self,
        ref: str,
        toks: tuple[str, ...],
        weight: float,
        display: str,
        group: int = 0,
    ) -> None:
        if len(toks) > MAX_PHRASE_TOKENS:
            return
        if group > self._group.get(toks, 0):
            self._group[toks] = group
        slot = self._phrases.setdefault(toks, {})
        if ref not in slot or weight > slot[ref][0]:
            slot[ref] = (weight, display)
        self._max_len = max(self._max_len, len(toks))

    def _finish(self) -> None:
        self._tok_df = defaultdict(int)
        for src in self._sources.values():
            for t in src.tokens:
                self._tok_df[t] += 1

    def _idf(self, df: int) -> float:
        return math.log(1 + self.size / max(1, df))

    # -- match ---------------------------------------------------------------

    def match(
        self,
        text: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        relative: float = DEFAULT_RELATIVE,
    ) -> list[Hint]:
        """Rank sources against ``text``; ``[]`` when nothing clears the bar."""
        msg = tokens(text)
        if not msg or not self._sources:
            return []
        scores: dict[str, float] = defaultdict(float)
        matched: dict[str, list[tuple[float, str]]] = defaultdict(list)
        covered: dict[str, set[str]] = defaultdict(set)  # tokens a phrase explained

        seen: set[tuple[str, ...]] = set()
        for n in range(1, min(self._max_len, len(msg)) + 1):
            for i in range(len(msg) - n + 1):
                gram = tuple(msg[i : i + n])
                if gram in seen or gram not in self._phrases:
                    continue
                seen.add(gram)
                owners = self._phrases[gram]
                if n == 1 and gram[0] in _STOPWORDS:
                    continue
                idf = self._idf(max(len(owners), self._group.get(gram, 0)))
                boost = 1 + 0.5 * (n - 1)
                for ref, (weight, display) in owners.items():
                    gained = weight * idf * boost
                    scores[ref] += gained
                    matched[ref].append((gained, display))
                    covered[ref].update(gram)

        qtokens = {t for t in msg if t not in _STOPWORDS}
        if len(qtokens) >= MIN_TOKEN_HITS:
            for ref, src in self._sources.items():
                extra = (qtokens & src.tokens) - covered[ref]
                if len(extra) >= MIN_TOKEN_HITS:
                    scores[ref] += TOKEN_WEIGHT * sum(
                        self._idf(self._tok_df[t]) for t in extra
                    )

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if not ranked or ranked[0][1] < min_score:
            return []
        floor = max(min_score, ranked[0][1] * relative)
        out: list[Hint] = []
        for ref, score in ranked:
            if score < floor or len(out) >= top_k:
                break
            src = self._sources[ref]
            terms: list[str] = []
            for _g, display in sorted(matched[ref], key=lambda m: -m[0]):
                if display not in terms:
                    terms.append(display)
            out.append(
                Hint(
                    ref=ref,
                    label=src.label,
                    kind=src.kind,
                    score=round(score, 2),
                    matched=tuple(terms[:3]),
                )
            )
        return out


class IndexCache:
    """Keeps a :class:`KeywordIndex` for one store directory, rebuilt on change.

    The hot path calls :meth:`get` every real user turn; the cost when nothing
    changed is two ``stat`` calls. The index is rebuilt when the journal or the
    entity map changes on disk (the curator rewrites both atomically), and is empty
    when the store does not exist yet.
    """

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._stamp: tuple[Any, ...] | None = None
        self._index = KeywordIndex()

    @property
    def size(self) -> int:
        return self._index.size

    def _signature(self) -> tuple[Any, ...]:
        from ._store import JOURNAL_FILENAME

        sig: list[Any] = []
        for name in (JOURNAL_FILENAME, ENTITY_MAP_FILENAME):
            try:
                st = (self._directory / name).stat()
                sig.append((st.st_mtime_ns, st.st_size))
            except OSError:
                sig.append(None)
        return tuple(sig)

    def get(self) -> KeywordIndex:
        stamp = self._signature()
        if stamp != self._stamp:
            self._index = (
                KeywordIndex()
                if stamp == (None, None)
                else KeywordIndex.from_store(self._directory)
            )
            self._stamp = stamp
        return self._index
