"""Review aid for the entity map: surfaces likely duplicates and ambiguity.

The main risk of a curator is *entity resolution drift* — the same thing named
two ways across batches, so the map looks full but connects nothing. This pure
function flags what a human should look at before building on the map. It
changes nothing.
"""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from ._store import normalize_term

SIMILAR_RATIO = 0.82
OVERLAP_JACCARD = 0.5


def find_issues(entity_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Likely problems in ``entity_map``.

    - ``ambiguous_terms`` — a normalized name/alias claimed by more than one
      entity (the hot path could not tell which was meant).
    - ``similar_names`` — pairs of canonical names that are near-spellings or one
      contains the other's words (possible duplicates or a missing alias).
    - ``overlapping_sources`` — entity pairs that share most of their sources
      (Jaccard >= 0.5): often one thing split in two, which name similarity
      misses (``gws`` vs ``google-workspace``).
    - ``single_source`` / ``no_aliases`` — entities that connect only one source
      or can only be detected by their canonical name.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    for ent in entity_map.values():
        canon = ent["canonical"]
        for term in [canon, *ent.get("aliases", [])]:
            key = normalize_term(term)
            if key:
                owners[key].add(canon)
    ambiguous = {k: sorted(v) for k, v in sorted(owners.items()) if len(v) > 1}

    names = sorted(entity_map, key=lambda n: normalize_term(n))
    norm = {n: normalize_term(n) for n in names}
    similar: list[tuple[str, str, str]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            na, nb = norm[a], norm[b]
            ratio = SequenceMatcher(None, na, nb).ratio()
            if ratio >= SIMILAR_RATIO:
                similar.append((a, b, f"spelling ~{ratio:.2f}"))
                continue
            ta, tb = set(na.split()), set(nb.split())
            if ta and tb and (ta < tb or tb < ta) and len(ta | tb) > 1:
                similar.append((a, b, "one name contains the other's words"))

    overlapping: list[tuple[str, str, str]] = []
    srcs = {n: set(entity_map[n].get("sources", [])) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sa, sb = srcs[a], srcs[b]
            if not sa or not sb:
                continue
            jac = len(sa & sb) / len(sa | sb)
            if jac >= OVERLAP_JACCARD:
                overlapping.append(
                    (a, b, f"{len(sa & sb)} shared of {len(sa | sb)} sources")
                )

    return {
        "entities": len(entity_map),
        "overlapping_sources": overlapping,
        "ambiguous_terms": ambiguous,
        "similar_names": similar,
        "single_source": sorted(
            n for n, e in entity_map.items() if len(e.get("sources", [])) == 1
        ),
        "no_aliases": sorted(n for n, e in entity_map.items() if not e.get("aliases")),
    }


def format_issues(issues: dict[str, Any]) -> str:
    """Plain-text rendering of :func:`find_issues` for the CLI."""
    lines = [f"{issues['entities']} entities"]
    if issues["ambiguous_terms"]:
        lines.append("\nAmbiguous terms (claimed by several entities):")
        for term, owners in issues["ambiguous_terms"].items():
            lines.append(f"  {term!r}: {', '.join(owners)}")
    if issues["similar_names"]:
        lines.append("\nPossible duplicates:")
        for a, b, why in issues["similar_names"]:
            lines.append(f"  {a}  ~  {b}   ({why})")
    if issues["overlapping_sources"]:
        lines.append("\nEntities sharing most of their sources (split duplicates?):")
        for a, b, why in issues["overlapping_sources"]:
            lines.append(f"  {a}  ~  {b}   ({why})")
    lines.append(
        f"\nSingle-source entities: {len(issues['single_source'])}"
        f" | no aliases: {len(issues['no_aliases'])}"
    )
    return "\n".join(lines)
