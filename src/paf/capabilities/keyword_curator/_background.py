"""The curator worker: discover changed sources, curate, append, rebuild, advance.

One pass (:func:`run_keyword_pass`):

1. Compare the scanned sources against the watermark → *changed* (new or
   modified) and *vanished* (tracked but gone).
2. Batch the changed sources and curate each batch, passing the entities already
   in the map so later batches reuse earlier canonical names.
3. Append each result to the journal, then advance + save the watermark for that
   batch — so a failure part-way keeps the finished batches and retries only the
   rest.
4. Rebuild ``entity-map.json`` from the journal, dropping sources no longer live.

The worker is the sole writer of the keyword store. Registration and the
turn-gated trigger are a later phase; this is a plain callable so it can be unit
tested with a fake curator and run by hand.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._curate import CurateFn, CurateResult, curate_batch
from ._sources import SourceItem, snapshots
from ._store import (
    ENTITY_MAP_FILENAME,
    KeywordJournalStore,
    derive_entity_map,
    make_record,
    write_entity_map,
)
from ._watermark import KeywordWatermark

logger = logging.getLogger(__name__)

DEFAULT_BATCH_CHARS = 16_000
"""~4k tokens of source text per batch — the reviewer's token-limit blowup is why
batching exists; never send a whole family in one pass."""
DEFAULT_MAX_BATCH_ITEMS = 8
DEFAULT_MAX_KNOWN = 250


@dataclass
class PassResult:
    """What one pass did."""

    keyed: list[str] = field(default_factory=list)
    """Refs curated and recorded this pass."""
    failed: dict[str, str] = field(default_factory=dict)
    """Refs whose batch failed → error text. Not advanced; retried next pass."""
    removed: list[str] = field(default_factory=list)
    """Refs dropped from the watermark because their source vanished."""
    batches: int = 0
    entities: int = 0
    """Entities in the rebuilt map."""

    @property
    def ok(self) -> bool:
        return not self.failed


def make_batches(
    items: Sequence[SourceItem],
    *,
    max_chars: int = DEFAULT_BATCH_CHARS,
    max_items: int = DEFAULT_MAX_BATCH_ITEMS,
) -> list[list[SourceItem]]:
    """Greedy, order-preserving batches bounded by total digest chars and count.

    A single source larger than ``max_chars`` still gets its own batch.
    """
    batches: list[list[SourceItem]] = []
    cur: list[SourceItem] = []
    used = 0
    for item in items:
        cost = len(item.text)
        if cur and (used + cost > max_chars or len(cur) >= max_items):
            batches.append(cur)
            cur, used = [], 0
        cur.append(item)
        used += cost
    if cur:
        batches.append(cur)
    return batches


def known_entities(
    journal: KeywordJournalStore,
    live_refs: set[str],
    *,
    max_known: int = DEFAULT_MAX_KNOWN,
) -> list[dict[str, Any]]:
    """Entities already in the map, most-referenced first, capped for the prompt."""
    derived = derive_entity_map(journal.read_all(), live_refs=live_refs)
    ranked = sorted(
        derived.values(), key=lambda e: (-len(e["sources"]), e["canonical"].casefold())
    )
    return ranked[:max_known]


def _record_for(item: SourceItem, res: CurateResult) -> dict[str, Any]:
    return make_record(
        item.ref,
        item.source_type,
        res.keywords,
        [e.model_dump() for e in res.entities],
    )


async def run_keyword_pass(
    directory: str | os.PathLike,
    sources: Mapping[str, SourceItem],
    *,
    curate: CurateFn = curate_batch,
    batch_chars: int = DEFAULT_BATCH_CHARS,
    max_batch_items: int = DEFAULT_MAX_BATCH_ITEMS,
    max_known: int = DEFAULT_MAX_KNOWN,
    limit: int | None = None,
) -> PassResult:
    """Run one curation pass over ``sources`` into the keyword store at ``directory``.

    ``limit`` caps how many changed sources are curated this pass (a trial run);
    the rest stay changed and are picked up next time.
    """
    journal = KeywordJournalStore(directory)
    wm = KeywordWatermark.load(directory)
    current = snapshots(sources)

    changed = wm.changed(current)
    vanished = wm.vanished(current)
    if limit is not None:
        changed = changed[: max(0, limit)]

    result = PassResult()
    live = set(sources)

    for batch in make_batches(
        [sources[ref] for ref in changed],
        max_chars=batch_chars,
        max_items=max_batch_items,
    ):
        result.batches += 1
        known = known_entities(journal, live, max_known=max_known)
        try:
            answers = await curate(batch, known)
        except Exception as exc:  # noqa: BLE001 - one bad batch must not sink the pass
            logger.warning("keyword curator batch failed: %s", exc)
            for item in batch:
                result.failed[item.ref] = f"{type(exc).__name__}: {exc}"
            continue
        by_ref = {a.source_ref: a for a in answers}
        done: dict[str, tuple[float, int]] = {}
        for item in batch:
            ans = by_ref.get(item.ref)
            if ans is None:
                result.failed[item.ref] = "curator returned no result for this source"
                continue
            journal.append(_record_for(item, ans))
            done[item.ref] = item.snapshot
            result.keyed.append(item.ref)
        # Journal first, watermark second: a crash between them only re-curates.
        wm.advance(done)
        wm.save(directory)

    if vanished:
        wm.advance({}, remove=vanished)
        wm.save(directory)
        result.removed = list(vanished)

    entity_map = derive_entity_map(journal.read_all(), live_refs=live)
    map_missing = not (Path(directory) / ENTITY_MAP_FILENAME).exists()
    if result.keyed or vanished or map_missing:
        write_entity_map(directory, entity_map)
    result.entities = len(entity_map)
    return result
