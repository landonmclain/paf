"""Append-only override sidecar + derived per-target override state.

The three-layer memory spine (journal -> index -> render) is extended with a
**narrow curation channel**: an append-only log of *intents* to modify existing
memory (archive / tier-change / reject). It sits beside the journal / index /
``MEMORY.md`` and is consumed by the render pass as one deterministic,
mechanical fold — no LLM, no mutation of the journal or index.

Two writers, one trust boundary:

- **assistant** — may only `archive` (it proposes "this fact is stale/wrong").
  An assistant archive is **always pending** until a reviewer resolves it.
- **reviewer** — may `archive` / `tier_change` / `reject`, and applies
  **directly** (it IS the approver). A reviewer op resolves any pending
  assistant archive on the same line (last-write-wins per target).

Derived state per ``target_line`` (a stable journal line number):

    none            -> no override; fact renders as-is
    pending         -> assistant archive, no verdict yet; fact shown + marker
    applied_archive -> reviewer archive; fact dropped from injection
    applied_tier    -> reviewer tier_change; fact moves under the new tier
    rejected        -> reviewer reject; fact stays as-was, marker cleared

Nothing is destroyed: the journal holds every fact forever; an archive can be
reverted by appending ``reject``. The sidecar is append-only; the *current*
state is always derived from the log (same journal->index pattern, applied to
the sidecar itself).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime

from paf.capabilities._storage import _append_jsonl_atomic

_OVERRIDES_FILENAME = "memory-overrides.jsonl"
_SCHEMA = 1

# Ops by writer.
_ASSISTANT_OPS = frozenset({"archive"})
_REVIEWER_OPS = frozenset({"archive", "tier_change", "reject", "supersede"})

# Derived per-target statuses (see module docstring).
_STATUS_NONE = "none"
_STATUS_PENDING = "pending"
_STATUS_APPLIED_ARCHIVE = "applied_archive"
_STATUS_APPLIED_TIER = "applied_tier"
_STATUS_REJECTED = "rejected"
_STATUS_SUPERSEDED = "superseded"


class OverrideState:
    """Derived, per-target override state (immutable value).

    ``status`` is one of the ``_STATUS_*`` constants. ``tier`` is set only when
    ``status == applied_tier`` (the target tier for a tier_change). ``by`` /
    ``reason`` / ``at`` reflect the **resolving** record (the reviewer verdict
    for applied/rejected, the assistant intent for pending).
    """

    __slots__ = ("status", "tier", "by", "reason", "at", "superseded_by")

    def __init__(
        self,
        status: str,
        *,
        tier: str | None = None,
        by: str = "",
        reason: str = "",
        at: str = "",
        superseded_by: int | None = None,
    ) -> None:
        self.status = status
        self.tier = tier
        self.by = by
        self.reason = reason
        self.at = at
        self.superseded_by = superseded_by

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"OverrideState(status={self.status!r}, tier={self.tier!r}, "
            f"by={self.by!r}, reason={self.reason!r})"
        )

    # --- convenience predicates (used by the render hook) -------------------

    @property
    def is_pending(self) -> bool:
        """True when an assistant archive is awaiting a reviewer verdict."""
        return self.status == _STATUS_PENDING

    @property
    def is_applied_archive(self) -> bool:
        """True when the reviewer archived the fact (drop from injection)."""
        return self.status == _STATUS_APPLIED_ARCHIVE

    @property
    def is_applied_tier(self) -> bool:
        """True when the reviewer moved the fact to a new tier."""
        return self.status == _STATUS_APPLIED_TIER

    @property
    def is_superseded(self) -> bool:
        """True when a newer fact replaced this one's value (it leaves the
        current sections; ``superseded_by`` names the replacement)."""
        return self.status == _STATUS_SUPERSEDED


def _now_iso() -> str:
    """UTC ISO-8601 timestamp for the `at` field (wall clock, audit only)."""
    return datetime.now(UTC).isoformat()


class OverrideStore:
    """Append-only override sidecar (one JSON record per line).

    ``directory`` is the memory scope dir — the same one that holds
    ``MEMORY.md``, the journal, and the index. The file is
    ``<directory>/memory-overrides.jsonl``.
    """

    def __init__(self, directory: str | os.PathLike) -> None:
        self._dir = os.fspath(directory)
        self._path = os.path.join(self._dir, _OVERRIDES_FILENAME)
        self._lock = threading.RLock()

    @property
    def directory(self) -> str:
        """The memory scope dir this sidecar sits in (journal/index/MEMORY.md home)."""
        return self._dir

    # --- append ------------------------------------------------------------

    def append(
        self,
        *,
        op: str,
        target_line: int,
        by: str,
        reason: str = "",
        tier: str | None = None,
        at: str | None = None,
        superseded_by: int | None = None,
        detail: dict | None = None,
    ) -> dict:
        """Append one override record; returns it (echoed for the caller).

        ``op`` must be valid for ``by``: ``archive`` for the assistant;
        ``archive``/``tier_change``/``reject``/``supersede`` for the reviewer.
        ``tier`` is required for ``tier_change`` and rejected for others;
        ``superseded_by`` (the replacing line) is required for ``supersede``
        and rejected for others. ``detail`` is a free-form audit dict (the
        pairwise reviewer records label, judge model, and the pair).
        """
        if by == "assistant":
            if op not in _ASSISTANT_OPS:
                raise ValueError(
                    f"assistant may only {sorted(_ASSISTANT_OPS)}, not {op!r}"
                )
        elif by == "reviewer":
            if op not in _REVIEWER_OPS:
                raise ValueError(
                    f"reviewer op must be one of {sorted(_REVIEWER_OPS)}, not {op!r}"
                )
        else:
            raise ValueError(f"unknown override writer {by!r}")
        if target_line < 1:
            raise ValueError(f"target_line must be >= 1, got {target_line}")
        if op == "tier_change" and tier is None:
            raise ValueError("tier_change requires a target tier")
        if op != "tier_change" and tier is not None:
            raise ValueError(f"op {op!r} does not take a tier")
        if op == "supersede" and (superseded_by is None or superseded_by < 1):
            raise ValueError("supersede requires superseded_by (the replacing line)")
        if op != "supersede" and superseded_by is not None:
            raise ValueError(f"op {op!r} does not take superseded_by")

        record = {
            "v": _SCHEMA,
            "op": op,
            "target_line": int(target_line),
            "by": by,
            "reason": reason,
            "at": at or _now_iso(),
        }
        if tier is not None:
            record["tier"] = tier
        if superseded_by is not None:
            record["superseded_by"] = int(superseded_by)
        if detail:
            record["detail"] = detail

        with self._lock:
            _append_jsonl_atomic(self._path, [record])
        return record

    # --- reads -------------------------------------------------------------

    def read_all(self) -> list[dict]:
        """All override records, in append order (raw, unfiltered)."""
        out: list[dict] = []
        if not os.path.exists(self._path):
            return out
        with self._lock:
            with open(self._path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        # A torn/corrupt line is skipped — the sidecar's current
                        # state is always derivable from the good records.
                        continue
        return out


# ---------------------------------------------------------------------------
# fold: raw append-only log -> derived per-target override state
# ---------------------------------------------------------------------------


def fold_overrides(records: list[dict]) -> dict[int, OverrideState]:
    """Derive the per-target override map from the raw append-only log.

    Deterministic: per ``target_line``, records are applied in append order with
    last-write-wins. An assistant ``archive`` puts the target into ``pending``;
    a later reviewer op on the same line resolves it (``archive`` -> applied,
    ``tier_change`` -> applied_tier, ``reject`` -> rejected). A reviewer op with
    no prior assistant intent still applies directly (the reviewer curates on
    its own). Returns ``{target_line: OverrideState}`` for every line that has
    any override.
    """
    state: dict[int, OverrideState] = {}
    for rec in records:
        op = rec.get("op")
        target = rec.get("target_line")
        by = rec.get("by")
        if not isinstance(target, int) or op not in _REVIEWER_OPS:
            continue
        if by == "assistant" and op == "archive":
            # pending until a reviewer resolves it
            state[target] = OverrideState(
                _STATUS_PENDING,
                by=by,
                reason=rec.get("reason", ""),
                at=rec.get("at", ""),
            )
        elif by == "reviewer":
            if op == "archive":
                state[target] = OverrideState(
                    _STATUS_APPLIED_ARCHIVE,
                    by=by,
                    reason=rec.get("reason", ""),
                    at=rec.get("at", ""),
                )
            elif op == "tier_change":
                state[target] = OverrideState(
                    _STATUS_APPLIED_TIER,
                    tier=rec.get("tier"),
                    by=by,
                    reason=rec.get("reason", ""),
                    at=rec.get("at", ""),
                )
            elif op == "reject":
                state[target] = OverrideState(
                    _STATUS_REJECTED,
                    by=by,
                    reason=rec.get("reason", ""),
                    at=rec.get("at", ""),
                )
            elif op == "supersede":
                state[target] = OverrideState(
                    _STATUS_SUPERSEDED,
                    by=by,
                    reason=rec.get("reason", ""),
                    at=rec.get("at", ""),
                    superseded_by=rec.get("superseded_by"),
                )
    return state


def load_overrides(directory: str | os.PathLike) -> dict[int, OverrideState]:
    """Load + fold the sidecar beside ``directory`` into a per-target map.

    Missing/corrupt sidecar -> empty map (no overrides applied).
    """
    store = OverrideStore(directory)
    return fold_overrides(store.read_all())
