"""Background memory-consolidation consumer — the ``MemoryConsolidationQueue``.

Wires the framework-level work-queue engine (:class:`paf.background.BackgroundQueueService`)
to the memory store, fired by the *watermark* trigger (:mod:`._watermark`)
instead of eviction.

This queue runs the owner **out-of-band** — the turn returns immediately and a
background worker drains the job. The turn never blocks on the owner's LLM work.
The eviction tier (``TurnBoundSlidingWindow``) is **trim-only** and never runs
the owner at all; the watermark is the sole driver, and the queue's FIFO +
``write_lock`` serialize journal writes so paths never stomp each other.

The heavy seams are injectable so the service + trigger mechanics can be
unit-tested without a model call or a fake persistence substrate:

- ``span_source`` — given a ``(run_id, turn_id)`` target, returns the unseen
  span's messages to fold. **Supplied by the builder**, bound to the same
  ``HistorySource`` the search capability holds (the queue never reconstructs a
  persistence store from a directory string). Tests supply a synthetic one.
- ``consolidate_span`` — folds a resolved span's messages into the journal.
  Defaults to the real owner (:func:`._consolidate.consolidate`); tests inject a
  lightweight stand-in (still journal-fenced, so correctness is honest).

The consumer writes through the **same** journal + ``write_render`` as the
eviction path and the benchmark — the measured contract is untouched (see the
watermark spec's "does this break the benchmark" section).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from paf.background import BackgroundQueueService, QueueState

from .._hashes import _content_hash
from .._turns import user_anchor_indices
from ._journal import JournalStore
from ._watermark import MemoryWatermark

# Cross-run overlap reconciliation (the memory-family lesson): each run's
# persisted history is CUMULATIVE — a snapshot carries the whole conversation
# from turn 0 forward — so spanning multiple runs re-collects prior runs' turns
# unless we dedup by turn identity (see ``resolve_unseen_span``). We use our own
# content-stable ``_content_hash`` rather than the harness ``message_hash``: the
# latter hashes raw serialized bytes including pydantic's per-instantiation
# ``timestamp``, so byte-identical content re-instantiated across runs/replay
# hashes differently and overlap never fires (the resume-duplication bug).

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage  # noqa: F401
    from pydantic_ai_harness.conversation_search import HistorySource

    from ._pairwise import ReviewReport

log = logging.getLogger(__name__)


def _turn_bounds(anchors: list[int], turn: int, n: int) -> tuple[int, int]:
    """Inclusive ``[start, end)`` message-slice for one user-turn's turn record.

    A turn spans from its user-prompt anchor to just before the next anchor.
    ``anchors[turn]`` is the anchor message index; the turn ends at the next
    anchor (or the end of the run).
    """
    start = anchors[turn] if 0 <= turn < len(anchors) else n
    end = anchors[turn + 1] if turn + 1 < len(anchors) else n
    return start, end


@dataclass(frozen=True)
class ResolvedSpan:
    """The persisted span to fold, plus where the watermark must advance to.

    ``messages`` are the ordered messages of **complete** turns: every full,
    settled turn newer than the stored watermark and strictly older than the
    in-flight tip turn (so a turn that is still being answered is never folded).
    ``watermark`` is the ``{run_id, turn_id}`` ref of the last complete turn
    included — the position the durable watermark advances to.
    """

    messages: list[ModelMessage]
    watermark: MemoryWatermark


async def resolve_unseen_span(
    source: HistorySource,
    watermark: MemoryWatermark,
    *,
    tip_run: str,
    tip_turn: int,
    conversation_id: str | None = None,
) -> ResolvedSpan:
    """Resolve the persisted span the owner hasn't yet consumed — **full turns only**.

    Reads the real persistent store — ``list_runs`` (ascending by ``started_at``)
    then ``run_history`` per run — and returns, as :class:`ResolvedSpan`, the
    ordered messages of complete turns that are:

      * strictly newer than the watermark's consumed position, **and**
      * strictly older than the in-flight tip ``(tip_run, tip_turn)``.

    The final turn (the one currently being answered, whose response has not
    settled) is deliberately **excluded** — the fold stops on the second-to-last
    turn — so only finished turns enter the journal. A single-turn transcript
    therefore folds nothing.

    When ``conversation_id`` is given, ONLY runs belonging to that conversation
    are considered (``run.conversation_id == conversation_id``). Memory's owner
    folds cross-conversation per agent (``None`` = all runs, the default); the
    thread summarizer is strictly per-conversation and passes its scoped
    ``{profile}:{cid}`` here so one conversation's beats never bleed into
    another's.

    The watermark is a ``{run_id, turn_id}`` ref (see ``_watermark.py``):
      - runs strictly before the watermark's run: fully consumed, skipped;
      - the watermark's run: turns ``<= turn_id`` consumed, the tail is new;
      - runs after, and a fresh watermark (nothing consumed): everything up to
        (but not including) the tip is new.

    Relies on the store's natural ordering — ``list_runs`` sorted by
    ``started_at`` ascending and turns ordinal within a run — so **ref alone
    anchors the watermark**; no timestamp is required (per design).
    """
    runs = await source.list_runs()

    if conversation_id is not None:
        runs = [
            r for r in runs if getattr(r, "conversation_id", None) == conversation_id
        ]

    tip_idx = next(
        (i for i, r in enumerate(runs) if r.run_id == tip_run),
        len(runs) - 1,
    )

    collected: list[ModelMessage] = []
    # Content-hash identity of every complete turn already folded into
    # ``collected`` (per-turn = concatenated per-message content hashes). Runs
    # are CUMULATIVE — each snapshot carries the whole conversation from turn 0
    # forward — so a later run repeats earlier runs' turns verbatim. Matching a
    # run against this set tells us exactly how many leading turns are already
    # covered, regardless of where the watermark sits (see the cross-run note
    # below).
    seen_turn_keys: set[str] = set()
    last_complete: MemoryWatermark | None = None

    # Position of the watermark's run. Runs strictly older than it are fully
    # consumed and MUST be skipped even when a legacy watermark carries no
    # time anchor (consumed_ts=None): without this, the default first_turn=0
    # below would scoop the ENTIRE pre-watermark history into the fold span,
    # blowing the owner's context and silently 0-folding — the live stall.
    wm_run_idx = 0
    for _i, _r in enumerate(runs):
        if _r.run_id == watermark.run_id:
            wm_run_idx = _i
            break

    # Seed ``seen_turn_keys`` with the turns the watermark has ALREADY consumed
    # (its run's turns ``0..turn_id``). A later cumulative run repeats that
    # history verbatim, so without this seed the run would re-collect turns the
    # owner folded on a PREVIOUS cadence — the many-run duplication. The seed is
    # derived from the durable watermark (not from this call's ``collected``),
    # so it is correct even when the entire span below is a prior-cadence carry.
    if watermark.turn_id is not None:
        wm_hist = await source.run_history(run_id=watermark.run_id)
        if wm_hist:
            wm_anchors = user_anchor_indices(wm_hist)
            wm_n = len(wm_hist)
            for _t in range(min(watermark.turn_id + 1, len(wm_anchors))):
                _s, _e = _turn_bounds(wm_anchors, _t, wm_n)
                seen_turn_keys.add("\n".join(_content_hash(m) for m in wm_hist[_s:_e]))

    for i, r in enumerate(runs):
        if i < wm_run_idx:
            # Strictly older than the watermark's own run: fully consumed.
            continue
        hist = await source.run_history(run_id=r.run_id)
        if not hist:
            continue

        anchors = user_anchor_indices(hist)
        if not anchors:
            continue

        n = len(hist)

        # Intra-run continuation: when this run is the watermark's run, the
        # {run_id, turn_id} ref tells us exactly which turns are already
        # consumed. A strictly older run (started before the time anchor, and
        # not the run holding the ref) is fully consumed and skipped.
        first_turn = 0
        if watermark.run_id == r.run_id and watermark.turn_id is not None:
            first_turn = watermark.turn_id + 1
        elif (
            watermark.consumed_ts is not None
            and watermark.run_id != r.run_id
            and r.started_at <= watermark.consumed_ts
        ):
            continue

        # CROSS-RUN OVERLAP (the memory-family lesson): each run's history is
        # CUMULATIVE — a snapshot carries the WHOLE conversation from turn 0
        # forward, so a later run repeats every prior run's turns verbatim.
        # Collecting from turn 0 on a later run re-adds the already-collected
        # prefix. The earlier prefix↔suffix reconciliation only worked while
        # the collected span stayed prefix-aligned at turn 0 (fresh watermark);
        # the moment a mid-history watermark advanced ``collected`` past turn 0,
        # a later cumulative run's prefix (always at turn 0) no longer matched,
        # overlap computed 0, and the run re-collected its entire history —
        # the many-run duplication seen in real conversations.
        #
        # Correct, structure-independent reconciliation: a turn is an overlap
        # iff its content identity is already in ``seen_turn_keys``. Cumulative
        # runs repeat leading turns, so this skips to the first genuinely-new
        # turn no matter how many runs precede it or where the watermark sits.
        for t in range(first_turn, len(anchors)):
            # Fold full turns only: stop strictly before the in-flight tip. A
            # turn at or after (tip_idx, tip_turn) has not finished settling.
            if (i, t) >= (tip_idx, tip_turn):
                break
            start, end = _turn_bounds(anchors, t, n)
            slice_msgs = hist[start:end]
            turn_key = "\n".join(_content_hash(m) for m in slice_msgs)
            # Duplicate turn from an earlier cumulative run: already folded.
            if turn_key in seen_turn_keys:
                continue
            collected.extend(slice_msgs)
            seen_turn_keys.add(turn_key)
            # Advance the ref AND the time anchor. consumed_ts is this run's
            # started_at — monotonic across runs, so the next cadence can gauge
            # progress even after a run boundary resets local turn_id.
            last_complete = MemoryWatermark(
                run_id=r.run_id, turn_id=t, consumed_ts=r.started_at
            )

    # Fresh watermark with nothing folded keeps its ref (never points past the
    # consumed position; an empty span leaves it untouched).
    final_wm = last_complete if last_complete is not None else watermark
    return ResolvedSpan(messages=collected, watermark=final_wm)


@dataclass(frozen=True)
class ConsolidationJob:
    """A request to fold the unseen persisted span into durable memory.

    ``run_id`` / ``turn_id`` mark the current (in-flight) tip position the
    trigger resolved against. ``messages`` optionally carries the **pre-resolved
    span** the trigger already computed; when present the worker folds it
    directly instead of re-resolving against the (not-yet-advanced) watermark.
    ``watermark`` optionally carries the **resolved target position** (the
    ``ResolvedSpan.watermark`` the trigger advanced to) so the worker, on a
    successful fold, advances THE OWNER WATERMARK to that exact position —
    not to the stale on-disk watermark it loaded for cadence reference.
    """

    run_id: str
    turn_id: int
    messages: list[ModelMessage] | None = None
    watermark: MemoryWatermark | None = None
    project: str | None = None
    """The conversation's project slug, resolved by the trigger when it
    schedules the span (a reassignment after that applies to the NEXT span).
    ``None`` = untagged conversation."""


#: Type of the span-source seam: resolve "(run_id, up_to_turn_id)" into the
#: :class:`ResolvedSpan` (complete-turn messages + the advance-to watermark).
#: This is where the real persistence read sits (``SnapshotHistorySource`` /
#: ``resolve_unseen_span``); a test supplies a synthetic span.
SpanSource = Callable[[str, int], Awaitable[ResolvedSpan]]

#: Type of the consolidate seam: fold a span's messages into the journal.
#: Defaults to the real owner; a test injects a journal-fenced stand-in.
ConsolidateSpan = Callable[[list["ModelMessage"], str, MemoryWatermark], Awaitable[int]]


@dataclass
class FoldOutcome:
    """What one consolidation job did, reported once the whole job is over.

    The pairwise reviewer runs inside the job, after the fold, so the fold's
    return alone is not "done": a status row that says done at that point is
    wrong for the minutes the reviewer is still working. ``error`` is a failure
    after the fold committed (render/index); ``review_error`` is a reviewer
    failure, which never fails the fold.
    """

    records: int
    review: ReviewReport | None = None
    review_error: str | None = None
    error: str | None = None


#: Notified just before the pairwise reviewer starts.
OnReviewing = Callable[[], Awaitable[None]]
#: Notified once a job that reached the fold is over, however it ended.
OnFoldFinished = Callable[[FoldOutcome], Awaitable[None]]


class MemoryConsolidationQueue(BackgroundQueueService[ConsolidationJob]):
    """Background owner queue fired by the watermark trigger.

    One instance == one queue == one worker per session (the engine's
    composition model). ``process_job`` folds the unseen span since the
    watermark via ``consolidate_span``, advances the watermark file, and
    re-renders ``MEMORY.md``.

    The span to fold is **provided by the caller** (``span_source``) — the
    queue never reconstructs a persistence store from a directory string. The
    agent that builds this queue supplies a ``span_source`` bound to the same
    ``HistorySource`` the search capability already holds (e.g. a
    ``SqliteStepStore`` / ``FileStepStore`` the runtime persists to). This keeps
    the queue a thin, honest consumer: it does not guess at where history
    lives. A test supplies a synthetic span source standing in for persistence.

    ``consolidate_span`` defaults to the real owner (:func:`._consolidate.consolidate`);
    a test injects a journal-fenced stand-in so the service + trigger mechanics
    are proven without an LLM call.

    NOTE: deliberately NOT a ``@dataclass``. Subclassing
    :class:`BackgroundQueueService` requires ``super().__init__()`` to run so
    the engine's ``_states`` dict is created (see ``tests/test_background.py``:
    a dataclass with its own fields generates an ``__init__`` that skips the
    base). We write an explicit ``__init__`` that calls the base, mirroring that
    established consumer pattern.
    """

    name: str = "memory-consolidation"

    def __init__(
        self,
        *,
        memory_directory: str = "",
        agent_name: str = "demo_assistant",
        name: str = "memory-consolidation",
        span_source: SpanSource | None = None,
        consolidate_span: ConsolidateSpan | None = None,
        pairwise_review: bool = False,
        on_reviewing: OnReviewing | None = None,
        on_finished: OnFoldFinished | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.memory_directory = memory_directory
        self.agent_name = agent_name
        self.span_source = span_source
        self.consolidate_span = consolidate_span
        # Run the pairwise reviewer (memory/_pairwise.py) right after each fold
        # that wrote facts. Spec `memory: pairwise_review: true`.
        self.pairwise_review = pairwise_review
        # Status hooks for the coordinator's process row (see FoldOutcome).
        self.on_reviewing = on_reviewing
        self.on_finished = on_finished
        # The job being processed's project — read by a coordinator-supplied
        # consolidate_span (which keeps the 3-arg hook signature) so the
        # extractor still gets the conversation's project. Jobs are FIFO on
        # one worker, so this never mixes two jobs.
        self.current_project: str | None = None
        # The step store's history source, so the extractor's source_refs are
        # numbered in each turn's own run (read_turn's frame). Set by the
        # coordinator; None falls back to slice ordinals (single-run paths).
        self.history_source: HistorySource | None = None

    def job_trace(self, session_id: str, job: ConsolidationJob) -> tuple[str, dict]:
        """Each consolidation is its OWN trace: memory is per-agent, not part of
        whichever conversation turn happened to trigger it (see
        ``BackgroundQueueService.job_trace``)."""
        attrs = {"agent": session_id, "run_id": job.run_id, "turn_id": job.turn_id}
        return (
            f"paf.memory.consolidate.{session_id}",
            {k: v for k, v in attrs.items() if v is not None},
        )

    async def process_job(
        self,
        session_id: str,
        job: ConsolidationJob,
        state: QueueState[ConsolidationJob],
    ) -> None:
        del state  # unused; single queue, FIFO serialization is enough
        directory = self.memory_directory
        if not directory:
            log.error(
                "memory_consolidation_queue missing memory_directory; dropping job"
            )
            return

        journal = JournalStore(directory)
        watermark = MemoryWatermark.load(directory, run_id=job.run_id)

        # Resolve the unseen span from persistence (or from the trigger's
        # precomputed material). The ADVANCE TARGET — where the owner watermark
        # moves to on a successful fold — is the trigger's resolved watermark
        # when supplied (job.watermark), else the resolver's own
        # messages_span.watermark; NEVER the stale on-disk watermark loaded
        # above (that is only the cadence reference).
        if job.messages is not None:
            messages_span = ResolvedSpan(
                messages=job.messages,
                watermark=watermark,
            )
            advance_target = (
                job.watermark if job.watermark is not None else messages_span.watermark
            )
        else:
            if self.span_source is None:
                log.error(
                    "memory_consolidation_queue has no span_source; cannot fold job "
                    "session=%s run=%s turn=%s — the builder must supply one",
                    session_id,
                    job.run_id,
                    job.turn_id,
                )
                return
            messages_span = await self.span_source(job.run_id, job.turn_id)
            advance_target = messages_span.watermark
        messages = messages_span.messages

        # The reviewer's watermark must exist BEFORE the fold, so the facts this
        # fold writes are reviewed (a first-ever review otherwise starts at the
        # tail and would skip them).
        if self.pairwise_review:
            from ._pairwise import load_watermark, save_watermark

            if load_watermark(directory) is None:
                save_watermark(directory, journal.count())

        # Fold into the journal (append-only; dedup/supersede at render).
        self.current_project = job.project
        if self.consolidate_span is not None:
            new_records = await self.consolidate_span(messages, directory, watermark)
        else:
            new_records = await self._default_consolidate_span(
                messages, directory, watermark, project=job.project
            )

        outcome = FoldOutcome(records=new_records or 0)
        try:
            # THE WATERMARK CONTRACT: advance when the fold COMPLETED WITHOUT
            # ERROR, regardless of how many records it wrote. "Found nothing
            # durable" (new_records == 0) is a successful, complete pass and MUST
            # advance — only a throw / timeout / no-response holds the watermark
            # back so that span is retried next cadence, never silently skipped.
            # The worker is now the SOLE owner-watermark writer (the trigger only
            # schedules; it no longer advances at fire-time).
            watermark.run_id = advance_target.run_id
            watermark.turn_id = advance_target.turn_id
            watermark.consumed_ts = advance_target.consumed_ts
            watermark.save(directory)
            log.info(
                "memory_consolidation_job session=%s success folded=%r "
                "watermark=%s:%s ts=%s",
                session_id,
                new_records,
                watermark.run_id,
                watermark.turn_id,
                watermark.consumed_ts,
            )

            # Re-render MEMORY.md (and the derived index) only when the fold actually
            # wrote new facts — a successful 0-record fold changed nothing to render.
            if new_records and new_records > 0:
                # Materialize MEMORY.md deterministically from the append-only journal —
                # the exact render the eviction path and benchmark produce.
                from ._render import write_render

                write_render(
                    journal,
                    str(Path(directory) / "MEMORY.md"),
                    agent_name=self.agent_name,
                )
                # Derived index of surviving facts (dedup winners, incremental, pure
                # derivation from the journal) — regenerated beside MEMORY.md so they
                # can't disagree. First fold on a journal with pre-existing history
                # builds the full index.
                from ._index import update_index

                update_index(directory)

                if self.pairwise_review:
                    # Keep every current fact unique and up to date. A review
                    # failure never fails the fold (the watermark above already
                    # advanced; the next fold's review picks up from the reviewer's
                    # own watermark).
                    from ._pairwise import review_new_facts

                    await self._notify(self.on_reviewing)
                    try:
                        outcome.review = await review_new_facts(
                            directory, agent_name=self.agent_name
                        )
                        log.info(
                            "pairwise review after fold: %s", outcome.review.summary()
                        )
                    except Exception as exc:  # noqa: BLE001
                        outcome.review_error = str(exc) or type(exc).__name__
                        log.exception("pairwise review after fold failed")
        except Exception as exc:
            outcome.error = str(exc) or type(exc).__name__
            raise
        finally:
            # Always close out the status row, so it can never be left showing
            # "reviewing" after a failure between the fold and the review.
            await self._notify(self.on_finished, outcome)

    async def _notify(self, hook: Callable[..., Awaitable[None]] | None, *args) -> None:
        """Call a status hook; a failing hook never fails the job."""
        if hook is None:
            return
        try:
            await hook(*args)
        except Exception:  # noqa: BLE001
            log.exception("memory consolidation status hook failed")

    async def _default_consolidate_span(
        self,
        messages: list[ModelMessage],
        directory: str,
        watermark: MemoryWatermark,
        project: str | None = None,
    ) -> int:
        """Real fold: run the owner over ``messages`` into the journal.

        Returns the number of journal records ADDED by this fold (a delta, not
        the running total) so the worker can tell whether anything was actually
        folded — the signal it uses to decide whether the watermark may advance.
        """
        from ._consolidate import consolidate
        from ._view import JournalViewStore

        journal = JournalStore(directory)
        before = journal.count()
        await consolidate(
            messages,
            store=JournalViewStore(directory),
            agent_name=self.agent_name,
            namespace="",
            keep_tail=0,
            journal=journal,
            project=project,
            history_source=self.history_source,
        )
        return journal.count() - before
