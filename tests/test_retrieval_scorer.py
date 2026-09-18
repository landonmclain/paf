"""Regression tests for the retrieval scorer in ``tests.run_lmem``.

Covers the crash where a NON-string ``gold_answer`` (e.g. a count like ``3``
for a multi-session question) was scored at all. The retrieval metric is now
**ID-membership recall@k** only: it compares the surfaced ``session=<id>`` set
against the gold ``answer_session_ids`` and needs no token-overlap text probe.

The old text-presence fallback (which called ``.strip()`` on a ``gold_answer``
and crashed on an int) was removed when LoCoMo was retired (2026-08). These
tests pin the surviving ID-membership path against regressions like q2's.

See LongMemEval-Testing.md "Memory-consolidation owner A/B".
"""

from tests.run_lmem import _id_membership_recall, _retrieval


def _recall_row(*, gold_answer, answer_session_ids, tool_outputs, text="answer"):
    """Build the minimal ``row`` dict the scorer reads."""
    return {
        "gold_answer": gold_answer,
        "answer_session_ids": answer_session_ids,
        "tool_outputs": tool_outputs,
        "text": text,
    }


def test_int_gold_answer_scored_safely():
    """q2 shape: count gold (int), truthy answer_session_ids, no session= tags.

    The old text-fallback raised ``AttributeError: 'int' object has no
    attribute 'strip'``. With the fallback gone, this must still not raise and
    must report recall 0 (no evidence surfaced).
    """
    row = _recall_row(
        gold_answer=3,  # the count; not a string
        answer_session_ids=["a", "b", "c"],
        tool_outputs=[
            "- Pick up new pair of Zara boots (exchanged 2/5)",
            "- Pick up dry cleaning (navy blue blazer)",
        ],
    )
    # Must not raise.
    result = _retrieval(row)
    assert "retrieval_recall" in result
    assert result["method"] == "id-membership"
    assert result["retrieval_recall"] == 0.0  # no session= tags surfaced


def test_no_surfaced_tags_scores_zero():
    """String gold with no session= tags -> id-membership, recall 0.

    The text-presence fallback that used to match the probe against tool output
    was removed; retrieval is scored purely on surfaced session ids.
    """
    row = _recall_row(
        gold_answer="Zara boots",
        answer_session_ids=["a"],
        tool_outputs=["pick up new pair of Zara boots"],
    )
    result = _id_membership_recall(row, ["a"])
    assert result["method"] == "id-membership"
    assert result["retrieval_recall"] == 0.0


def test_id_membership_path_preferred_when_tags_present():
    """When session= tags ARE surfaced, use ID-membership, not text."""
    row = _recall_row(
        gold_answer="anything",  # text not relevant; ids do the work
        answer_session_ids=["answer_abc_1"],
        tool_outputs=["search_conversation_history -> session=answer_abc_1"],
    )
    result = _id_membership_recall(row, ["answer_abc_1"])
    assert result["method"] == "id-membership"
    assert result["retrieval_recall"] == 1.0
