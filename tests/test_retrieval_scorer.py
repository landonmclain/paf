"""Regression tests for the retrieval scorer in ``tests.run_lmem``.

Covers the crash where a NON-string ``gold_answer`` (e.g. a count like ``3``
for a multi-session question) with no ``session=`` tags in the tool outputs
fell into the text-fallback branch and called ``.strip()`` on an ``int``
(``AttributeError: 'int' object has no attribute 'strip'``).

See LongMemEval-Testing.md "Memory-consolidation owner A/B" — q2's official
grade was lost to this crash even though the memory write succeeded.
"""

import pytest

from tests.run_lmem import _id_membership_recall, _retrieval


def _recall_row(*, gold_answer, answer_session_ids, tool_outputs, text="answer"):
    """Build the minimal ``row`` dict the scorer reads."""
    return {
        "gold_answer": gold_answer,
        "answer_session_ids": answer_session_ids,
        "tool_outputs": tool_outputs,
        "text": text,
    }


def test_int_gold_answer_with_no_session_tags_does_not_crash():
    """q2 shape: count gold (int), truthy answer_session_ids, but no session= tags.

    This used to raise ``AttributeError: 'int' object has no attribute 'strip'``
    in the text-fallback branch inside ``_id_membership_recall``.
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
    result = _retrieval(row, [])  # gold_session_texts ignored: answer_session_ids present
    assert "retrieval_recall" in result
    assert result["method"] in ("id-membership", "text-fallback")


def test_string_gold_answer_no_tags_uses_text_fallback():
    """String gold with no session= tags matches via text-presence fallback."""
    row = _recall_row(
        gold_answer="Zara boots",
        answer_session_ids=["a"],
        tool_outputs=["pick up new pair of Zara boots"],
    )
    result = _id_membership_recall(row, ["a"])  # id-membership path (gold ids present)
    # surfaced empty -> falls to text fallback; the probe "Zara boots" appears
    assert result["method"] == "text-fallback"


def test_id_membership_path_preferred_when_tags_present():
    """When session= tags ARE surfaced, use ID-membership, not text."""
    row = _recall_row(
        gold_answer="anything",  # text not relevant; ids do the work
        answer_session_ids=["answer_abc_1"],
        tool_outputs=["search_index -> session=answer_abc_1"],
    )
    result = _id_membership_recall(row, ["answer_abc_1"])
    assert result["method"] == "id-membership"
    assert result["retrieval_recall"] == 1.0