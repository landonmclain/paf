"""`/resume` argument matching in ui/chat.py."""

from __future__ import annotations

from ui.chat import match_conversation

ROWS = [
    {"conversation_id": "paf-1", "title": "quantconnect"},
    {"conversation_id": "paf-2", "title": "paf security and guardrails"},
    {"conversation_id": "paf-3", "title": "paf code review and cleanup"},
    {"conversation_id": "chat-demo_assistant-ab12", "title": None},
    {"conversation_id": "paf-5", "title": "paf"},
]


def test_number_picks_listed_row() -> None:
    assert match_conversation(ROWS, "2") == ("paf-2", [])
    assert match_conversation(ROWS, "9") == (None, [])


def test_exact_id_wins() -> None:
    assert (
        match_conversation(ROWS, "chat-demo_assistant-ab12")[0]
        == "chat-demo_assistant-ab12"
    )


def test_unique_name_substring_case_insensitive() -> None:
    assert match_conversation(ROWS, "QUANT") == ("paf-1", [])
    assert match_conversation(ROWS, "ab12")[0] == "chat-demo_assistant-ab12"


def test_ambiguous_returns_candidates() -> None:
    cid, hits = match_conversation(ROWS, "and")
    assert cid is None
    assert {r["conversation_id"] for r in hits} == {"paf-2", "paf-3"}


def test_exact_name_breaks_a_tie() -> None:
    # "paf" is a substring of several names, but one conversation is named exactly "paf".
    assert match_conversation(ROWS, "paf")[0] == "paf-5"


def test_no_match() -> None:
    assert match_conversation(ROWS, "staging") == (None, [])
