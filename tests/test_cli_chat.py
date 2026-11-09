"""Tests for the paf chat CLI (paf.cli.chat)."""

from __future__ import annotations

from paf.cli.chat import match_conversation

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
    assert match_conversation(ROWS, "no-such-project") == (None, [])


def test_status_refreshes_coalesce_during_a_burst() -> None:
    """Many refresh requests while a read runs cost one follow-up read, not one each."""
    import asyncio

    from paf.cli.chat import ChatApp

    app = object.__new__(ChatApp)
    app._refresh_task = None
    app._refresh_dirty = False
    reads = 0

    async def slow_read() -> None:
        nonlocal reads
        reads += 1
        await asyncio.sleep(0.05)

    app._read_status = slow_read  # type: ignore[method-assign]

    async def scenario() -> None:
        first = asyncio.create_task(app.refresh_status())
        await asyncio.sleep(0.01)  # first read is running
        for _ in range(10):  # a burst of background events
            app._request_refresh()
        await app.refresh_status()  # an awaiting caller sees the follow-up read
        await first

    asyncio.run(scenario())
    assert reads == 2


def test_missing_api_key_is_one_line_not_a_traceback():
    from pydantic_ai.exceptions import UserError

    from paf.cli.chat import _user_error_message

    msg = _user_error_message(
        UserError(
            "Set the `OPENAI_API_KEY` environment variable or pass it via "
            "`OpenAIProvider(api_key=...)` to use the OpenAI provider."
        )
    )
    assert msg.startswith("paf chat: OPENAI_API_KEY is not set")
    assert ".env" in msg and "\n" not in msg
    assert _user_error_message(UserError("Unknown model: x. More text.")) == (
        "paf chat: Unknown model: x."
    )


def test_projects_root_command(tmp_path, monkeypatch):
    from paf.capabilities._storage import FlatFileStorage
    from paf.cli import chat
    from paf.conversations import ProjectStore

    app = object.__new__(chat.ChatApp)
    app.projects = ProjectStore(FlatFileStorage(tmp_path / "registry"))
    app.projects.create("example")
    output = []
    monkeypatch.setattr(chat, "_out", output.append)
    root = tmp_path / "directory with spaces"
    root.mkdir()
    app._projects_command(f"root example {root}")
    assert app.projects.get("example")["root"] == str(root)
    app._projects_command("root example missing")
    assert "absolute path" in output[-1]
    app._projects_command("root example -")
    assert "root" not in app.projects.get("example")
    app._projects_command("root unknown -")
    assert "unknown project" in output[-1]
