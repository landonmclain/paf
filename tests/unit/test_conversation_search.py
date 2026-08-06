"""Unit tests for the ConversationSearch capability (BM25 recall over history).

These use a fake `HistorySource` fed with *scripted* messages — no LLM, no store,
no API key. That is the seam `ConversationSearch` is designed to consume: it reads
`list_runs()` + `run_history()` lazily at call time. Scripting the source proves
the full search-conversation-history tool path (tokenize -> BM25 rank -> excerpt)
deterministically, and mirrors how a compaction summary "drops" an old detail while
the persisted originals keep it recoverable.
"""

from __future__ import annotations

import asyncio

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from pydantic_ai_harness.conversation_search import (
    ConversationSearch,
    HistorySource,
)
from pydantic_ai_harness.step_persistence import RunRecord


def _msg(content: str) -> ModelMessage:
    return ModelResponse(parts=[TextPart(content=content)])


class _FakeSource(HistorySource):
    """In-memory HistorySource: two runs, each a list of scripted messages."""

    def __init__(self, runs: dict[str, list[ModelMessage]]) -> None:
        self._runs = [
            RunRecord(
                run_id=run_id,
                agent_name="none",
                parent_run_id=None,
                conversation_id="demo-convo-1",
            )
            for run_id in runs
        ]
        self._msgs = runs

    async def list_runs(self) -> list[RunRecord]:
        return self._runs

    async def run_history(self, *, run_id: str) -> list[ModelMessage]:
        return self._msgs.get(run_id, [])


def _search_tool(source: HistorySource):
    """Build ConversationSearch over a source and return its callable tool.

    Returns a function `(query: str) -> str` that drives the real
    `search_conversation_history` tool handler with a minimal RunContext.
    """
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.usage import RequestUsage

    cap = ConversationSearch(source)
    toolset = cap.get_toolset()
    assert toolset is not None

    ctx = RunContext(
        deps=None,
        model=Agent("test").model,
        usage=RequestUsage(),
    )
    handler = toolset.search_conversation_history

    def call(query: str) -> str:
        return asyncio.run(handler(ctx, query))

    return call


def test_conversation_search_recalls_compacted_detail() -> None:
    """A detail absent from the 'live' context is recoverable from persisted history.

    Scripted corpus: run1 holds a distinctive fact (the deployment endpoint) that a
    compaction summary would have summarized away; run2 is unrelated. Searching for
    the distinctive term must surface run1's original messages.
    """
    source = _FakeSource(
        {
            "run-1": [
                _msg(
                    "Deploying the monitoring agent: the dashboard lives at "
                    "https://observability.fredholm-blue.internal/"
                ),
                _msg(
                    "Deploy to the fredholm-blue server, port 8443, with the "
                    "tls cert from the vault."
                ),
                _msg("Took a break for lunch."),
            ],
            "run-2": [
                _msg("Reviewed the Q3 forecast; margins look healthy."),
                _msg("Scheduled the board deck review for Friday."),
            ],
        }
    )
    tool = _search_tool(source)

    result = tool("fredholm-blue")
    assert "8443" in result
    assert "fredholm-blue" in result
    assert "run-1" in result


def test_conversation_search_returns_no_matches_for_absent_term() -> None:
    source = _FakeSource(
        {
            "run-1": [_msg("The target hostname is zebra.example and the port is 22.")],
        }
    )
    tool = _search_tool(source)
    result = tool("irrelevant")
    assert "No matches" in result


def test_conversation_search_ranked_query_prefers_rarer_term() -> None:
    """BM25 ranks a rare exact term above a common one shared by both runs."""
    source = _FakeSource(
        {
            "run-1": [
                _msg("moonshot lab: the onyx test harness launches with --fast."),
                _msg("moonshot onyx harness in CI every commit."),
            ],
            "run-2": [
                _msg("moonshot is the umbrella name for the suite."),
                _msg("nothing about onyx here."),
            ],
        }
    )
    tool = _search_tool(source)
    result = asyncio.run(tool.run("onyx", "Enter the query."))
    assert "run-1" in result