"""One-file demo: pydantic-ai agent + message history + durable storage.

Lean portfolio-style proof that pydantic-ai (via the harness `StepPersistence`
capability) already ships the storage layer — we embrace it rather than rebuild
it. This is a tiny CLI loop; the only "storage code" is how we find the prior
run and hand its history back to the agent.

The pydantic-native pieces doing the work:
  - `StepPersistence(store=SqliteStepStore(database=...))` — a *capability* that
    snapshots run state, appends a step event log, and externalizes big media,
    all to durable storage every settled step + at run end.
  - `conversation_id=` — pydantic's primitive for grouping many turns into one
    dialogue (the storage key), passed to *every* Agent.run() call.
  - `store.list_runs(conversation_id=...)` — enumerate that dialogue's runs.
  - `continue_run(store, run_id=...) -> list[ModelMessage]` — load the latest
    settled snapshot as a fresh history.
  - `Agent.run(message_history=...)` — feed that history back in to continue.
  - `TieredCompaction` / `compact_now` — pydantic-ai-harness's context
    management: cheap passes first (rolling window), summarize only if still
    over budget; `compact_now` drives the same strategy from a `/compact`
    command between runs.
  - `ReportContextUsage` — a no-token gauge so the UI can show `context: NN%`.
  - `ConversationSearch(source=SnapshotHistorySource(store))` — a *capability*
    that gives the model a `search_conversation_history` tool (BM25) over the
    same store `StepPersistence` writes to. Because persistence keeps the full
    snapshots, the model can recall anything compaction (sliding window or
    summary) dropped from the live context, plus anything from past runs — the
    lossless-recall leg of the loop.

We add: a conversation_id, a loop, ~4 lines to reload history on restart, a
tiny renderer to replay the prior transcript, and the compaction wiring.

Needs an OPENAI_API_KEY (live nano model). Run twice to see persistence:
  1)  export PYTHONPATH="" && rm -f ui/demo.db && .venv/bin/python ui/cli_demo.py
        -> no prior transcript, send a few turns, quit
  2)  .venv/bin/python ui/cli_demo.py
        -> replays the saved transcript, then a follow-up needing prior context

Try `/compact` mid-session: it collapses the oldest turns into a summary and
prints the before/after message + token counts.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ThinkingPartDelta,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai_harness.compaction import (
    ReportContextUsage,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
    compact_now,
    estimate_token_count,
)
from pydantic_ai_harness.conversation_search import (
    ConversationSearch,
    SnapshotHistorySource,
)
from pydantic_ai_harness.step_persistence import (
    SqliteStepStore,
    StepPersistence,
    continue_run,
)

CONVERSATION_ID = "demo-convo-1"
MODEL = (
    "openai:gpt-5.4-nano"  # $0.10/1M input, $0.625/1M output (cheapest capable nano)
)
DB = Path(__file__).resolve().parent / "demo.db"

# Controls the LENGTH of the summarizing-tier summary. The harness ships a
# structured default (Intent / Key decisions / Artifacts / Current state / Next
# steps / Open questions) but has NO hard token cap on the summary itself — the
# model sizes it from this prompt. Override it if you want a bounded summary.
SUMMARY_PROMPT = (
    "You are a context summarization assistant. The conversation below will be "
    "replaced by your summary, so it must carry everything needed to continue "
    "the task.\n\n"
    "Write a single summary under these section headings, omitting a section "
    "only if it has no content:\n"
    "## Intent\n"
    "## Key decisions\n"
    "## Artifacts\n"
    "## Current state\n"
    "## Next steps\n\n"
    "KEEP THE ENTIRE SUMMARY UNDER 300 TOKENS. Do not replay completed actions; "
    "state results and what still matters. Respond ONLY with the summary — no "
    "preamble, no markdown fences.\n\n"
    "<messages>\n{messages}\n</messages>"
)

# Context-management policy. Production would use a high budget (e.g.
# target_fraction=0.75 of the 400K window ~ 300K tokens) and larger windows.
# For a *demo* we keep everything small so a handful of turns trips the
# escalation and you can watch each tier compress the history in real time.
# target_tokens caps the total context after compaction at ~1.5k tokens.
COMPACTION = TieredCompaction(
    tiers=[
        # Keep the budget TINY so a handful of turns trips the escalation and
        # pushes early details out of the live context fast. With only the last
        # 2 messages kept, anything older must be recovered via
        # search_conversation_history — that's the wiring we're demoing.
        SlidingWindowCompaction(max_messages=4, keep_messages=2),
        SummarizingCompaction(
            max_messages=2, keep_messages=2, summary_prompt=SUMMARY_PROMPT
        ),
    ],
    target_tokens=1500,  # cap total context at ~1.5k tokens for the demo
)

context_usage: list[str] = []


def on_context_usage(usage) -> None:
    """Show live context usage after compaction; remember the last reading."""
    pct = usage.fraction * 100
    note = "" if usage.resolved else " (window unresolved — guess)"
    context_usage.append(f"context: {pct:.0f}%{note}")


def _part_text(part) -> str:
    """Pull the human text out of a str|list[str] content field."""
    if isinstance(part.content, str):
        return part.content
    return " ".join(str(chunk) for chunk in part.content)


def _stream_part_content(part) -> str:
    """Dig the text out of a TextPart/ThinkingPart in a start event."""
    content = getattr(part, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return " ".join(str(chunk) for chunk in content)


def render_history(history: list[ModelMessage]) -> None:
    """Replay the persisted transcript as you>/bot> lines, blank-line separated."""
    print("== prior transcript ==")
    for msg in history:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    print(f"you> {_part_text(part)}\n")
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    print(f"bot> {_part_text(part)}\n")
    print("== end prior transcript ==")


# --- live streaming: token-stream output + tool call/results ------------------
# Driven by agent.run_stream_events(...) per the official docs
# (https://pydantic.dev/docs/ai/examples/streaming/#streaming-all-events).
# The raw event stream gives us BOTH:
#   - live deltas (TextPartDelta renders token-by-token; ToolCallPartDelta is
#     the args being constructed)  -> the streaming experience, and
#   - the completed run (AgentRunResultEvent.result is an AgentRunResult with
#     .output and .new_messages()) -> the authoritative transcript to persist.
# We deliberately ignore ToolCallPartDelta (args already print in full on the
# FunctionToolCallEvent line), and we stream TextPartDelta live to the console,
# using FunctionToolCallEvent/FunctionToolResultEvent to interleave tool IO.

_SESSION = {"streaming": False}  # True while a bot text part is mid-stream


def _ensure_newline() -> None:
    """End the current live text line so a tool banner starts fresh."""
    if _SESSION["streaming"]:
        print("", flush=True)
        _SESSION["streaming"] = False


async def main() -> None:
    # DB is anchored to the script's dir (not the shell cwd) so persistence lands
    # in the same predictable spot no matter where you invoke the script from.
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not set. Provide it to call the live nano model:\n"
            f"  export OPENAI_API_KEY=sk-...   (or use MODEL={MODEL!r})"
        )
    store = SqliteStepStore(
        database=DB
    )  # swappable: FileStepStore or InMemoryStepStore

    agent = Agent(
        MODEL,  # live nano model, needs OPENAI_API_KEY
        capabilities=[
            StepPersistence(store=store, agent_name="demo"),
            # Lossless recall: BM25 search over the same store StepPersistence
            # writes to. Lets the model retrieve anything compaction (sliding
            # window or summary) dropped from the live context, plus past runs.
            ConversationSearch(source=SnapshotHistorySource(store)),
            COMPACTION,
            ReportContextUsage(on_usage=on_context_usage),
        ],
    )

    # Durable resume: find this dialogue's runs in storage, load the latest
    # settled snapshot as history. Fresh conversation -> EmptyError -> start new.
    runs = await store.list_runs(conversation_id=CONVERSATION_ID)
    if runs:
        history = await continue_run(store, run_id=runs[-1].run_id)
        restored = True
    else:
        history = []
        restored = False
    print(f"== restored-from-storage: {restored} | {len(history)} messages prior ==")
    if history:
        render_history(history)

    try:
        while True:
            prompt = input("you> ").strip()
            if prompt.lower() in {"quit", "exit", "/q"}:
                break
            # Manual compaction: collapse the oldest turns into a summary using
            # the same strategy the agent uses automatically, but driven here
            # (compact_now builds a throwaway RunContext between runs).
            if prompt.lower() == "/compact":
                before_msgs, before_tokens = len(history), estimate_token_count(history)
                history = await compact_now(
                    COMPACTION, history, model=MODEL, focus="maintain task continuity"
                )
                after_msgs, after_tokens = len(history), estimate_token_count(history)
                print(
                    f"\n[compact] {before_msgs}->{after_msgs} msgs, "
                    f"{before_tokens}->{after_tokens} est. tokens\n"
                )
                continue

            # Stream the run live via agent.run_stream_events: token-by-token
            # text (TextPartDelta), tool IO (FunctionToolCall/Result), and the
            # completed AgentRunResult (result.output + .new_messages()) so we
            # still persist the authoritative transcript.
            result = None
            print("bot> ", end="", flush=True)
            async with agent.run_stream_events(
                prompt,
                message_history=history,
                conversation_id=CONVERSATION_ID,
            ) as event_stream:
                async for event in event_stream:
                    if isinstance(event, PartStartEvent):
                        # The streaming framework puts the FIRST chunk of a part
                        # in PartStartEvent.part.content (deltas carry the rest),
                        # so emit it here or the leading text is silently dropped.
                        _SESSION["streaming"] = True
                        print(_stream_part_content(event.part), end="", flush=True)
                    elif isinstance(event, PartDeltaEvent):
                        delta = event.delta
                        if isinstance(delta, TextPartDelta):
                            print(delta.content_delta, end="", flush=True)
                        elif isinstance(delta, ThinkingPartDelta):
                            pass  # reasoning tokens: not shown in this demo
                        # ToolCallPartDelta ignored -- full args on the call line
                    elif isinstance(event, FunctionToolCallEvent):
                        _ensure_newline()
                        print(f"🛠️ tool → {event.part.tool_name}({event.part.args})")
                    elif isinstance(event, FunctionToolResultEvent):
                        print(f"✅ → {event.part.content}")
                    elif isinstance(event, AgentRunResultEvent):
                        result = event.result  # authoritative full transcript
            _ensure_newline()
            print()
            # AgentRunResultEvent always fires when the graph settles, so `result`
            # is set here -- but guard for static checkers / a hypothetical early
            # stream teardown.
            assert result is not None, (
                "run_stream_events aborted before AgentRunResultEvent"
            )
            # ACKNOWLEDGE: result.new_messages() returns only THIS turn's messages
            # (message_history + older runs are excluded), so overwriting history
            # with it would drop the prior session. Accumulate instead.
            history = [*history, *result.new_messages()]
    except (EOFError, KeyboardInterrupt):
        pass

    if context_usage:
        print("== context usage trace ==")
        for line in context_usage:
            print(f"   {line}")
    print(f"== session persisted to {DB} — restart to continue on the same dialog ==")


if __name__ == "__main__":
    asyncio.run(main())
