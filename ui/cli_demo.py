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

We add: a conversation_id, a loop, ~4 lines to reload history on restart, and a
tiny renderer so restarting actually replays the prior transcript (readable).

Needs an OPENAI_API_KEY (live nano model). Run twice to see persistence:
  1)  export PYTHONPATH="" && rm -f ui/demo.db && .venv/bin/python ui/cli_demo.py
        -> no prior transcript, send a few turns, quit
  2)  .venv/bin/python ui/cli_demo.py
        -> replays the saved transcript, then a follow-up needing prior context
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
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


def _part_text(part) -> str:
    """Pull the human text out of a str|list[str] content field."""
    if isinstance(part.content, str):
        return part.content
    return " ".join(str(chunk) for chunk in part.content)


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
        capabilities=[StepPersistence(store=store, agent_name="demo")],
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
            result = await agent.run(
                prompt, message_history=history, conversation_id=CONVERSATION_ID
            )
            print(f"\nbot> {result.output}\n")
            # ACCUMULATE, don't replace: result.new_messages() returns only THIS
            # turn's messages (message_history + older runs are excluded), so
            # overwriting `history` with it would drop the prior session and let
            # the model lose context after turn 1. Prepend what we fed in, then
            # append the fresh turn so the history stays complete across turns.
            history = [*history, *result.new_messages()]
    except (EOFError, KeyboardInterrupt):
        pass

    print(f"== session persisted to {DB} — restart to continue on the same dialog ==")


if __name__ == "__main__":
    asyncio.run(main())
