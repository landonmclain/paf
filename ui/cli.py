"""Real CLI REPL on top of :class:`paf.runtime.Session`.

The production-shaped loop: one ``Session`` built from ``specs/assistant.yaml``
(which already wires ``StepPersistence``, ``ConversationSearch``, and the
``TieredCompaction`` ladder), a stable ``conversation_id`` for durable cross-turn
history, and a live renderer over :meth:`Session.stream`'s ``StreamEvent``s.

``Session`` owns everything ``cli_demo.py`` hand-rolled — tracing setup/teardown,
the build, message-history accumulation per turn, and the ``run_stream_events``
driver. This file is deliberately thin: it maps events to the console.

Usage:
    python ui/cli.py
    # defaults to conversation_id "cli-primary"

Commands:
    quit / exit / /q   exit

Needs OPENAI_API_KEY (live nano model). Run twice with the same
conversation_id to see persistence + recall: start a session, quit, restart,
then ask for something only reachable via search_conversation_history.
"""

from __future__ import annotations

import asyncio
import os
import sys

from paf.runtime import Session, StreamEvent

CONVERSATION_ID = "cli-primary"

# ── render: map StreamEvent -> console --------------------------------------
# StreamEvent(kind=...) -> what to draw. Text streams in place (no newline until
# a tool banner or the answer ends); thinking is suppressed to stderr; tools are
# interleaved as lines so they never collide with the streaming answer.


def _safe(s: str) -> str:
    """Make a model-produced string safe to print under utf-8.

    ``FunctionToolCallEvent.args`` can carry lone UTF-16 surrogate code units
    (the model streaming a partially-decoded arg), which ``print`` then blows up
    on with ``UnicodeEncodeError`` — crashing the whole REPL mid-turn. Drop any
    unencodable characters rather than raise.
    """
    return s.encode("utf-8", errors="replace").decode("utf-8")


def render(event: StreamEvent) -> None:
    """Draw one StreamEvent to the terminal, tracking the streaming answer."""
    if event.kind == "text":
        if event.text:
            print(event.text, end="", flush=True)
    elif event.kind == "thinking":
        # Reasoning tokens: quiet by default. Uncomment to show/annotate.
        pass
    elif event.kind == "tool_call":
        sys.stdout.write("\n")  # end any in-flight streaming text
        print(f"\U0001f9ed tool \u2192 {event.tool_name}({_safe(event.args)})")
    elif event.kind == "tool_result":
        print(f"\u2705 \u2192 {_safe(event.text)}")
    # Unknown kinds: ignore.


def _startup_banner(session: Session) -> None:
    """Show whether this conversation resumes durable history.

    Session.restore() seeds self.history from the shared store (continue_run).
    Returns the count of prior messages loaded -- 0 = a fresh conversation.
    """
    try:
        restored = asyncio.run(session.restore())
    except Exception as exc:  # noqa: BLE001 - persistence is optional sugar here
        print(f"(persistence/resume skipped: {exc})")
        return
    if restored:
        print(f"[resumed: {restored} prior message(s)]")
    else:
        print(f"[new conversation: {session.conversation_id}]")


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY not set. Provide it to call the live model:\n"
            "  export OPENAI_API_KEY=sk-..."
        )
        return 2

    with Session("assistant", conversation_id=CONVERSATION_ID) as session:
        _startup_banner(session)
        print('type a message, or "quit" to exit')

        while True:
            try:
                prompt = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                return 0

            if not prompt:
                continue
            if prompt.lower() in {"quit", "exit", "/q"}:
                print("bye")
                return 0

            print("bot> ", end="", flush=True)
            # stream() accumulates history + land last_result itself; we only
            # render.
            asyncio.run(_consume(session, prompt))
            print()  # trailing newline after the streamed answer

    return 0


async def _consume(session: Session, prompt: str) -> None:
    """Drive one streamed turn to completion, rendering each event."""
    async for event in session.stream(prompt, conversation_id=session.conversation_id):
        render(event)


if __name__ == "__main__":
    raise SystemExit(main())
