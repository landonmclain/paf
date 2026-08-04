"""Thin CLI — the M1 user interface (a basic script in its own ui/ folder).

Run a single agent turn end-to-end:
    uv run python -m ui.cli "Say hello to Maya"          # uses .env model
    uv run python -m ui.cli --test "Say hello to Maya"   # keyless 'test' model
"""

from __future__ import annotations

import asyncio
import sys

from paf.main import get_storage, init_runtime
from paf.observability.instrumentation import shutdown_observability


async def run(prompt: str, use_test_model: bool) -> None:
    init_runtime()
    storage = get_storage()

    from paf.agents.example_agent import build_agent

    agent = build_agent(model_name="test" if use_test_model else None)
    result = await agent.run(prompt)
    print(result.output.model_dump_json(indent=2))

    # Exercise the storage seam so inspectable state exists.
    storage.set("last_greeting", result.output.model_dump())
    print("\n[state] wrote data/state/last_greeting.json")

    # Flush + shut down the tracer so the short-lived process's spans persist.
    shutdown_observability()


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    use_test_model = "--test" in sys.argv
    prompt = " ".join(args) or "Say hello to Maya."
    asyncio.run(run(prompt, use_test_model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
