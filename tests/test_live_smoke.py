"""Live smoke test: run the real one-turn agent against the model.

Marked `live` and excluded from the default `pytest` run (see pyproject
`addopts = -m 'not live'`). Runs only when invoked explicitly with
`pytest -m live`, which requires `.env` with a working model key.

This is a regression gate for the README "it works" path: build the
assistant from spec, run one real turn, assert a response came back.
It costs tokens and hits the network — that's why it's opt-in.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from paf import main_simple

pytestmark = pytest.mark.live


def test_main_simple_runs_one_turn_end_to_end() -> None:
    """main_simple.main() builds the agent, runs a turn, and prints output."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            main_simple.main()
    finally:
        # main() flushes observability; ensure it always happens even on failure.
        from paf.observability.instrumentation import shutdown_observability

        shutdown_observability()

    out = buf.getvalue()
    assert "Agent: assistant" in out
    # The response itself — must be a non-empty model reply (not just the header).
    body = out.split("Agent: assistant", 1)[-1].strip()
    assert body, "expected a non-empty model response"
