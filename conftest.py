"""Root pytest configuration: load the project .env into os.environ BEFORE collection.

Without this, bare `pytest` (no 'uv run --env-file .env') drops OPENROUTER_API_KEY
from os.environ, and every test that builds/runs a live agent fails with
"Set the OPENROUTER_API_KEY environment variable" even though .env exists -- OR
worse, tests marked `skipif(os.getenv('OPENROUTER_API_KEY') is None)` silently
skip, hiding the coverage.

Key detail: this MUST run at IMPORT time of this conftest, NOT in a session
fixture. pytest imports test modules (evaluating top-level `skipif` markers)
before any fixture runs, so a fixture-based load would be too late and the
key-dependent tests would still skip/fail.

paf.config loads .env via pydantic-settings `env_file=` but that only populates
*settings fields* -- it does NOT copy vars into os.environ. Tests / providers
read os.getenv('OPENROUTER_API_KEY'), so we make the real intent explicit here:
the project's canonical env file is always loaded into the process env for tests.

This makes `pytest -q` behave identically to `uv run --env-file .env pytest`.
"""

from __future__ import annotations

from pathlib import Path

env_path = Path(__file__).resolve().parent / ".env"
if env_path.is_file():
    from dotenv import load_dotenv

    # override=False: don't clobber a key already set in the shell/CI env.
    load_dotenv(env_path, override=False)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _placeholder_provider_keys(monkeypatch):
    """Tests that build a real agent (without calling a model) need a key to exist.

    Set placeholders only when none is configured, at test time rather than import
    time, so `skipif(os.getenv(...) is None)` markers still see the real environment.
    CI has no keys; a developer's `.env` supplies real ones and wins.
    """
    import os

    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(name):
            monkeypatch.setenv(name, "test-placeholder")
