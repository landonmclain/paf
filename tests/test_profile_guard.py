"""Guard: no test may depend on the live/production profiles.

Tests must use ``TEST_PROFILE`` (tests/framework.py) or a test-local
``AgentProfile``, never the production allowlists (assistant/demo/contextbench),
which change as capabilities are promoted. This guard fails if any collected
test file references a live profile constant or constructs a ``Session`` with a
live agent name — except ``test_capabilities.py``, the designated home for
production-profile registry tests.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

# Live profile constants and live Session agent names that tests must not use.
_LIVE_PROFILE_NAMES = (
    "ASSISTANT_PROFILE",
    "DEMO_PROFILE",
    "CONTEXTBENCH_PROFILE",
)
_LIVE_AGENT_NAMES = ("assistant", "demo_assistant", "contextbench")

# test_capabilities.py is the designated home for production-profile registry
# tests (the PROFILES mapping and no-dead-refs integrity). test_profile_guard.py
# is this guard itself, which must name the live constants to scan for. Every
# other test file must be decoupled from the live profiles.
_ALLOWED_FILES = {"test_capabilities.py", "test_profile_guard.py"}


def _collected_test_files() -> list[Path]:
    """Top-level test modules (retired/ and fixtures/ are not collected)."""
    return sorted(p for p in TESTS.glob("test_*.py"))


def test_no_test_depends_on_live_profiles() -> None:
    offenders: list[str] = []
    for path in _collected_test_files():
        if path.name in _ALLOWED_FILES:
            continue
        src = path.read_text()
        for name in _LIVE_PROFILE_NAMES:
            if name in src:
                offenders.append(f"{path.name}: references {name}")
        for agent in _LIVE_AGENT_NAMES:
            if f'Session("{agent}"' in src:
                offenders.append(f"{path.name}: Session('{agent}')")
    assert not offenders, (
        "Tests must not depend on the live/production profiles — use "
        "TEST_PROFILE (tests/framework.py) or a test-local AgentProfile. "
        "Found:\n" + "\n".join(offenders)
    )
