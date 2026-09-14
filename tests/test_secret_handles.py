"""Secrets: secret-handle resolution + result redaction.

The harness ``ToolGuardrail`` is the shell; paf owns the policy
(``secret_handle_guard`` / ``secret_result_guard``). The model proposes a
handle (``$SECRET:key``); the runtime resolves it to the real value at
execution time, so the real value never enters the window. These tests prove
the handle resolves, a missing secret retries, a leaked secret is redacted,
and the capability is wired into the demo profile.
"""

from pydantic_ai_harness import ToolGuardrail
from pydantic_ai_harness.guardrails import ToolCallInfo, ToolResultInfo

from paf.capabilities.build import _build_secrets
from paf.capabilities.guardrails.secret_handles import (
    secret_handle_guard,
    secret_result_guard,
)
from paf.capabilities.profiles import DEMO_PROFILE
from paf.capabilities.registry import CAPABILITY_TYPES

REAL = "sk_live_1234567890abcdef"


def _call(name: str = "call_api", **args) -> ToolCallInfo:
    return ToolCallInfo(name=name, args=args, tool_call_id="1")


def test_handle_resolves_at_execution(monkeypatch):
    """A $SECRET:stripe_key arg is replaced with the real value."""
    monkeypatch.setenv("stripe_key", REAL)
    verdict = secret_handle_guard(_call(api_key="$SECRET:stripe_key"))
    assert verdict.action == "replace"
    assert verdict.replacement["api_key"] == REAL


def test_plain_args_pass_through(monkeypatch):
    """Args with no handle are allowed unchanged."""
    monkeypatch.setenv("stripe_key", REAL)
    verdict = secret_handle_guard(_call(api_key="sk_plain"))
    assert verdict.action == "allow"


def test_missing_secret_retries(monkeypatch):
    """A handle naming an unavailable secret retries with a message."""
    monkeypatch.delenv("stripe_key", raising=False)
    verdict = secret_handle_guard(_call(api_key="$SECRET:stripe_key"))
    assert verdict.action == "retry"
    assert "stripe_key" in verdict.message


def test_result_redaction(monkeypatch):
    """A raw secret leaked into a tool result is replaced with its handle."""
    monkeypatch.setenv("stripe_key", REAL)
    info = ToolResultInfo(
        name="call_api", args={}, tool_call_id="1", result=f"here is {REAL}"
    )
    verdict = secret_result_guard(info)
    assert verdict.action == "replace"
    assert REAL not in verdict.replacement
    assert "$SECRET:stripe_key" in verdict.replacement


def test_result_short_values_not_redacted(monkeypatch):
    """Short env values are not treated as secrets (no false positives)."""
    monkeypatch.setenv("stripe_key", "abc")
    info = ToolResultInfo(
        name="call_api", args={}, tool_call_id="1", result="the value is abc"
    )
    verdict = secret_result_guard(info)
    assert verdict.action == "allow"


def test_wired_into_demo_profile():
    """Secrets is registered and allowlisted on the demo profile."""
    assert "Secrets" in DEMO_PROFILE.capability_names
    assert CAPABILITY_TYPES["Secrets"] is ToolGuardrail
    cap = _build_secrets(store=None)
    assert isinstance(cap, ToolGuardrail)
