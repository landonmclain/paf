"""Secrets: secret-handle resolution + result redaction.

The harness ``ToolGuardrail`` is the shell; paf owns the policy
(``secret_handle_guard`` / ``secret_result_guard``). The model proposes a
handle (``$SECRET:key``); the runtime resolves it to the real value at
execution time, so the real value never enters the window. These tests prove
the handle resolves, a missing secret retries, a leaked secret is redacted,
and the capability is wired into the demo profile.
"""

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import ToolGuardrail
from pydantic_ai_harness.guardrails import ToolCallInfo, ToolResultInfo

from paf.capabilities.build import _build_secrets
from paf.capabilities.guardrails.secret_handles import (
    secret_handle_guard,
    secret_result_guard,
)
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


def test_wired_when_allowlisted():
    """Secrets is registered and builds when a profile allowlists it."""
    from paf.capabilities import AgentProfile

    on = AgentProfile(name="secrets-on", capability_names=("Secrets",))
    assert "Secrets" in on.capability_names
    assert CAPABILITY_TYPES["Secrets"] is ToolGuardrail
    cap = _build_secrets(store=None)
    assert isinstance(cap, ToolGuardrail)


class _SecretProposingModel(TestModel):
    """TestModel that proposes the handle as the api_key arg."""

    def gen_tool_args(self, tool_def):
        return {"api_key": "$SECRET:stripe_key"}


def test_integration_tool_gets_real_value_window_never_sees_it(monkeypatch):
    """End-to-end: the tool executes with the real value; the window never sees it.

    The model proposes ``$SECRET:stripe_key``; the runtime resolves it to the
    real value at execution time. The tool receives the real value, and the real
    value never appears anywhere in the run's messages.
    """
    monkeypatch.setenv("stripe_key", REAL)
    seen = {}

    async def call_api(api_key: str) -> str:
        seen["api_key"] = api_key
        return "ok"

    agent = Agent(
        _SecretProposingModel(call_tools=["call_api"]),
        capabilities=[
            ToolGuardrail(guard=secret_handle_guard, result_guard=secret_result_guard)
        ],
        tools=[call_api],
    )
    result = agent.run_sync("call the api")

    # The tool executed with the real value, not the handle.
    assert seen["api_key"] == REAL
    # The model proposed the handle, not the real value.
    assert "$SECRET:stripe_key" in str(result.all_messages())
    # The real value never entered the window.
    assert REAL not in str(result.all_messages())


# --- result redaction: credentials only, never plain configuration --------------
from paf.capabilities.guardrails.secret_handles import (  # noqa: E402
    make_secret_result_guard,
)


def _result(text: str) -> ToolResultInfo:
    return ToolResultInfo(name="t", args={}, tool_call_id="1", result=text)


def test_ordinary_config_values_are_not_redacted(monkeypatch):
    """The trace bug: HOME ("/home/alice") was scrubbed to "$SECRET:HOME"."""
    for name, value in {
        "HOME": "/home/alice",
        "PWD": "/home/alice/projects/paf",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "SHELL": "/bin/bash",
        "USER": "alice",
        "LANG": "en_US.UTF-8",
        "TERM": "xterm-256color",
        "SSH_AUTH_SOCK": "/run/user/1000/keyring/ssh",
        "XAUTHORITY": "/run/user/1000/.mutter-Xwaylandauth.ABC",
        "CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH": "true",
    }.items():
        monkeypatch.setenv(name, value)
    text = (
        "/home/alice -> ERROR: FileNotFoundError /home/alice/projects/paf "
        "en_US.UTF-8 alice xterm-256color /usr/local/bin:/usr/bin:/bin"
    )
    for guard in (make_secret_result_guard(), make_secret_result_guard([])):
        assert guard(_result(text)).action == "allow"


def test_credential_named_variables_are_redacted(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-abcdef0123456789")
    monkeypatch.setenv("GITHUB_TOKEN", "tok_0123456789abcdef")
    monkeypatch.setenv("DB_PASSWORD", "correct horse battery")
    out = make_secret_result_guard()(
        _result(
            "k=sk-or-v1-abcdef0123456789 t=tok_0123456789abcdef p=correct horse battery"
        )
    ).replacement
    assert out == (
        "k=$SECRET:OPENROUTER_API_KEY t=$SECRET:GITHUB_TOKEN p=$SECRET:DB_PASSWORD"
    )


def test_credential_shaped_values_are_redacted_under_any_name(monkeypatch):
    monkeypatch.setenv("MYSTERY_VAR", "ghp_0123456789abcdefghijkl")
    monkeypatch.setenv("OTHER_THING", "AKIAABCDEFGHIJKLMNOP")
    out = make_secret_result_guard()(
        _result("a=ghp_0123456789abcdefghijkl b=AKIAABCDEFGHIJKLMNOP")
    ).replacement
    assert "ghp_" not in out and "AKIA" not in out


def test_path_valued_credential_named_variables_are_left_alone(monkeypatch):
    """A key FILE path is configuration, not the key."""
    monkeypatch.setenv("SSH_KEY_FILE", "/home/alice/.ssh/id_ed25519")
    guard = make_secret_result_guard()
    assert guard(_result("using /home/alice/.ssh/id_ed25519")).action == "allow"


def test_with_an_allowlist_only_listed_names_get_a_handle(monkeypatch):
    """Unlisted credentials become "[redacted]": no variable name, no handle."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-abcdef0123456789")
    monkeypatch.setenv("BILLING_REF", "acct-98765-secret-ref")  # innocuous name
    guard = make_secret_result_guard(["BILLING_REF"])
    out = guard(_result("a=sk-or-v1-abcdef0123456789 b=acct-98765-secret-ref"))
    assert out.replacement == "a=[redacted] b=$SECRET:BILLING_REF"
    assert "OPENROUTER" not in out.replacement


def test_empty_allowlist_never_reveals_a_variable_name(monkeypatch):
    monkeypatch.setenv("STRIPE_KEY", "sk_live_1234567890abcdef")
    out = make_secret_result_guard([])(
        _result("x=sk_live_1234567890abcdef")
    ).replacement
    assert out == "x=[redacted]"


def test_longer_secret_containing_a_shorter_one_is_fully_redacted(monkeypatch):
    monkeypatch.setenv("SHORT_KEY", "abcdefgh1234")
    monkeypatch.setenv("LONG_KEY", "abcdefgh1234-and-more-secret")
    out = make_secret_result_guard([])(
        _result("v=abcdefgh1234-and-more-secret")
    ).replacement
    assert out == "v=[redacted]"


def test_builder_wires_the_allowlist_into_the_result_guard(monkeypatch):
    monkeypatch.setenv("STRIPE_KEY", "sk_live_1234567890abcdef")
    cap = _build_secrets(None, {"allowed": []})
    assert cap.result_guard(_result("x=sk_live_1234567890abcdef")).replacement == (
        "x=[redacted]"
    )
    assert (
        _build_secrets(None, None)
        .result_guard(_result("x=sk_live_1234567890abcdef"))
        .replacement
        == "x=$SECRET:STRIPE_KEY"
    )
