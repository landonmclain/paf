"""paf-owned secret-handle guard policy.

The harness ``ToolGuardrail`` is the shell; paf owns the policy as callables
here (the settled don't-pin stance). This module is the policy for "Secrets
Never in the Window": the model proposes a handle (``$SECRET:key``), the
runtime resolves it to the real value at execution time, and the real value
never enters the model's context window.

Resolution source for the first wiring: environment variables. ``$SECRET:key``
resolves to ``os.environ[key]``. The resolver is a single function so the store
is swappable later (e.g. a profile-scoped ``SecretStore``) without touching the
guard.

Two guards ride the same ``ToolGuardrail``:

* ``secret_handle_guard`` (the ``guard=`` stage) resolves handles in tool
  arguments right before the tool executes, via ``replace``. The model proposed
  the handle; the runtime swaps in the real value; the real value never lands
  in the window.
* ``secret_result_guard`` (the ``result_guard=`` stage) is a backstop that
  redacts any raw secret value that leaks into a tool result, so a tool that
  echoes a secret back does not hand it to the model.
"""

from __future__ import annotations

import os

from pydantic_ai_harness.guardrails import GuardrailResult, ToolCallInfo, ToolResultInfo

_HANDLE_PREFIX = "$SECRET:"

# Env values short enough that replacing them everywhere would be noise (e.g.
# "1", "true", "dev"). The result_guard only redacts values that look like
# secrets, so it does not mangle ordinary short strings in tool output.
_MIN_SECRET_LENGTH = 8


def _resolve(name: str) -> str:
    """Resolve a secret handle name to its real value (env vars for now)."""
    return os.environ[name]


def secret_handle_guard(info: ToolCallInfo) -> GuardrailResult:
    """Resolve ``$SECRET:key`` handles in tool args at execution time.

    ``replace`` hands the tool the full resolved argument mapping, so the real
    value is what executes while the window only ever saw the handle. A handle
    naming an unavailable secret is a ``retry``: the model learns the secret is
    not available and can ask the user or choose another path.
    """
    replacement = dict(info.args)
    for key, value in info.args.items():
        if isinstance(value, str) and value.startswith(_HANDLE_PREFIX):
            name = value[len(_HANDLE_PREFIX):]
            try:
                replacement[key] = _resolve(name)
            except KeyError:
                return GuardrailResult.retry(
                    f"Secret '{name}' is not available. Ask the user for it or choose another path."
                )
    if replacement != dict(info.args):
        return GuardrailResult.replace(replacement)
    return GuardrailResult.allow()


def secret_result_guard(info: ToolResultInfo) -> GuardrailResult:
    """Redact raw secret values that leaked into a tool result (backstop).

    Scans the result for any env value that looks like a secret and replaces it
    with its handle, so a tool that echoes a secret back does not hand the real
    value to the model. Conservative by design: only values at least
    ``_MIN_SECRET_LENGTH`` long are considered, to avoid mangling ordinary short
    strings in tool output.
    """
    result = info.result
    if not isinstance(result, str):
        return GuardrailResult.allow()
    redacted = result
    for name, value in os.environ.items():
        if len(value) >= _MIN_SECRET_LENGTH and value in redacted:
            redacted = redacted.replace(value, f"{_HANDLE_PREFIX}{name}")
    if redacted != result:
        return GuardrailResult.replace(redacted)
    return GuardrailResult.allow()
