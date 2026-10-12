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
import re
from collections.abc import Callable, Collection

from pydantic_ai_harness.guardrails import GuardrailResult, ToolCallInfo, ToolResultInfo

_HANDLE_PREFIX = "$SECRET:"

# Env values short enough that replacing them everywhere would be noise (e.g.
# "1", "true", "dev"). The result_guard only redacts values that look like
# secrets, so it does not mangle ordinary short strings in tool output.
_MIN_SECRET_LENGTH = 8


def _resolve(name: str) -> str:
    """Resolve a secret handle name to its real value (env vars for now)."""
    return os.environ[name]


def make_secret_handle_guard(
    allowed: Collection[str] | None = None,
) -> Callable[[ToolCallInfo], GuardrailResult]:
    """Build the handle-resolving guard, optionally restricted to ``allowed`` names.

    ``allowed=None`` keeps the original behavior (any environment variable can be
    named by a handle). A collection - including an empty one - restricts the
    guard to those names, so a handle naming anything else is refused as if the
    secret did not exist, without confirming whether the variable is set. That is
    the setting for agents where the model must not be able to ask for
    ``OPENROUTER_API_KEY`` and have it forwarded into a tool argument.
    """
    permitted = None if allowed is None else frozenset(allowed)

    def guard(info: ToolCallInfo) -> GuardrailResult:
        replacement = dict(info.args)
        for key, value in info.args.items():
            if isinstance(value, str) and value.startswith(_HANDLE_PREFIX):
                name = value[len(_HANDLE_PREFIX) :]
                if permitted is not None and name not in permitted:
                    return GuardrailResult.retry(_UNAVAILABLE.format(name=name))
                try:
                    replacement[key] = _resolve(name)
                except KeyError:
                    return GuardrailResult.retry(_UNAVAILABLE.format(name=name))
        if replacement != dict(info.args):
            return GuardrailResult.replace(replacement)
        return GuardrailResult.allow()

    return guard


_UNAVAILABLE = (
    "Secret '{name}' is not available. Ask the user for it or choose another path."
)

# Unrestricted guard: any environment variable may be named by a handle.
secret_handle_guard = make_secret_handle_guard()


# Environment variable NAMES that mark a value as a credential. Matching is
# case-insensitive and by substring (``OPENROUTER_API_KEY``, ``stripe_key``,
# ``GITHUB_TOKEN``). Deliberately absent: ``PWD`` (the working directory, not a
# password) and bare ``AUTH`` (``XAUTHORITY``, ``SSH_AUTH_SOCK`` are paths).
_SECRET_NAME_PARTS: tuple[str, ...] = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE",
    "COOKIE",
    "DSN",
    "DATABASE_URL",
    "CONNECTION_STRING",
)

# Value SHAPES that are credentials whatever the variable is called.
_SECRET_VALUE_SHAPES = re.compile(
    r"^(?:(?:sk|rk|pk)[-_]|gh[pousr]_|github_pat_|xox[abprs]-|AKIA[0-9A-Z]{16}|AIza|eyJ)"
)

_REDACTED = "[redacted]"


def _looks_like_path(value: str) -> bool:
    """A filesystem path (``/home/me``, ``~/x``), never a credential here."""
    return value.startswith(("/", "~")) and not any(c.isspace() for c in value)


def _is_secret(name: str, value: str) -> bool:
    """Whether an environment variable's value should be scrubbed from output.

    A secret is a long-enough value that is not a path and either has a
    credential-looking NAME or a credential-looking SHAPE. Everything else
    (``HOME``, ``PATH``, ``USER``, ``SHELL``, ``LANG``, ``TERM``...) is ordinary
    configuration: scrubbing it corrupts legitimate output and leaks variable
    names for no security gain.
    """
    if len(value) < _MIN_SECRET_LENGTH or _looks_like_path(value):
        return False
    upper = name.upper()
    return any(part in upper for part in _SECRET_NAME_PARTS) or bool(
        _SECRET_VALUE_SHAPES.match(value)
    )


def make_secret_result_guard(
    allowed: Collection[str] | None = None,
) -> Callable[[ToolResultInfo], GuardrailResult]:
    """Build the result-redaction backstop.

    Scrubs credential values from a tool result so a tool that echoes one back
    does not hand it to the model. A value is scrubbed when its variable
    qualifies under :func:`_is_secret`, or when it is on the ``allowed`` list
    (those are credentials by declaration whatever they are called).

    Replacement text: a variable the model is permitted to reference becomes its
    ``$SECRET:NAME`` handle (``allowed=None`` permits all; otherwise only listed
    names), so it can pass it on to a tool. Any other becomes ``[redacted]``,
    which names no variable and offers no handle. Longest values are replaced
    first so a secret that contains another is not left half-visible.
    """
    permitted = None if allowed is None else frozenset(allowed)

    def guard(info: ToolResultInfo) -> GuardrailResult:
        result = info.result
        if not isinstance(result, str):
            return GuardrailResult.allow()
        found: dict[str, str] = {}
        for name, value in os.environ.items():
            declared = permitted is not None and name in permitted
            if value in found:
                continue
            if (declared and len(value) >= _MIN_SECRET_LENGTH) or _is_secret(
                name, value
            ):
                found[value] = (
                    f"{_HANDLE_PREFIX}{name}"
                    if permitted is None or name in permitted
                    else _REDACTED
                )
        redacted = result
        for value in sorted(found, key=len, reverse=True):
            redacted = redacted.replace(value, found[value])
        if redacted != result:
            return GuardrailResult.replace(redacted)
        return GuardrailResult.allow()

    return guard


# Unrestricted backstop (handles for every credential-looking variable).
secret_result_guard = make_secret_result_guard()
