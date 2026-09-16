"""OpenRouter model construction and defaults.

pydantic-ai resolves an ``openrouter:`` model string via the default provider,
which builds an ``AsyncOpenAI`` client with SDK defaults: a 600s read timeout
and ``max_retries=2``. Against OpenRouter that is a trap: traffic is pinned to
a provider chain, and a provider that goes slow-but-alive (rather than
erroring) makes OpenRouter hold the request indefinitely — its
``allow_fallbacks`` only fires on provider errors, not hangs. The SDK's blind
retries then route right back to the same stalled provider, so the effective
worst case is ~600s x several attempts.

``model_with_client`` builds an ``OpenRouterModel`` whose provider wraps a
custom ``AsyncOpenAI`` client with a bounded read timeout and ``max_retries=0``:
a stall surfaces as a clean ``ModelAPIError`` after ~120s instead of a long
freeze, and retry policy is owned here rather than by the SDK's same-provider
blind retries.
"""

from __future__ import annotations

import os

from openai import AsyncOpenAI
from openai import Timeout as OpenAITimeout
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

OPENROUTER_CLIENT_TIMEOUT_S = 120.0
OPENROUTER_CLIENT_CONNECT_S = 5.0


def model_with_client(model_name: str) -> OpenRouterModel | None:
    """Build an ``OpenRouterModel`` wrapping a bounded-timeout, no-retry client.

    ``model_name`` is the bare OpenRouter model slug (without the ``openrouter:``
    prefix, e.g. ``deepseek/deepseek-v4-flash-0731``). The returned model's
    provider reuses an ``AsyncOpenAI`` client pointed at OpenRouter with
    ``max_retries=0`` and a ~120s read timeout, so a pinned provider that goes
    slow-but-alive fails fast instead of freezing the turn.

    If ``OPENROUTER_API_KEY`` is missing, returns ``None`` so callers can fall
    back to the plain string path (whose own missing-key error then applies)
    rather than constructing a key-less ``AsyncOpenAI`` up front.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=OpenAITimeout(
            OPENROUTER_CLIENT_TIMEOUT_S, connect=OPENROUTER_CLIENT_CONNECT_S
        ),
        max_retries=0,
    )
    return OpenRouterModel(
        model_name, provider=OpenRouterProvider(openai_client=client)
    )
