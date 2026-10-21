"""Ollama model construction and defaults.

pydantic-ai resolves an ``ollama:`` model string via the default provider,
which builds an ``AsyncOpenAI`` client pointed at the local Ollama server with
SDK defaults: a 600s read timeout and ``max_retries=2``. Against a local server
that is mostly harmless, but a stalled model load (first-token latency while
Ollama pulls or loads a model) can hold the request for a long time, and the
SDK's blind retries just re-issue the same stalled request.

``model_with_client`` builds an ``OllamaModel`` whose provider wraps a custom
``AsyncOpenAI`` client with a bounded read timeout and ``max_retries=0``: a
stall surfaces as a clean ``ModelAPIError`` after ~120s instead of a long
freeze, and retry policy is owned here rather than by the SDK's blind retries.
"""

from __future__ import annotations

import os

from openai import AsyncOpenAI
from openai import Timeout as OpenAITimeout
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_CLIENT_TIMEOUT_S = 120.0
OLLAMA_CLIENT_CONNECT_S = 5.0


def model_with_client(model_name: str) -> OllamaModel:
    """Build an ``OllamaModel`` wrapping a bounded-timeout, no-retry client.

    ``model_name`` is the bare Ollama model tag (without the ``ollama:`` prefix,
    e.g. ``llama3.1``). The returned model's provider reuses an ``AsyncOpenAI``
    client pointed at the local Ollama server with ``max_retries=0`` and a
    ~120s read timeout, so a stalled model load fails fast instead of freezing
    the turn.

    Unlike the OpenRouter provider there is no required API key (Ollama is
    local), so this always returns a model; the OpenAI client just needs a
    non-empty placeholder key.
    """
    client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
        timeout=OpenAITimeout(
            OLLAMA_CLIENT_TIMEOUT_S, connect=OLLAMA_CLIENT_CONNECT_S
        ),
        max_retries=0,
    )
    return OllamaModel(
        model_name, provider=OllamaProvider(openai_client=client)
    )
