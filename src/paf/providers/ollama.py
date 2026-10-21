"""Local-model provider: bounded-timeout OpenAI-compatible client.

The ``ollama:`` model prefix routes here (see ``specs.py``) and builds an
``OllamaModel`` whose provider wraps a custom ``AsyncOpenAI`` client pointed
at ``OLLAMA_BASE_URL``. In this deployment that is llama.cpp's
``llama-server`` (127.0.0.1:8080/v1, a systemd user service) serving the
Qwen3.8 GGUF directly — not the ollama binary — but any OpenAI-compatible
``/v1/chat/completions`` endpoint works, since pydantic-ai's Ollama provider
is just an AsyncOpenAI wrapper and the model tag is opaque to the server.

Why a custom client: pydantic-ai's default builds an ``AsyncOpenAI`` client
with SDK defaults (600s read timeout, ``max_retries=2``). Against a local
server that is mostly harmless, but first-token latency on long contexts can
hold the request for a long time, and the SDK's blind retries just re-issue
the same stalled request. Here a bounded read timeout (120s) and
``max_retries=0`` make a stall surface as a clean ``ModelAPIError`` instead of
a long freeze, with retry policy owned here rather than by the SDK.
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

    ``model_name`` is the bare model tag (without the ``ollama:`` prefix, e.g.
    ``qwen3.8:27b-90k``); it is opaque to the server, which serves whatever
    model it was started with. The returned model's provider reuses an
    ``AsyncOpenAI`` client pointed at the local server with ``max_retries=0``
    and a ~120s read timeout, so a stalled prefill fails fast instead of
    freezing the turn.

    Unlike the OpenRouter provider there is no required API key (the server is
    local), so this always returns a model; the OpenAI client just needs a
    non-empty placeholder key.
    """
    client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
        timeout=OpenAITimeout(OLLAMA_CLIENT_TIMEOUT_S, connect=OLLAMA_CLIENT_CONNECT_S),
        max_retries=0,
    )
    return OllamaModel(model_name, provider=OllamaProvider(openai_client=client))
