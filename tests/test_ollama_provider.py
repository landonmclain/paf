"""The ``ollama:`` provider sends past thinking as ``reasoning_content``, not inline.

Servers such as Strata drop historical ``reasoning_content`` from the next prompt.
Sent inline as ``<think>`` text it is replayed on every step of a long agent turn.
"""

from __future__ import annotations

import asyncio

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters

from paf.providers.ollama import model_with_client


def _assistant_message(model, messages):
    async def go():
        prepared = model.prepare_messages(messages)
        if asyncio.iscoroutine(prepared):
            prepared = await prepared
        return await model._map_messages(prepared, ModelRequestParameters())

    return next(m for m in asyncio.run(go()) if m["role"] == "assistant")


def test_thinking_is_sent_in_the_reasoning_content_field():
    model = model_with_client("any-tag")
    messages = [
        ModelRequest(parts=[UserPromptPart(content="go")]),
        ModelResponse(
            parts=[
                ThinkingPart(
                    content="step by step",
                    id="reasoning_content",
                    provider_name="ollama",
                ),
                TextPart(content="Reading the file."),
            ]
        ),
    ]
    msg = _assistant_message(model, messages)
    assert msg["reasoning_content"] == "step by step"
    assert msg["content"] == "Reading the file."


def test_thinking_without_a_provider_id_is_not_inlined_as_think_tags():
    model = model_with_client("any-tag")
    messages = [
        ModelRequest(parts=[UserPromptPart(content="go")]),
        ModelResponse(
            parts=[ThinkingPart(content="from elsewhere"), TextPart(content="ok")]
        ),
    ]
    msg = _assistant_message(model, messages)
    assert "<think>" not in (msg.get("content") or "")
