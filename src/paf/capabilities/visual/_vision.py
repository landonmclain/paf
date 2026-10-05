"""The one vision side-call: image bytes + a prompt in, the model's text out.

Shared by the `describe_screenshot` tool and the upload describer
(``_uploads.py``) so neither duplicates it. The `Agent` and message types are
imported lazily so this module imports without a live key. The model is any
pydantic-ai model string -- ``openrouter:openai/gpt-6-luna`` today, a local
``ollama:...`` model just as well.
"""

from __future__ import annotations

from typing import Any, cast

from paf.degeneration_guard import run_guarded


async def _run_vision(
    model: str,
    images: list[bytes],
    prompt: str,
    media_type: str = "image/png",
    model_settings: dict[str, Any] | None = None,
) -> str:
    """Send `images` + `prompt` to `model` and return its text.

    `media_type` labels every image in the call (callers pass one image, or
    same-typed captures). `model_settings` comes from the spec's
    ``visual.model_settings`` (e.g. ``openai_reasoning_effort: none`` to stop a
    local thinking model reasoning before every description).
    """
    from pydantic_ai import Agent, ModelSettings
    from pydantic_ai.messages import BinaryContent, TextContent

    if not images:
        return "Vision describe: no image bytes to describe."

    content: list[TextContent | BinaryContent] = [TextContent(prompt)]
    content.extend(BinaryContent(data=raw, media_type=media_type) for raw in images)

    agent = Agent(model, model_settings=cast(ModelSettings, model_settings))
    result = await run_guarded(agent, content)
    return str(result.output)
