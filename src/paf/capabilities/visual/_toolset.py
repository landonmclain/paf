"""paf-owned visual toolset -- `describe_screenshot` over a vision model.

A single tool that takes a stored screenshot reference (produced by the
`Visual` capability's ``before_model_request`` hook, which stashes raw
`screenshot`/`navigate` captures under a `shot_...` ref) plus a text prompt,
sends the referenced PNG to a vision-capable model, and returns the model's
text description. The vision model and `pydantic-ai` are loaded lazily at call
time (via `VISION_MODEL` / `os.getenv`), so the toolset constructs without a
key and degrades to a clear, model-visible message on a missing model or a
decode/API error -- the same recoverable-tool convention as `WebToolset`.

This is intentionally NOT a sub-agent: no child run, no delegation. It is a
contained side-call that returns `str(result.output)` straight to the parent
agent's context, keeping the parent model (deepseek) on its pinned,
reproducible path. The parent model asks a TARGETED question (e.g. "is the
padding OK?") and gets a targeted answer -- it never sees the raw pixels
itself, which keeps a text-only main model from crashing on BinaryContent.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Awaitable, Callable
from typing import Concatenate, ParamSpec

from pydantic_ai.toolsets import FunctionToolset

from paf.degeneration_guard import run_guarded

from ._store import _store

_P = ParamSpec("_P")

# Default vision model for `describe_screenshot` -- a native omnimodal model that
# accepts image input and returns text. Override via VISION_MODEL / .env.
_VISION_MODEL_DEFAULT = "openrouter:xiaomi/mimo-v2.5"


def _recoverable(  # noqa: UP047  # ParamSpec/Concatenate signature-preserving decorator; PEP 695 inline type params can't express it
    fn: Callable[Concatenate[VisualToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[VisualToolset, _P], Awaitable[str]]:
    """Let the model retry on transient, correctable vision failures.

    A missing model, a missing ref, or a provider error is signalled through the
    returned string (matching the Web toolset convention), so the model can
    explain/address it rather than the run aborting. Any other exception still
    aborts.
    """

    @functools.wraps(fn)
    async def wrapper(self: VisualToolset, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001  # vision provider may raise arbitrary errors
            return f"Vision describe failed: {type(e).__name__}: {e}"

    return wrapper


def _vision_model() -> str:
    """The vision model slug from `VISION_MODEL` (default Xiaomi MiMo-V2.5)."""
    return os.getenv("VISION_MODEL", _VISION_MODEL_DEFAULT)


async def _run_vision(images: list[bytes], prompt: str) -> str:
    """Send `images` + `prompt` to the vision model and return its text.

    The one vision side-call, kept here so the capability and toolset never
    duplicate it. The `Agent` and message types are imported lazily so this
    module imports without a live key.
    """
    from pydantic_ai import Agent
    from pydantic_ai.messages import BinaryContent, TextContent

    if not images:
        return "Vision describe: no image bytes to describe."

    content: list[TextContent | BinaryContent] = [TextContent(prompt)]
    content.extend(BinaryContent(data=raw, media_type="image/png") for raw in images)

    agent = Agent(_vision_model())
    result = await run_guarded(agent, content)
    return str(result.output)


class VisualToolset(FunctionToolset):
    """Vision-description tools for an agent.

    Exposes `describe_screenshot`. The vision model is read from `VISION_MODEL`
    at call time; `pydantic-ai` is imported lazily inside the tool so this
    module imports without a live key.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_function(self.describe_screenshot, name="describe_screenshot")

    @_recoverable
    async def describe_screenshot(self, ref: str, prompt: str) -> str:
        """Describe a captured screenshot with a vision model.

        Args:
            ref: The `shot_<...>` reference for a screenshot captured earlier in
                this run (see the stub text left in your context by the browser
                `screenshot` route). Do not fabricate one.
            prompt: What you want to know about the image, e.g. "is the padding
                OK?" or "does the hero image look centered?".

        Returns:
            The vision model's text answer to your prompt about that screenshot.
        """
        images = _store.get(ref)
        if images is None:
            return (
                "describe_screenshot: no screenshot found for ref "
                f'"{ref}". Use the ref shown in the [screenshot captured] stub '
                "text from an earlier browser capture. Do not fabricate a ref."
            )
        return await _run_vision([images], prompt)
