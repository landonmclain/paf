"""paf-owned visual toolset -- `describe_screenshot` over a vision model.

A single tool that takes base64-encoded PNG(s) plus a text prompt, sends them to
a vision-capable model, and returns the model's text description. The vision
model and `pydantic-ai` are loaded lazily at call time (via `VISION_MODEL` /
`os.getenv`), so the toolset constructs without a key and degrades to a clear,
model-visible message on a missing model or a decode/API error -- the same
recoverable-tool convention as `WebToolset`.

This is intentionally NOT a sub-agent: no child run, no delegation. It is a
contained side-call that returns `str(result.output)` straight to the parent
agent's context, keeping the parent model (deepseek) on its pinned,
reproducible path.
"""

from __future__ import annotations

import base64
import binascii
import functools
import os
from collections.abc import Awaitable, Callable
from typing import Concatenate, ParamSpec

from pydantic_ai.toolsets import FunctionToolset

_P = ParamSpec('_P')

# Default vision model for `describe_screenshot` -- a native omnimodal model that
# accepts image input and returns text. Override via VISION_MODEL / .env.
_VISION_MODEL_DEFAULT = 'openrouter:xiaomi/mimo-v2.5'


def _recoverable(  # noqa: UP047  # matches fs/shell/web `_recoverable` convention
    fn: Callable[Concatenate[VisualToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[VisualToolset, _P], Awaitable[str]]:
    """Let the model retry on transient, correctable vision failures.

    A missing model, a base64 decode failure, or a provider error is signalled
    through the returned string (matching the Web toolset convention), so the
    model can explain/address it rather than the run aborting. Any other
    exception still aborts.
    """

    @functools.wraps(fn)
    async def wrapper(self: VisualToolset, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001  # vision provider may raise arbitrary errors
            return f'Vision describe failed: {type(e).__name__}: {e}'

    return wrapper


def _vision_model() -> str:
    """The vision model slug from `VISION_MODEL` (default Xiaomi MiMo-V2.5)."""
    return os.getenv('VISION_MODEL', _VISION_MODEL_DEFAULT)


def _is_base64_png(token: str) -> bool:
    """True when `token` cleanly decodes as base64 that starts with a PNG magic.

    A real PNG's decoded bytes begin with ``\\x89PNG\\r\\n\\x1a\\n``. Checking the
    magic (not just decode-ability) filters whitespace/corrupt tokens so a
    concatenated multi-image `images_b64` arg is split on real captures.
    """
    if not token:
        return False
    try:
        raw = base64.b64decode(token, validate=True)
    except (binascii.Error, ValueError):
        return False
    return raw.startswith(b'\x89PNG\r\n\x1a\n')


class VisualToolset(FunctionToolset):
    """Vision-description tools for an agent.

    Exposes `describe_screenshot`. The vision model is read from `VISION_MODEL`
    at call time; `pydantic-ai` is imported lazily inside the tool so this
    module imports without a live key.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_function(self.describe_screenshot, name='describe_screenshot')

    @_recoverable
    async def describe_screenshot(self, images_b64: str, prompt: str) -> str:
        """Describe one or more screenshots with a vision model.

        Args:
            images_b64: One or more base64-encoded PNG images, concatenated
                (each decoded independently). Pass the raw base64 from a prior
                `screenshot` capture -- no `data:` prefix.
            prompt: What you want to know about the image(s), e.g. "describe
                this rendered page and identify anything broken".

        Returns:
            The vision model's text description of the image(s).
        """
        from pydantic_ai import Agent
        from pydantic_ai.messages import BinaryContent, TextContent

        # Each whitespace-delimited token that decodes as a real PNG is one
        # image, so a caller can pass several concatenated captures in one call.
        tokens = images_b64.split()
        images = [base64.b64decode(t) for t in tokens if _is_base64_png(t)]
        if not images:
            return (
                'describe_screenshot: no valid base64 PNG image found in '
                '`images_b64`. Pass the raw base64 from a prior `screenshot` '
                'capture (no `data:` prefix).'
            )

        content: list[TextContent | BinaryContent] = [TextContent(prompt)]
        content.extend(BinaryContent(data=raw, media_type='image/png') for raw in images)

        agent = Agent(_vision_model())
        result = await agent.run(content)
        return str(result.output)
