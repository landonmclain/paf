"""Vision description capability -- route captured screenshots to a vision model.

paf-owned capability in the standard seam (`AbstractCapability` +
`FunctionToolset`). It serves two jobs, both gated on whether a vision model is
configured:

1. **Hook (``before_model_request``)** -- the safety net. When the main agent
   model is text-only (i.e. a separate vision model is configured via the spec's
   `visual:` block), a
   screenshot taken by a harness browser tool (`screenshot` / `navigate`)
   arrives as raw `BinaryContent` bytes the text-only model would crash on
   (ModelHTTPError 404 "image input"). The hook intercepts each such captured
   image, stashes its bytes in the shared in-process `_store`, and replaces the
   pixel part with a short text stub naming the `shot_...` ref, so the model can
   call `describe_screenshot` with a TARGETED question instead of 404-ing.

   When NO vision model is configured the hook passes the image through
   untouched -- the main model is presumed multimodal and should handle it
   natively (a text-only model with no vision backend will error naturally, as
   before; we deliberately do not guess).

2. **Tool (``describe_screenshot``)** -- the intent channel. The model asks the
   vision model exactly what it wants ("is the padding OK?") about a captured
   screenshot reference.

The vision model is read from the spec's `visual:` block at build time;
`pydantic-ai` is loaded lazily at call time, so the capability constructs and
registers even before a key is configured, and the tool reports a clear message
rather than failing hard when decoding fails or the model call errors. The
describe call is a contained "image in, text out" side-call -- NOT a sub-agent:
no child run, no delegation, no inherited tools -- keeping the parent model
(deepseek) on its pinned, reproducible path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelRequestPart,
    TextContent,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext

from ._store import _store
from ._toolset import VisualToolset

_STUB_PREFIX: str = "[screenshot captured: "
_STUB_SUFFIX: str = (
    "] To inspect it, call describe_screenshot with its ref and your own "
    'targeted question (e.g. "is the padding OK?").'
)


def _is_image(binary: BinaryContent) -> bool:
    """True for a binary part we can route to vision (image media types)."""
    media = (binary.media_type or "").lower()
    return media == "image/png" or media in (
        "image/jpeg",
        "image/jpg",
        "image/gif",
        "image/webp",
    )


def _stub_text(ref: str) -> str:
    """The stub that replaces raw pixels so a text-only model stays safe."""
    return f"{_STUB_PREFIX}{ref}{_STUB_SUFFIX}"


def _walk_and_stash(parts: list[ModelRequestPart]) -> list[ModelRequestPart]:
    """Replace BinaryContent image parts in `parts` with text stubs.

    For each image media-type `BinaryContent` found in a part's multimodal
    `content` (both `ToolReturnPart` and `UserPromptPart` -- a harness
    screenshot surfaces the pixels in both), saves the bytes to the shared
    `_store` and swaps the pixel for a text stub naming the returned ref.
    Returns a NEW list (leaves inputs intact) so the hook can replace a
    message's `parts` without mutating history.
    """
    out: list[ModelRequestPart] = []
    changed = False
    for part in parts:
        content = getattr(part, "content", None)
        if isinstance(part, (ToolReturnPart, UserPromptPart)):
            new_content = _replace_images(content)
            if new_content is not _SENTINEL:
                changed = True
                out.append(replace(part, content=new_content))
                continue
        out.append(part)
    return out if changed else parts


# Sentinel meaning "no images found" (distinct from an empty rebuilt list).
_SENTINEL = object()


def _replace_images(content: Any) -> Any:
    """Recursively replace image BinaryContent parts in `content` with stubs.

    `content` is `ToolReturnContent`, which may be a single multimodal element,
    a sequence of elements, or a mapping of them. Returns the sentinel when no
    image parts were present (caller leaves the value untouched); otherwise
    returns a structurally-equivalent value with images swapped for text stubs.
    """
    if isinstance(content, BinaryContent):
        if _is_image(content):
            ref = _store.save(content.data)
            return TextContent(_stub_text(ref))
        return _SENTINEL

    if isinstance(content, (list, tuple)):
        rebuilt: list[Any] = []
        changed_here = False
        for item in content:
            sub = _replace_images(item)
            if sub is _SENTINEL:
                rebuilt.append(item)
            else:
                rebuilt.append(sub)
                changed_here = True
        if not changed_here:
            return _SENTINEL
        return type(content)(rebuilt) if isinstance(content, tuple) else rebuilt

    if isinstance(content, dict):
        rebuilt_dict: dict[Any, Any] = {}
        changed_here = False
        for key, value in content.items():
            sub = _replace_images(value)
            if sub is _SENTINEL:
                rebuilt_dict[key] = value
            else:
                rebuilt_dict[key] = sub
                changed_here = True
        if not changed_here:
            return _SENTINEL
        return rebuilt_dict

    return _SENTINEL


@dataclass
class Visual(AbstractCapability):
    """Vision-model description of captured screenshots.

    Provides the `describe_screenshot` tool (ask a targeted question about a
    captured screenshot) and the `before_model_request` hook that stashes raw
    capture pixels as refs so a text-only main model never crashes on
    BinaryContent. `model` is the vision model slug from the spec's `visual:`
    block; when it is unset the capability is inert — the hook passes images
    through untouched (presumed multimodal main model) and the tool reports a
    clear message rather than failing hard.
    """

    model: str | None = None
    # The spec's `visual.model_settings`, passed to every vision call.
    model_settings: dict[str, Any] | None = None

    def get_toolset(self) -> VisualToolset:
        """Build and return the visual toolset."""
        return VisualToolset(model=self.model, model_settings=self.model_settings)

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Stash captured screenshot pixels as refs when a vision model is set.

        Only active when a vision model is configured (the capability's `model`
        from the spec `visual:` block) -- that is the signal that the main model
        is text-only and a separate vision backend handles image inspection.
        When none is set, images pass through untouched so a genuinely
        multimodal main model sees them natively.
        """
        if not self.model:
            return request_context

        messages = list(request_context.messages)
        new_messages: list[Any] = []
        changed = False
        for message in messages:
            if not isinstance(message, ModelRequest):
                new_messages.append(message)
                continue
            new_parts = _walk_and_stash(list(message.parts))
            if new_parts is not message.parts:
                changed = True
            new_messages.append(replace(message, parts=new_parts))

        if changed:
            request_context.messages = new_messages
        return request_context
