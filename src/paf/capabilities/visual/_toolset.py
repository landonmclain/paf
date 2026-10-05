"""paf-owned visual toolset -- `describe_screenshot` over a vision model.

A single tool that takes a stored screenshot reference (produced by the
`Visual` capability's ``before_model_request`` hook, which stashes raw
`screenshot`/`navigate` captures under a `shot_...` ref) plus a text prompt,
sends the referenced PNG to a vision-capable model, and returns the model's
text description. The vision model is passed in at construction (from the
capability's spec `visual:` model) and `pydantic-ai` is loaded lazily at call
time, so the toolset constructs without a key and degrades to a clear,
model-visible message on a missing model or a decode/API error -- the same
recoverable-tool convention as `WebToolset`.

This is intentionally NOT a sub-agent: no child run, no delegation. It is a
contained side-call that returns `str(result.output)` straight to the parent
agent's context, keeping the parent model (deepseek) on its pinned,
reproducible path. The parent model asks a TARGETED question (e.g. "is the
padding OK?") and gets a targeted answer -- it never sees the raw pixels
itself, which keeps a text-only main model from crashing on BinaryContent.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import Any, Concatenate, ParamSpec

from pydantic_ai.toolsets import FunctionToolset

from ._store import _store
from ._uploads import UploadStore, is_upload_id
from ._vision import _run_vision

_P = ParamSpec("_P")


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


class VisualToolset(FunctionToolset):
    """Vision-description tools for an agent.

    Exposes `describe_screenshot`. The vision model is passed in at construction
    (from the Visual capability's spec `visual:` model); `pydantic-ai` is
    imported lazily inside the tool so this module imports without a live key.
    """

    def __init__(
        self,
        model: str | None = None,
        uploads: UploadStore | None = None,
        model_settings: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._model_settings = model_settings
        self._uploads = uploads or UploadStore()
        self.add_function(self.describe_screenshot, name="describe_screenshot")

    @_recoverable
    async def describe_screenshot(self, ref: str, prompt: str) -> str:
        """Ask a vision model a question about a screenshot or an uploaded image.

        Args:
            ref: Either a `shot_<...>` reference for a browser screenshot
                captured earlier in this run (from the stub text the browser
                `screenshot` route left in your context), or an `img_<...>` id
                of an image the user uploaded (from an `<image id="...">` block
                in their message). Do not fabricate one.
            prompt: What you want to know about the image, e.g. "is the padding
                OK?" or "what is the exact error text in the dialog?".

        Returns:
            The vision model's text answer to your prompt about that image.
        """
        if is_upload_id(ref):
            image = self._uploads.model_image(ref)
            if image is None:
                return f'describe_screenshot: no uploaded image found for id "{ref}".'
            data, media_type = image
        else:
            data, media_type = _store.get(ref), "image/png"
            if data is None:
                return (
                    "describe_screenshot: no screenshot found for ref "
                    f'"{ref}". Use the ref shown in the [screenshot captured] stub '
                    "text from an earlier browser capture, or an img_ id from an "
                    "<image> block. Do not fabricate a ref."
                )
        if not self._model:
            return (
                "describe_screenshot: no vision model is configured for this "
                "agent (the spec's `visual:` block has no model), so it cannot "
                "inspect images."
            )
        return await _run_vision(
            self._model, [data], prompt, media_type, self._model_settings
        )
