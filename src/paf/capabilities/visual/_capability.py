"""Vision description capability -- `describe_screenshot` over a vision model.

paf-owned capability in the standard seam (`AbstractCapability` +
`FunctionToolset`). Reads the vision model from the `VISION_MODEL` environment
variable at call time (default `openrouter:xiaomi/mimo-v2.5`); the model and
`pydantic-ai` are loaded lazily so the capability constructs and registers even
before a key is configured, and the tool reports a clear message rather than
failing hard when decoding fails or the model call errors.

This is a plain "image in, text out" side-call -- NOT a sub-agent. The agent
that owns the screenshot tool captures a PNG (base64 or on disk, read by the
caller), passes it here, and gets a description string back. No child Agent, no
delegation, no inherited tools.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability

from ._toolset import VisualToolset


@dataclass
class Visual(AbstractCapability):
    """Vision-model description of captured screenshots.

    Exposes a single tool (`describe_screenshot`). The vision model and
    `pydantic-ai` are loaded lazily from the environment at call time, so the
    capability constructs and registers even before a key is configured; the
    tool then reports the missing model / decode error rather than failing hard.
    """

    def get_toolset(self) -> VisualToolset:
        """Build and return the visual toolset."""
        return VisualToolset()
