"""Visual capability -- vision-model description of captured screenshots.

paf-native capability (`AbstractCapability` + `FunctionToolset`, same shape as
`Web` / `FileSystem` / `Shell` / `Git`). Exposes a single tool,
`describe_screenshot`, that hands one or more base64-encoded PNG images to a
vision-capable model and returns its text description to the agent.

This is deliberately NOT a sub-agent: there is no child `Agent`, no delegation,
and no inherited toolset. It is a contained vision side-call -- exactly like the
Memory capability invoking its consolidation model, or `_build_subagents`
constructing a leaf `Agent` -- that takes image bytes in and returns a string
out. The vision model is selected via `VISION_MODEL` (default
`openrouter:xiaomi/mimo-v2.5`, a native omnimodal model that accepts image input
and returns text).
"""

from ._capability import Visual
from ._toolset import VisualToolset

__all__ = [
    'Visual',
    'VisualToolset',
]
