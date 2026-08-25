"""Browser capability — a real, stateful Chromium browser for agents, mobile-first.

paf-native capability (`AbstractCapability` + `FunctionToolset`, same shape as
`Web` / `FileSystem` / `Shell` / `Git`). This is a full paf-owned copy of the
pydantic-ai-harness `PlaywrightBrowser` (the same "copy the vendor source and
own it" pattern used for Memory / FileSystem / Shell), modified so the browser
is **mobile by default** and can switch viewport/mobile settings at runtime.

Mobile is the only target that matters here (per Landon): the context is born
at a 390×844 touch viewport with a real iPhone user agent, `isMobile` and
`hasTouch` on, and a retina device-scale factor — so responsive sites actually
serve and render their mobile layout. All of it is configurable on the
capability (desktop is one flip of ``mobile=False``), and an agent can switch
profiles mid-run through the ``set_viewport`` tool.

Screenshots ride the same `shot_...` ref → `describe_screenshot` pipeline as
the Visual capability, so a mobile capture is viewed through the vision model
exactly like a desktop one.
"""

from ._capability import Browser
from ._toolset import BrowserSession, BrowserToolset

__all__ = [
    'Browser',
    'BrowserSession',
    'BrowserToolset',
]
