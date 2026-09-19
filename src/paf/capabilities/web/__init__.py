"""Web capability -- Tavily-backed web search and page extraction for agents.

paf-owned web search and page extraction, surfaced through the standard
capability seam (`AbstractCapability` + `FunctionToolset` + registry + profile
allowlist + builder). Exposes `search_web` (open web search) and
`fetch_web_page` (domain-gated extract). The `TAVILY_API_KEY` is read from the
environment at call time; tools degrade to clear messages when it or
`tavily-python` is missing.
"""

from ._capability import Web
from ._toolset import WebToolset

__all__ = [
    "Web",
    "WebToolset",
]
