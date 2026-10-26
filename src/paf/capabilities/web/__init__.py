"""Web capability -- web search and page extraction for agents.

paf-owned web search and page extraction, surfaced through the standard
capability seam (`AbstractCapability` + `FunctionToolset` + registry + profile
allowlist + builder). Exposes `search_web` (open web search) and
`fetch_web_page` (domain-gated extract). Backed by Tavily when `TAVILY_API_KEY`
is set, otherwise by keyless DuckDuckGo plus an SSRF-protected local fetch.
"""

from ._capability import Web
from ._toolset import SearchBackend, WebToolset

__all__ = [
    "SearchBackend",
    "Web",
    "WebToolset",
]
