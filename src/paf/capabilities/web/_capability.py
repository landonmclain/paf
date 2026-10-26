"""Web capability -- web search and page extraction (Tavily or DuckDuckGo).

paf-owned web capability in the standard seam (`AbstractCapability` +
`WebToolset`). `search_backend: auto` uses Tavily when `TAVILY_API_KEY` is set and
keyless DuckDuckGo otherwise; a missing key or package degrades to a clear,
model-visible message.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability

from ._toolset import SearchBackend, WebToolset


@dataclass
class Web(AbstractCapability):
    """Web search + page extraction tools for an agent.

    Exposes two tools (`search_web`, `fetch_web_page`). The Tavily key and
    package are loaded lazily at call time, so the capability constructs and
    registers even before a key is configured; tools then report what is
    missing rather than failing hard.

    `search_backend` (`web.search_backend` in the spec) is `tavily`,
    `duckduckgo` (no key; page fetch runs locally behind an SSRF guard), or
    `auto` (Tavily when `TAVILY_API_KEY` is set, otherwise DuckDuckGo).

    `search_domains` is the extract trust boundary — which domains
    `fetch_web_page` will pull content from, declared per-spec
    (`web.search_domains` in the spec YAML). Defaults to
    `PREFERRED_DOC_DOMAINS` when not supplied. Open web *search* is never
    restricted by this (search returns only low-risk snippets).
    """

    search_domains: list[str] | None = None
    search_backend: SearchBackend = "auto"

    def get_toolset(self) -> WebToolset:
        """Build and return the web toolset."""
        return WebToolset(
            search_domains=self.search_domains, search_backend=self.search_backend
        )
