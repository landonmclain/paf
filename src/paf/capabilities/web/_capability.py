"""Web capability -- Tavily-backed web search and page extraction.

paf-owned web capability. Adapted from
`coding_agent/app/tools/web_search_tools.py` into the paf capability seam
(`AbstractCapability` + `WebToolset`). Reads `TAVILY_API_KEY` from the
environment at call time; without it (or without `tavily-python`) the tools
degrade to clear, model-visible messages.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability

from ._toolset import WebToolset


@dataclass
class Web(AbstractCapability):
    """Tavily-backed web search + page extraction tools for an agent.

    Exposes two tools (`search_web`, `fetch_web_page`). The Tavily API key and
    `tavily-python` package are loaded lazily from the environment at call time,
    so the capability constructs and registers even before a key is configured;
    individual tools then report the missing key/package rather than failing
    hard.

    `search_domains` is the extract trust boundary — which domains
    `fetch_web_page` will pull content from, declared per-spec
    (`web.search_domains` in the spec YAML). Defaults to
    `PREFERRED_DOC_DOMAINS` when not supplied. Open web *search* is never
    restricted by this (search returns only low-risk snippets).
    """

    search_domains: list[str] | None = None

    def get_toolset(self) -> WebToolset:
        """Build and return the web toolset."""
        return WebToolset(search_domains=self.search_domains)
