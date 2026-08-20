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

    Exposes three tools (`search_web_docs`, `search_official_docs`,
    `fetch_web_page`). The Tavily API key and `tavily-python` package are loaded
    lazily from the environment at call time, so the capability constructs and
    registers even before a key is configured; individual tools then report the
    missing key/package rather than failing hard.
    """

    def get_toolset(self) -> WebToolset:
        """Build and return the web toolset."""
        return WebToolset()
