"""Web capability -- Tavily-backed web search and page extraction for agents.

paf-owned adaptation of `coding_agent/app/tools/web_search_tools.py`, surfaced
through the standard capability seam (`AbstractCapability` + `FunctionToolset` +
registry + profile allowlist + builder). Exposes `search_web_docs`,
`search_official_docs`, and `fetch_web_page`. The `TAVILY_API_KEY` is read from
the environment at call time; tools degrade to clear messages when it or
`tavily-python` is missing.
"""

from ._capability import Web
from ._toolset import WebToolset

__all__ = [
    'Web',
    'WebToolset',
]
