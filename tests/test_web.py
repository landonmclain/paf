"""Web capability: backend selection, DuckDuckGo path, SSRF-protected fetch."""

from __future__ import annotations

import asyncio

import pytest

from paf.capabilities.web import WebToolset, _toolset
from paf.capabilities.web._toolset import (
    _resolve_backend,
    _with_site_filter,
)
from paf.specs import _spec_web_config


def test_auto_backend_follows_tavily_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    assert _resolve_backend("auto") == "tavily"
    monkeypatch.delenv("TAVILY_API_KEY")
    assert _resolve_backend("auto") == "duckduckgo"
    assert _resolve_backend("tavily") == "tavily"


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="search_backend"):
        WebToolset(search_backend="bing")  # type: ignore[arg-type]


def test_site_filter():
    assert _with_site_filter("q", None) == "q"
    assert _with_site_filter("q", ["a.com", "b.com"]) == "q (site:a.com OR site:b.com)"


def test_spec_web_config(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("web:\n  search_backend: duckduckgo\n  search_domains: [a.com]\n")
    assert _spec_web_config(p) == {
        "search_domains": ["a.com"],
        "search_backend": "duckduckgo",
    }
    p.write_text("web:\n  search_backend: bing\n")
    with pytest.raises(ValueError, match="search_backend"):
        _spec_web_config(p)


def test_duckduckgo_search_formats_results(monkeypatch):
    async def scenario():
        seen = {}

        def fake(query, max_results):
            seen["query"] = query
            return [{"title": "T", "href": "https://docs.python.org/x", "body": "snip"}]

        monkeypatch.setattr(_toolset, "_ddgs_text", fake)
        out = await WebToolset(search_backend="duckduckgo").search_web(
            "asyncio", preferred_domains=["docs.python.org"]
        )
        assert "DuckDuckGo search results" in out
        assert "https://docs.python.org/x" in out and "snip" in out
        assert seen["query"] == "asyncio (site:docs.python.org)"

    asyncio.run(scenario())


def test_duckduckgo_search_failure_is_a_message(monkeypatch):
    async def scenario():
        def boom(query, max_results):
            raise RuntimeError("rate limited")

        monkeypatch.setattr(_toolset, "_ddgs_text", boom)
        out = await WebToolset(search_backend="duckduckgo").search_web("q")
        assert "DuckDuckGo search failed" in out and "rate limited" in out

    asyncio.run(scenario())


def test_local_fetch_enforces_allowlist():
    async def scenario():
        out = await WebToolset(
            search_domains=["docs.python.org"], search_backend="duckduckgo"
        ).fetch_web_page("https://example.com/")
        assert "Web fetch blocked" in out

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1/",
    ],
)
def test_local_fetch_blocks_private_hosts_even_if_allowlisted(url):
    async def scenario():
        host = url.split("/")[2]
        out = await WebToolset(
            search_domains=[host], search_backend="duckduckgo"
        ).fetch_web_page(url)
        assert out.startswith("Web fetch failed")

    asyncio.run(scenario())


@pytest.mark.live
def test_duckduckgo_live_search_and_fetch(monkeypatch):
    async def scenario():
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        tools = WebToolset(search_domains=["docs.python.org"], search_backend="auto")
        assert "docs.python.org" in await tools.search_web("python asyncio gather")
        page = await tools.fetch_web_page(
            "https://docs.python.org/3/library/asyncio.html"
        )
        assert "SSRF-protected" in page and "asyncio" in page

    asyncio.run(scenario())
