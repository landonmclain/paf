"""paf-owned web toolset -- Tavily-backed web search and page extraction.

Adapted from `coding_agent/app/tools/web_search_tools.py` (personal codebase)
into the paf capability idiom (`FunctionToolset` over the Tavily client).
Semantics and docstrings of the three tools are preserved; session logging is
dropped (not paf's pattern).

The Tavily API key is read from the environment at call time (via
`os.getenv('TAVILY_API_KEY')`), so the toolset can be constructed without a key
and each tool degrades to a clear, model-visible message when the key or the
`tavily-python` package is missing -- exactly the source's behavior.

Tools:
- `search_web_docs` -- general web search for current docs/examples/changelogs.
- `search_official_docs` -- search restricted to known official/source domains.
- `fetch_web_page` -- extract readable content from a specific URL.

Guidance encoded in the tool docstrings: prefer `search_web_docs` /
`search_official_docs` before guessing about current APIs, and follow up with
`fetch_web_page` on a specific URL before relying on a source for code changes
or exact API behavior.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any, Concatenate, Literal, ParamSpec
from urllib.parse import urlparse

from pydantic_ai import RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

_P = ParamSpec('_P')

SearchDepth = Literal['basic', 'advanced']
ExtractDepth = Literal['basic', 'advanced']
ExtractFormat = Literal['markdown', 'text']

DEFAULT_MAX_RESULTS = 5
MAX_SEARCH_RESULTS = 10
DEFAULT_EXTRACT_CHARS = 12_000
MAX_EXTRACT_CHARS = 30_000

PREFERRED_DOC_DOMAINS = [
    'docs.pydantic.dev',
    'ai.pydantic.dev',
    'docs.streamlit.io',
    'docs.tavily.com',
    'docs.python.org',
    'github.com',
    'developer.mozilla.org',
    'nodejs.org',
    'nextjs.org',
    'react.dev',
    'playwright.dev',
    'pytest.org',
    'api.composer.trade',
]


def _recoverable(  # noqa: UP047  # matches fs/shell `_recoverable` convention
    fn: Callable[Concatenate[WebToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[WebToolset, _P], Awaitable[str]]:
    """Let the model retry on transient, correctable web failures.

    Specifically, a missing Tavily key or uninstalled `tavily-python` is
    signalled through the returned string (matching the source), so the model
    can explain/address it; a configured-but-failing upstream call is caught and
    returned as a message rather than aborting the run. Any other exception
    still aborts.
    """

    @functools.wraps(fn)
    async def wrapper(self: WebToolset, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001  # Tavily may raise arbitrary client errors
            return f'Web tool failed: {type(e).__name__}: {e}'

    return wrapper


def _preview_text(text: str, limit: int = 8000) -> str:
    """Truncate from the head, preserving the beginning of the content.

    Mirrors `coding_agent`'s `preview_text`: extracted page content is densest
    at the top, so we keep the head and report the omitted length.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + f'\n... [truncated, total_chars={len(text)}]'


def _safe_json(data: object) -> str:
    """Serialize arbitrary Tavily payloads to indented JSON."""
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _get_tavily_client() -> tuple[Any | None, str | None]:
    """Return a `TavilyClient`, or a (None, error_message) pair.

    Mirrors the source: missing key or missing package yields a clear,
    actionable message instead of an exception.
    """
    api_key = os.getenv('TAVILY_API_KEY', '')
    if not api_key:
        return None, 'TAVILY_API_KEY is not set. Add it to your .env first.'

    try:
        from tavily import TavilyClient  # type: ignore[import-not-found]
    except ImportError:
        return None, (
            'tavily-python is not installed. Install it with:\n\n'
            'uv add tavily-python\n\n'
            'or:\n\n'
            'pip install tavily-python'
        )

    return TavilyClient(api_key=api_key), None


def _domain_from_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower()


def _normalize_domains(domains: list[str] | None) -> list[str] | None:
    """Lowercase, de-scheme, and de-path each domain; drop empties."""
    if not domains:
        return None

    normalized: list[str] = []
    for domain in domains:
        cleaned = domain.strip().lower()
        cleaned = cleaned.removeprefix('https://').removeprefix('http://')
        cleaned = cleaned.split('/', 1)[0]
        if cleaned:
            normalized.append(cleaned)

    return normalized or None


def _format_search_result(result: dict, index: int) -> str:
    title = result.get('title') or '<untitled>'
    url = result.get('url') or '<no url>'
    domain = _domain_from_url(url) if url != '<no url>' else '<unknown>'
    score = result.get('score')
    published_date = result.get('published_date')
    content = result.get('content') or result.get('snippet') or ''

    pieces = [
        f'{index}. {title}',
        f'   URL: {url}',
        f'   Domain: {domain}',
    ]

    if score is not None:
        pieces.append(f'   Score: {score}')

    if published_date:
        pieces.append(f'   Published: {published_date}')

    if content:
        pieces.append(f'   Snippet: {_preview_text(str(content), 700)}')

    return '\n'.join(pieces)


def _format_search_response(response: dict, max_results: int) -> str:
    query = response.get('query')
    answer = response.get('answer')
    results = response.get('results') or []
    response_time = response.get('response_time')

    lines: list[str] = ['Tavily search results']

    if query:
        lines.append(f'Query: {query}')

    if response_time is not None:
        lines.append(f'Response time: {response_time}')

    if answer:
        lines.extend(['', 'Answer:', str(answer)])

    lines.extend(['', f'Results shown: {min(len(results), max_results)}'])

    for index, result in enumerate(results[:max_results], start=1):
        lines.extend(['', _format_search_result(result, index)])

    if not results:
        lines.append('')
        lines.append('No results returned.')

    return '\n'.join(lines)


def _format_extract_response(response: dict, max_chars: int) -> str:
    results = response.get('results') or []
    failed_results = response.get('failed_results') or []
    response_time = response.get('response_time')

    lines: list[str] = ['Tavily extracted page content']

    if response_time is not None:
        lines.append(f'Response time: {response_time}')

    if failed_results:
        lines.append('')
        lines.append('Failed results:')
        lines.append(_safe_json(failed_results))

    if not results:
        lines.append('')
        lines.append('No content extracted.')
        return '\n'.join(lines)

    for index, result in enumerate(results, start=1):
        url = result.get('url') or result.get('source') or '<unknown url>'
        raw_content = (
            result.get('raw_content')
            or result.get('content')
            or result.get('text')
            or ''
        )

        content = _preview_text(str(raw_content), max_chars)

        lines.extend(
            [
                '',
                f'--- Extracted result {index} ---',
                f'URL: {url}',
                f'Chars returned: {len(content)}',
                '',
                content,
            ]
        )

    return '\n'.join(lines)


class WebToolset(FunctionToolset):
    """Tavily-backed web search + page extraction tools for an agent.

    Provides `search_web_docs`, `search_official_docs`, and `fetch_web_page`.
    The API key is read from `TAVILY_API_KEY` at call time; without it the tools
    return a clear message so the model can tell the user what's missing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_function(self.search_web_docs, name='search_web_docs')
        self.add_function(self.search_official_docs, name='search_official_docs')
        self.add_function(self.fetch_web_page, name='fetch_web_page')

    async def for_run(self, ctx: RunContext) -> AbstractToolset:
        """Return a fresh instance per run (stateless here, keeps isolation)."""
        return WebToolset()

    # -- tools -------------------------------------------------------------

    @_recoverable
    async def search_web_docs(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        search_depth: SearchDepth = 'basic',
        include_answer: bool = False,
        preferred_domains: list[str] | None = None,
    ) -> str:
        """
        Search the web for current documentation, examples, changelogs, GitHub
        issues, and framework/library references.

        Use this before guessing about current APIs, package behavior, framework
        conventions, version-specific errors, or best practices.
        """
        max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
        include_domains = _normalize_domains(preferred_domains)

        client, error = _get_tavily_client()
        if error:
            return error
        assert client is not None  # error is None exactly when a client is returned

        try:
            kwargs: dict[str, Any] = {
                'query': query,
                'max_results': max_results,
                'search_depth': search_depth,
                'include_answer': include_answer,
                'include_raw_content': False,
            }

            if include_domains:
                kwargs['include_domains'] = include_domains

            response = client.search(**kwargs)
        except Exception as e:  # noqa: BLE001  # Tavily raises arbitrary client errors
            return f'Tavily search failed: {type(e).__name__}: {e}'

        return _format_search_response(response, max_results=max_results)

    @_recoverable
    async def search_official_docs(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        search_depth: SearchDepth = 'basic',
    ) -> str:
        """
        Search mostly official documentation/source domains.

        Use this for coding tasks where the exact library/framework API matters.
        """
        max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))

        client, error = _get_tavily_client()
        if error:
            return error
        assert client is not None  # error is None exactly when a client is returned

        try:
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=False,
                include_raw_content=False,
                include_domains=PREFERRED_DOC_DOMAINS,
            )
        except Exception as e:  # noqa: BLE001  # Tavily raises arbitrary client errors
            return f'Tavily official-doc search failed: {type(e).__name__}: {e}'

        return _format_search_response(response, max_results=max_results)

    @_recoverable
    async def fetch_web_page(
        self,
        url: str,
        max_chars: int = DEFAULT_EXTRACT_CHARS,
        extract_depth: ExtractDepth = 'basic',
        content_format: ExtractFormat = 'markdown',
    ) -> str:
        """
        Extract readable content from a specific URL.

        Use this after search_web_docs/search_official_docs before relying on a
        source for code changes or exact API behavior.
        """
        max_chars = max(1000, min(max_chars, MAX_EXTRACT_CHARS))

        client, error = _get_tavily_client()
        if error:
            return error
        assert client is not None  # error is None exactly when a client is returned

        try:
            response = client.extract(
                url,
                extract_depth=extract_depth,
                format=content_format,
            )
        except Exception as e:  # noqa: BLE001  # Tavily raises arbitrary client errors
            return f'Tavily extract failed: {type(e).__name__}: {e}'

        return _format_extract_response(response, max_chars=max_chars)
