"""Playwright capability -- a real, stateful Chromium browser for agents.

The dated external-service assumptions this package relies on (aria snapshots,
selector engines, service-worker blocking, binary detection, timeouts) are
recorded in the `_toolset` module docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.durable_exec._base import (
    BaseDurabilityCapability,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT

from ._toolset import (
    DEFAULT_ACTION_TIMEOUT_MS,
    DEFAULT_MAX_CONTENT_TOKENS,
    DEFAULT_NAVIGATION_TIMEOUT_MS,
    BrowserSession,
    BrowserToolset,
    EgressPolicy,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from playwright.async_api import StorageState
    from pydantic_ai.agent import AbstractAgent

_INSTRUCTIONS = """\
You have a real browser for pages the lighter tools cannot handle — things behind
login/session cookies, JavaScript-heavy SPA behavior, and interactive multi-step flows.

Textual tool results are truncated to roughly {max_content_tokens} tokens (use `get_text`
with a selector to read a specific section of a large page). Allowed domains: {allowed_domains}.
"""


@dataclass
class Browser(AbstractCapability[AgentDepsT]):
    """A real, stateful Chromium browser for an agent, via async Playwright.

    Adds eighteen tools -- navigate, snapshot, click, type_text, press_key,
    select_option, hover, wait_for, screenshot, get_text, scroll, go_back,
    go_forward, execute_js, tabs, handle_next_dialog, console_messages,
    network_requests -- backed by a Chromium context that persists across tool
    calls within a run. Reach for it when
    the lighter web tools fall short: pages behind login/session cookies,
    JavaScript-rendered SPAs, and interactive multi-step flows. For query-based
    research prefer a web-search tool; for a static URL prefer a web-fetch tool.

    ```python
    from pydantic_ai import Agent
    from paf.capabilities.browser import Browser

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Browser()])
    ```

    Requires the `playwright` optional extra and the Chromium binary:

    ```bash
    pip install 'pydantic-ai-harness[playwright]'
    playwright install chromium
    ```

    Egress: `allowed_domains` defaults to a loopback-only allowlist (`localhost`,
    `127.0.0.1`, `::1`); pass `allowed_domains=[...]` to bound navigation to other
    hosts, or `None` for open egress. The allowlist bounds navigation
    and the request kinds that move data (`fetch`, XHR, EventSource, WebSocket,
    `sendBeacon`) in any frame, leaving passive subresources and sub-frame
    documents unbounded so a permitted page keeps its assets and its
    identity-provider frames. Independently, `block_private_addresses=True` (the
    default) refuses private, loopback, link-local and other reserved addresses
    for every request kind in every frame, even under open egress, resolving a
    hostname first so a name pointing at one of those addresses is refused too.
    Neither is a general security boundary: an unanswered DNS lookup is refused
    but Chromium resolves the name again before it connects, so rebinding is not
    closed, and a proxy-based enforcement mode is tracked in
    https://github.com/pydantic/pydantic-ai-harness/issues/415. Set
    `allowed_domains` when the agent may act on untrusted input.

    Chromium starts lazily on the first browser-tool call and is closed when the
    run ends (on success, error, or cancellation); runs that never call a browser
    tool pay no Playwright cost. When no browser can be started the tool returns a
    `playwright install chromium` hint instead of ending the run, so an agent that
    can run a shell can install it and carry on, and the process gets a
    `BrowserUnavailableWarning` for the developer who is not reading tool results.
    Set `auto_install_chromium=True` to fetch the binary automatically instead. Set `cdp_url` to attach to a Chromium that is already running
    (managed-browser providers, benchmark harnesses) rather than launching one.

    Every browser operation runs inside an OpenTelemetry span, and what the page
    did during it (console output, responses, requests the egress policy refused,
    dialogs it opened, tabs it opened) is attached to that span as span events and
    readable by the agent through `console_messages` and `network_requests`.

    Durable execution (e.g. `TemporalDurability`) is not supported: durability
    replays tool calls as activities and a live Chromium page cannot survive
    replay or worker restart, so combining both on one agent raises `UserError`
    at agent construction.
    """

    headless: bool = True
    """Run Chromium without a visible window. `True` suits servers and CI."""

    allowed_domains: list[str] | None = field(
        default_factory=lambda: ["localhost", "127.0.0.1", "::1"]
    )
    """Egress allowlist. Defaults to loopback-only (`localhost`, `127.0.0.1`, `::1`); `None` allows every public host -- see the egress note above.

    Each entry matches its exact host and any subdomain, so `example.com` reaches
    `api.example.com`. It bounds top-level navigation and the requests a page's
    scripts use to move data (`fetch`, XHR, EventSource, WebSocket, `sendBeacon`);
    passive subresources and sub-frame documents are left alone, so a permitted
    page keeps its CDN assets and its identity-provider frames. Enforced at two
    layers: a network route guard aborts a refused request before it leaves, and
    each tool re-checks the resulting URL and bounces to `about:blank` so
    disallowed content never reaches the model.

    For anything this does not express -- a denylist, apex-only matching, locking
    down every request type, a rule of your own -- pass an `EgressPolicy` as
    `policy` instead.
    """

    policy: EgressPolicy | None = None
    """Full egress policy, for rules the two shorthands above cannot express.

    Mutually exclusive with `allowed_domains` and `block_private_addresses`, which
    build the default policy when this is unset. Subclass `EgressPolicy` and
    override `refuse` for a decision the fields do not cover. Absent from
    `from_spec`, since a policy can carry arbitrary code.
    """

    block_private_addresses: bool = True
    """Refuse addresses that are not globally routable, for every request kind in every frame.

    Covers the cloud metadata endpoint (`169.254.169.254`), loopback
    (`127.0.0.1`, `::1`, `localhost`), and the RFC 1918 ranges, independent of
    `allowed_domains` -- open egress still cannot reach them. A hostname is
    resolved first, so a name pointing at one of those addresses is refused too,
    as is one whose lookup does not answer (see the egress note above). Set
    `False` when the agent should reach a local app or an internal dashboard.
    """

    screenshot_on_navigate: bool = False
    """Attach a screenshot (as image content) to every `navigate` result."""

    max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS
    """Approximate token budget for textual tool results."""

    action_timeout_ms: int = DEFAULT_ACTION_TIMEOUT_MS
    """Default deadline for element actions (click, type, read, wait), in milliseconds.

    Shorter than the navigation budget on purpose. An action that misses is
    normally a selector matching nothing, and a long deadline makes that look
    like a hung agent rather than a fast failure the model can react to. Raise it
    for pages whose elements appear slowly.
    """

    navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS
    """Default deadline for navigation and load settling, and for starting or attaching to the browser, in milliseconds."""

    chromium_sandbox: bool = True
    """Run the launched Chromium with its renderer sandbox.

    On by default, unlike Playwright itself: this capability opens pages nobody
    vetted, and the sandbox is what keeps a renderer that a crafted page
    compromises from reaching the host. Set `False` where the sandbox cannot
    start -- a container without the kernel privileges it needs is the usual case
    -- and accept that a renderer exploit then runs with the agent's own access.
    Ignored when `cdp_url` is set: that browser is already running under its own
    configuration.
    """

    auto_install_chromium: bool = False
    """Fetch the Chromium binary via `playwright install chromium` on the first miss.

    Off by default: a library should not download a browser as a side effect. When
    the binary is missing the browser tool returns a clear install hint, which an
    agent that can run a shell can act on. Set `True` to opt into the automatic
    download instead.
    """

    storage_state: StorageState | None = field(default=None, repr=False)
    """Playwright storage state (cookies, localStorage) loaded into the browser context at launch.

    Obtain it in your own code -- `await context.storage_state()`, or
    `json.loads(Path('auth.json').read_text())` for a file written by
    `playwright codegen --save-storage` -- so the login runs where you control
    it and the credentials never have to reach wherever the agent runs. This is
    session material equivalent to the account: keep it out of source control
    and out of logs, and discard it when the session expires.

    An object rather than a path so the capability never assumes a filesystem it
    can read: agents often run where the local disk is neither durable nor
    writable, so where the state lives stays the caller's decision. For the same
    reason `from_spec` does not accept it.
    """

    cdp_url: str | None = field(default=None, repr=False)
    """Attach to a Chromium already running at this Chrome DevTools Protocol endpoint.

    When set, the capability connects instead of launching, so no local Chromium
    binary is needed and `auto_install_chromium` does not apply. Used for
    managed-browser providers and benchmark harnesses that hand the agent a
    browser. A new browser context is still created for the run, so
    `storage_state` and the egress guards apply as they do for a launched
    browser, and the run does not inherit the sessions already open in that
    Chrome. Provider endpoints sometimes carry an auth token in the URL; treat
    those as secrets.
    """

    viewport: tuple[int, int] = (390, 844)
    """Browser viewport as `(width, height)` CSS pixels.

    Mobile is the only layout that matters by default: a 390×844 iPhone-class
    portrait viewport. Set e.g. `(768, 1024)` for a tablet or `(1280, 720)` for
    a desktop. Because viewport/mobile flags are a Chromium *context* property
    (only settable at `new_context()` time), changing one recreates the page.
    """

    mobile: bool = True
    """Master switch for mobile emulation.

    `True` (default) applies `is_mobile`, `has_touch`, the retina
    `device_scale_factor` and (when set) the phone `user_agent` so responsive
    sites serve and render their mobile layout. `False` turns those off for a
    plain desktop render at the same `viewport`.
    """

    is_mobile: bool = True
    """Whether the meta viewport tag is honored (mobile page rendering) when `mobile=True`."""

    has_touch: bool = True
    """Whether the context has a touch surface (touch events, no hover) when `mobile=True`."""

    device_scale_factor: int = 3
    """Pixel density (devicePixelRatio) — 3 is retina/iPhone-class density, when `mobile=True`."""

    user_agent: str | None = None
    """User-agent string sent to sites, when `mobile=True`.

    `None` (default) lets Playwright's mobile context use its own mobile UA. Set
    an explicit iPhone UA to force a specific device fingerprint.
    """

    _session: BrowserSession = field(init=False, repr=False)
    _toolset: BrowserToolset[AgentDepsT] = field(init=False, repr=False)
    _policy: EgressPolicy = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.policy is not None and (
            self.allowed_domains is not None or not self.block_private_addresses
        ):
            raise UserError(
                "Pass either policy or the allowed_domains/block_private_addresses shorthands, not both: "
                "set those fields on the policy instead, so one object decides."
            )
        self._policy = self.policy or EgressPolicy(
            allowed_domains=self.allowed_domains,
            block_private_addresses=self.block_private_addresses,
        )
        self._session = BrowserSession(
            policy=self._policy,
            headless=self.headless,
            storage_state=self.storage_state,
            cdp_url=self.cdp_url,
            auto_install_chromium=self.auto_install_chromium,
            chromium_sandbox=self.chromium_sandbox,
            viewport=self.viewport,
            mobile=self.mobile,
            is_mobile=self.is_mobile,
            has_touch=self.has_touch,
            device_scale_factor=self.device_scale_factor,
            user_agent=self.user_agent,
            launch_timeout_ms=self.navigation_timeout_ms,
        )
        self._toolset = BrowserToolset[AgentDepsT](
            session=self._session,
            screenshot_on_navigate=self.screenshot_on_navigate,
            max_content_tokens=self.max_content_tokens,
            action_timeout_ms=self.action_timeout_ms,
            navigation_timeout_ms=self.navigation_timeout_ms,
        )

    def for_agent(
        self, agent: AbstractAgent[AgentDepsT, object]
    ) -> AbstractCapability[AgentDepsT]:
        """Refuse to bind to a durable-execution agent.

        Durable execution (e.g. `TemporalDurability`) wraps the toolsets captured
        at agent construction and replays tool calls as activities. A live
        Chromium page cannot be checkpointed across activity boundaries or worker
        restarts, so the composition cannot work; without this guard it fails
        deep inside the first browser tool call instead.

        Detection matches `BaseDurabilityCapability`, the shared base of the
        bundled Temporal/DBOS/Prefect integrations. Pydantic AI exposes no public
        marker for the durability tier, and the `innermost` ordering position is
        not one: `InputGuard` also declares `innermost`, so ordering alone would
        reject the supported guard-plus-browser composition.
        """
        siblings: list[AbstractCapability[AgentDepsT]] = []
        agent.root_capability.apply(siblings.append)
        for sibling in siblings:
            if isinstance(sibling, BaseDurabilityCapability):
                raise UserError(
                    "PlaywrightBrowser does not support durable execution (e.g. TemporalDurability): "
                    "a live Chromium browser cannot survive activity replay or worker restart. "
                    "Run the browser agent outside the durable agent, or remove the durability capability."
                )
        return self

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Browser[AgentDepsT]:
        """Return a fresh instance per run so concurrent runs never share a page or browser."""
        return replace(self)

    def get_toolset(self) -> BrowserToolset[AgentDepsT]:
        """Provide the eighteen browser tools."""
        return self._toolset

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], str | None]:
        """When-to-use guidance for the browser."""

        def _instructions(ctx: RunContext[AgentDepsT]) -> str | None:
            return _INSTRUCTIONS.format(
                max_content_tokens=self.max_content_tokens,
                allowed_domains=self._policy.describe(),
            )

        return _instructions

    async def wrap_run(
        self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler
    ) -> AgentRunResult[Any]:
        """Hold the run's browser session open, and release it however the run ends.

        Chromium starts on the first browser-tool call, not here, so a run that
        never browses never launches one. The run's tracer is adopted here so
        browser spans follow the agent's instrumentation settings rather than a
        tracer of this module's choosing.
        """
        self._session.tracer = ctx.tracer
        async with self._session:
            return await handler()

    @classmethod
    def from_spec(
        cls,
        *,
        headless: bool = True,
        allowed_domains: list[str] | None = None,
        block_private_addresses: bool = True,
        screenshot_on_navigate: bool = False,
        max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS,
        action_timeout_ms: int = DEFAULT_ACTION_TIMEOUT_MS,
        navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
        auto_install_chromium: bool = False,
        chromium_sandbox: bool = True,
        cdp_url: str | None = None,
        viewport: tuple[int, int] = (390, 844),
        mobile: bool = True,
        is_mobile: bool = True,
        has_touch: bool = True,
        device_scale_factor: int = 3,
        user_agent: str | None = None,
    ) -> Browser[AgentDepsT]:
        """Construct the capability from serializable spec options (all fields are plain scalars/lists).

        A spec carries connection configuration, not session credentials:
        `storage_state` is deliberately absent, so a spec naming it raises rather
        than moving cookies into whatever stores the spec. Set it on the
        constructed capability instead.
        """
        return cls(
            headless=headless,
            allowed_domains=list(allowed_domains)
            if allowed_domains is not None
            else None,
            block_private_addresses=block_private_addresses,
            screenshot_on_navigate=screenshot_on_navigate,
            max_content_tokens=max_content_tokens,
            action_timeout_ms=action_timeout_ms,
            navigation_timeout_ms=navigation_timeout_ms,
            auto_install_chromium=auto_install_chromium,
            chromium_sandbox=chromium_sandbox,
            cdp_url=cdp_url,
            viewport=viewport,
            mobile=mobile,
            is_mobile=is_mobile,
            has_touch=has_touch,
            device_scale_factor=device_scale_factor,
            user_agent=user_agent,
        )
