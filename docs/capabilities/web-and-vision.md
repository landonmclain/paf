# Web and vision

Reaching beyond the machine: searching and reading the web, driving a browser,
and looking at images. These are the capabilities that bring in outside content,
which is also the main source of prompt injection. `secure` includes none of
them, and in the others every result passes through
[injection screening](../security/prompt-injection.md) when the profile has it.

| Capability | Tools | Spec block | Needs |
| --- | --- | --- | --- |
| [Web](#web) | `search_web`, `fetch_web_page` | `web` | `TAVILY_API_KEY` |
| [Browser](#browser) | `navigate`, `click`, `type_text`, `snapshot`, `screenshot`, … | `browser` | Playwright and Chromium |
| [Visual](#visual) | `describe_screenshot` | `visual` | a vision model |

## Web

Search and page extraction through [Tavily](https://tavily.com).

- **`search_web(query, max_results?, search_depth?, include_answer?, preferred_domains?)`**
  is open search over the public web. `preferred_domains` narrows one query to
  chosen sites. The model is told to search before guessing about current APIs,
  versions or errors.
- **`fetch_web_page(url, max_chars?, extract_depth?, content_format?)`** returns
  a page as markdown or text.

The two are deliberately unequal. Search returns short snippets, so it is open.
Extraction pulls whole pages into context, which is where injected text would
come in, so **`fetch_web_page` only reads allowlisted domains**:

```yaml
web:
  search_domains:
    - docs.pydantic.dev
    - github.com
```

Absent, a built-in list of documentation sites applies. The page is marked as
untrusted data in what the model reads, and the model is told to treat it as a
document to quote, never as instructions.

The key is read from the environment when a tool is called. Without the key (or
the `tavily-python` package) the tools reply with a clear message instead of
failing the run.

## Browser

A real Chromium browser the agent can operate: for sites that need JavaScript,
logins you've set up, or checking how a page looks. It's driven with Playwright,
headless, and starts on the first browser call (a run that never browses pays
nothing).

**Mobile by default.** The browser presents as an iPhone: a 390×844 touch
viewport, retina scale, and an iPhone user agent, so responsive sites serve
their mobile layout. `set_viewport` switches profile mid-run.

**Tools:**

| | |
| --- | --- |
| Navigate | `navigate`, `go_back`, `go_forward`, `tabs`, `wait_for` |
| Read | `snapshot` (the page's accessibility tree), `get_text`, `screenshot`, `console_messages`, `network_requests` |
| Act | `click`, `type_text`, `press_key`, `select_option`, `hover`, `scroll`, `handle_next_dialog` |
| Other | `execute_js`, `set_viewport` |

With a vision model configured, a screenshot doesn't reach the main model as
pixels: it's stored as a `shot_…` reference and described by the
[Visual](#visual) capability on request.

### Where it can go

The browser is locked down by default, with two independent rules and deny
wins:

- **`allowed_domains`** bounds which hosts it can reach, subdomains included.
  Navigation is checked, and so are the requests a page's scripts use to send
  data (`fetch`, XHR, WebSocket, `sendBeacon`). Passive assets (images, fonts,
  scripts) load freely, so an allowed page still renders. **The default is
  loopback only** (`localhost`, `127.0.0.1`, `::1`).
- **`block_private_addresses`** (default `true`) refuses loopback, LAN and
  link-local addresses in every frame, resolving names first so a hostname
  pointing at one is refused too. Cloud metadata (`169.254.169.254`) is refused
  always.

The two together mean the default browser reaches **nothing**: its allowlist
names loopback, and the private-address guard forbids loopback. To browse a
public site, list it. To browse a local development server, list it **and** set
`block_private_addresses: false`.

```yaml
browser:
  allowed_domains: [docs.example.com]
```

### Limits

- **Not a hard boundary.** The egress rules don't stop DNS rebinding, because
  Chromium resolves a name again before it connects. List only hosts you trust
  an agent to read, and keep browser-enabled agents away from sensitive tools.
- **Chromium runs with its own sandbox off.** On Ubuntu 23.10 and later, an
  AppArmor rule stops unprivileged user namespaces, and Chromium's sandbox then
  fails at launch. paf disables it so the browser starts. That weakens
  isolation from the page, so enable user namespaces on the host if you can, and
  treat the browser as exposed to whatever it loads.
- **Setup.** `playwright install chromium` once. If Chromium isn't there, the
  tool says so and the run continues.
- **No durable execution.** A live page can't survive replay, so the browser
  can't be combined with Temporal-style durability.

## Visual

Lets a text-only model work with images, by routing them to a vision model.

- **`describe_screenshot(ref, prompt)`** asks the vision model a specific
  question ("is the padding right?", "what does the error dialog say?") about
  `ref`: a `shot_…` from the browser, or an `img_…` id for an image you sent.

```yaml
visual:
  model: openrouter:xiaomi/mimo-v2.5
```

With a `visual.model`, images that reach the conversation (browser captures, or
images you send, with `/image <path> [message]` in `paf chat` or an upload in the
web UI) are swapped for short text stubs, so a text-only model never receives
bytes it can't read. The vision model is a contained side-call: image in, text
out, no tools and no conversation history. It is not a subagent.

Without a `visual:` block the capability does nothing and images pass to the
main model unchanged, which is right if that model can see.
