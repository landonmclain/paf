# paf

**paf** is a small, opinionated agent framework built on
[pydantic-ai](https://ai.pydantic.dev) V2. You describe an agent in a YAML spec,
choose which capabilities it may use, and paf assembles the agent and runs it
with durable conversations, long-term memory, and context management.

It ships with a terminal chat (`paf chat`), an SSE server for web front ends,
and a Python API (`Session`) that both are built on.

## Why it exists

paf started from a handful of design principles drawn from earlier agent
projects, not from a library. When pydantic-ai V2 turned out to provide much of
what those principles called for (capabilities as the extension point, typed
tools, agent specs, per-step persistence), building thin on top of it made more
sense than writing a new foundation.

So paf adds only what pydantic-ai deliberately leaves open:

- **A capability catalog and profiles.** One registry of everything an agent
  could use, plus named allowlists that decide what a given agent does use.
- **Spec loading that handles real objects.** YAML declares what YAML can
  describe; capabilities that need live Python objects (stores, search indexes,
  compaction strategies) are built in code and merged in.
- **Storage, recall, and memory.** Every conversation is persisted to plain
  files and stays searchable, and a background process distills durable facts
  into a memory the agent reads on every request.

The principles behind it:

- **Extensible by construction.** New behavior arrives as a capability, not a
  fork of the runtime.
- **Single agent first.** An `Agent` is the unit; delegation and background
  agents are built from the same pieces.
- **Lossless, inspectable state.** JSON and JSONL on disk, no database to run.
  You can read, grep, and diff everything the agent remembers.
- **Structured traces over logs.** Telemetry goes to Langfuse over
  OpenTelemetry.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/landonmclain/paf.git
cd paf
uv sync
cp .env.example .env   # then set OPENROUTER_API_KEY
```

Start a conversation:

```bash
uv run paf chat                    # the default agent, demo_assistant
uv run paf chat --new              # the same agent, in a fresh conversation
```

The default model is DeepSeek V4 Flash via [OpenRouter](https://openrouter.ai),
so an `OPENROUTER_API_KEY` is the only required setting. To use the browser
capability, install Chromium once with `uv run playwright install chromium`.

## How it fits together

```
specs/<name>.yaml ──► build_agent ──► Agent ──► Session ──► paf chat
  model, instructions,    │                       │          ui/sse.py
  profile, per-capability │                       │          your code
  settings                ▼                       ▼
                 profile (allowlist)        data/  (runs, memory,
                 + capability catalog              conversations, plans)
```

- A **spec** (`specs/<name>.yaml`) names the model, the instructions, the
  profile, and settings for each capability (file-tool roots, shell backend,
  web domains, memory options, and so on).
- A **profile** (`src/paf/capabilities/profiles.py`) is the allowlist of
  capabilities that agent gets, drawn from the catalog in
  `src/paf/capabilities/registry.py`.
- `build_agent` (`src/paf/specs.py`) loads the spec with pydantic-ai's
  `Agent.from_spec` and merges in the capabilities built in Python
  (`src/paf/capabilities/build.py`).
- A **`Session`** (`src/paf/runtime.py`) owns a conversation: it restores prior
  history, stamps the date on each message, compacts context between turns,
  streams events, and schedules background work after each turn. Turns run as
  tasks the session owns, so a client can disconnect and reattach to a running
  turn.

## Agents

| Agent | Spec | Purpose |
| --- | --- | --- |
| `demo_assistant` | `specs/demo_assistant.yaml` | Default for `paf chat`. Everything enabled, including experimental pieces. |
| `secure_assistant` | `specs/secure_assistant.yaml` | Locked-down agent: scoped file tools and sandboxed Python only, with every guard on. Needs Docker. |

Other specs in `specs/` serve benchmarks and tests. Subagents that agents can
delegate to (coding, code review, research, writing) live in
`specs/sub_agents/`, and the background agents that maintain memory and
summaries live in `specs/background_agents/`.

A spec is plain YAML. Abridged:

```yaml
name: my_agent
profile: assistant
model: openrouter:deepseek/deepseek-v4-flash-0731
instructions: |
  You are a capable general-purpose assistant and technical collaborator...

filesystem:
  root_dir: ~/projects/my-project
shell:
  cwd: ~/projects/my-project
  executor:
    backend: local   # or docker, for a locked-down container
```

To add an agent, copy a spec, give it a new `name`, and point `profile` at the
capability set it should have. Every key a spec can set, with its default, is in
the [spec reference](docs/spec-reference.md), and what each capability does is in
[docs/capabilities](docs/capabilities/README.md).

## Capabilities

Each capability bundles tools, instructions, and hooks. A profile picks from
these.

**Conversation and memory**

- **StepPersistence** saves every run, step by step, to a flat-file store.
- **ConversationSearch** gives the model `search_conversation_history` (BM25
  over the full transcript) and `read_turn` (the complete original turn behind
  a search hit).
- **Memory** keeps a durable notebook of facts, injected into every request
  (see [Memory](#memory)).
- **ThreadSummarizer** summarizes each conversation in the background.
- **Compaction** keeps long conversations within budget: it trims the window at
  turn boundaries and clears stale tool results, never cutting into the turn in
  progress.

**Working with files and code**

- **FileSystem** reads, writes, edits, and searches files under a configured
  root.
- **Shell** runs commands in the foreground or background, locally or in a
  Docker container.
- **Git** gives status, diffs, checkpoints, and restores of explicit paths.
- **CodebaseMap** is a navigation-first code browser: orient with
  `get_overview`, then `lookup_file`, then read a tight range with
  `read_block`; `search_codebase` is there when navigation isn't enough.
- **Vault** navigates and edits a folder of Markdown notes, with links and
  backlinks.
- **Workspace** tells the model where each tool family's relative paths start,
  and the user's time zone.
- **Python** runs model-written code with `run_python` in a per-call Docker
  container (no network, read-only root, resource limits). It refuses to run on
  the host unless a spec explicitly allows it.

**Web and vision**

- **Web** searches the web and extracts pages (`search_web`, `fetch_web_page`,
  Tavily when a key is set, otherwise keyless DuckDuckGo).
- **Browser** drives headless Chromium with a mobile-first viewport, limited to
  allowed domains.
- **Visual** describes screenshots and images with a vision model.

**Orchestration**

- **Subagents** delegates tasks to named child agents in isolated runs, in the
  foreground or background (`delegate_task`, `check_task`, `cancel_task`).
- **Planning** gives the model a task plan it maintains as it works
  (`write_plan`, `add_task`, `update_task_status`, and so on), shown live in the
  CLI toolbar.
- **Skills** lists and loads skill documents on demand, and lets the model
  create and edit them.
- **KeywordCurator** points the model at relevant skills and notes. A
  background agent (on when the spec sets `keywords: curate: true`) maintains a
  vocabulary of entities and their aliases from the agent's skill libraries and
  vault; each turn, matching terms inject a short `<relevance-hints>` block,
  with no model call on the hot path.
- **ScheduledTasks** lets the agent schedule work on cron expressions
  (`schedule_task`, `list_scheduled_tasks`, `run_scheduled_task`, and so on).
  A job runs either as an agent turn in the conversation that scheduled it, or
  as a script. Jobs fire while the SSE server (`ui/sse.py`) is running, since
  the scheduler lives there.
- **Push** gives the agent `send_push`, to notify devices subscribed through a
  web front end, via Web Push. Sends are rate-limited and deduplicated. It
  pairs naturally with scheduled tasks ("check X, and tell me only if Y").
- **ToolOutputLimits** shrinks oversized tool results before they reach the
  model.
- **WarnOnCacheBusts** warns when the cacheable prompt prefix changes mid-run.

`demo_assistant` also connects MCP servers, and ships with two keyless examples
in `specs/mcp/demo_servers.json`: Time (a local server started with `uvx`, pinned to
a tested version) and DeepWiki (docs for public GitHub repos; its requests go to
DeepWiki's hosted service). Point the spec's `mcp: config:` at your own
`mcpServers` JSON to use other servers.

Native pydantic-ai capabilities (web search, thinking, MCP, and others) can be
listed directly in a spec's `capabilities:` block, with no Python needed.

## Security

The runtime, not the model, enforces what an agent may do:

- **Approvals.** A risky call pauses until a person approves or denies it, in
  the terminal or in the browser. With nobody attached, it is refused.
- **Shell command policy.** Each command is parsed (pipes, `sudo`, `bash -c`,
  `find -delete` included) and allowed, sent for approval, or refused. The
  lenient policy asks before anything irreversible.
- **Tool gating.** Per-call allow / approve / block rules for any tool, by path
  ("writes are fine under `out/`, elsewhere ask").
- **Prompt-injection screening.** Tool results are scored by a local
  classifier; hits are withheld, redacted or marked untrusted, and later writes
  in the run need approval.
- **Secret handles.** The model names credentials as `$SECRET:NAME` and never
  sees the values; credentials in tool output are scrubbed.
- **Sandbox.** Shell commands and `run_python` can run in a Docker container
  with no network and a read-only root; credential files are blanked inside it
  and refused by the file tools.

The `secure` profile (`secure_assistant`) turns all of it on. See
[docs/security](docs/security/README.md) for how the layers fit and how to
configure each.

## Memory

Memory is owned by paf and stored as plain files under `data/memory/<agent>/`:

- `memory-journal.jsonl` is the **source of truth**: an append-only log, one
  record per fact, each with the date it became true and a reference back to
  the turn it came from.
- `MEMORY.md` is a **rendered view** of the journal. It is what the agent sees,
  and you can read it with `/memory` in the CLI.

The roles are split. A background **owner agent** reads new conversation turns
and appends facts to the journal; it never edits or deletes. Duplicates and
outdated facts are resolved when the view is rendered, so the journal itself is
never rewritten. The assistant only reads memory, and when it needs more than
memory holds, it recalls in order: what's in context, then
`search_conversation_history`, then `read_turn`.

Because each fact carries its source reference, anything in memory can be
traced back to the exact turn that produced it.

## Using paf

### Terminal: `paf chat`

The answer streams into normal scrollback, so the terminal's own scrolling,
search, and copy work. Thinking shows dimmed, each tool call gets one line, and
Markdown renders as each block completes. A toolbar at the bottom shows
background jobs and plan progress.

- **Type while a turn is running to steer it.** Output pauses, and what you send
  is delivered into the running turn.
- **Background work reports back.** When a background shell command or subagent
  finishes, its follow-up turn streams in, even while you're at the prompt.
- **Conversations persist.** `paf chat` reopens your last conversation; `/list`
  and `/resume` switch between them, and `/name` and `/project` organize them.
- **Piped input is one turn.** `cat task.md | paf chat --new` sends the whole
  input as one message, then exits.

`/help` lists every command, including `/image <path>` to send an image and
`/plan`, `/tasks`, and `/memory` to inspect state. From a source checkout,
`python ui/chat.py` is the same command.

### Web front ends: `ui/sse.py`

A FastAPI server exposing the same sessions over Server-Sent Events: a stream
per turn, a separate feed for background processes, steer and cancel, and
endpoints for conversations, projects, plans, image uploads, and Web Push
subscriptions. Turns run on the server, so a page refresh reattaches to a turn
in progress. It also hosts the scheduler for scheduled tasks, and can push a
notification to your devices when a turn finishes.

```bash
uv run python ui/sse.py   # serves on port 8090
```

### Python

```python
from paf.runtime import Session, StreamEvent

session = Session("demo_assistant", conversation_id="notes-1")

# One turn, final text only.
print(session.run_sync("What did we decide about the release date?"))


# Or stream events: text, thinking, tool_call, tool_result.
async def ask(prompt: str) -> None:
    async for event in session.stream(prompt):
        if isinstance(event, StreamEvent) and event.kind == "text":
            print(event.text, end="", flush=True)


session.close()
```

The same `conversation_id` resumes the same conversation in a later process.
For long-running clients, `start_turn` returns a handle you can subscribe to,
cancel, or reattach to, and `enqueue` steers a turn that's in progress.

## Configuration

Settings come from the environment or a `.env` file (see `.env.example`):

| Setting | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | Required for the default model. |
| `OTLP_ENDPOINT`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | Send traces to Langfuse. Leave blank to turn tracing off. |
| `PAF_HOME` | Where `specs/` and `data/` live. Defaults to the repository root in a source checkout, otherwise the current directory. |
| `TIMEZONE` | The user's IANA time zone, for the date stamped on messages. Defaults to the machine's. |
| `PAF_VAPID_PRIVATE_KEY`, `PAF_VAPID_PUBLIC_KEY`, `PAF_VAPID_SUBJECT` | Enable Web Push. Generate once with `scripts/gen_vapid.py --write`; unset means push is off. |

Everything paf writes goes under `data/`, as plain files:

```
data/
  paf/            persisted runs (one directory per run)
  memory/<agent>/ memory journal, rendered MEMORY.md, review state
  conversations/  conversation titles, projects, hidden flags
  plans/          the model's task plans, per conversation
  processes/      background shell and subagent records
  scheduled_tasks/ scheduled jobs and their run history
  keywords/       the keyword curator's vocabulary
  push/           Web Push device subscriptions
  uploads/        images sent to the agent
  approvals/      approval requests and decisions, per agent (JSONL)
```

## Development

```bash
uv run pytest                      # offline suite; live tests are opt-in
uv run pytest -m live              # real model calls (needs .env)
uv run ruff check src tests ui
uv run ruff format --check src tests ui
npx pyright
uv run --group docs mkdocs serve   # documentation site at 127.0.0.1:8000
```

The offline suite uses pydantic-ai's `TestModel`, so it needs no API key. CI
runs the same checks on every push, plus a strict build of the documentation
site (`mkdocs.yml`, `docs/`), so a broken link or anchor fails it.

## Project layout

```
src/paf/
  runtime.py        Session: the turn runtime every client uses
  specs.py          spec loading (agent name → YAML → Agent)
  background.py     post-turn background work (memory, summaries)
  clients.py        helpers shared by the CLI and the SSE server
  push.py           Web Push delivery (VAPID, subscriptions, sender)
  config.py         settings and paths
  capabilities/     the catalog, profiles, builders, and each capability
  cli/              paf chat
  observability/    Langfuse / OpenTelemetry setup
specs/              agent specs, plus sub_agents/ and background_agents/
ui/                 sse.py (web server), chat.py (CLI shim)
tests/              unit tests and the benchmark harness
```

## License

[Apache-2.0](LICENSE)
