# Orchestration and extensions

Capabilities that organise work: delegating to other agents, planning, loading
know-how on demand, running things later, telling you about it, and connecting
outside tools.

| Capability | Tools | Spec block |
| --- | --- | --- |
| [Subagents](#subagents) | `delegate_task`, `check_task`, `cancel_task` | `subagents` |
| [Planning](#planning) | `write_plan`, `read_plan`, `add_task`, `update_task_status`, … | none |
| [Skills](#skills) | `list_skills`, `load_skill`, `list_skill_files`, `read_skill_file`, `create_skill`, `edit_skill` | `skills` |
| [KeywordCurator](#keywordcurator) | none: injects `<relevance-hints>` | `keywords` |
| [ScheduledTasks](#scheduledtasks) | `schedule_task`, `list_scheduled_tasks`, `update_scheduled_task`, `remove_scheduled_task`, `run_scheduled_task` | none |
| [Push](#push) | `send_push` | none |
| [MCP](#mcp) | whatever each server offers | `mcp` |

## Subagents

An agent can hand a self-contained job to a named **subagent**: a separate
agent with its own model, instructions and tools, run in a fresh context.

- **`delegate_task(agent_name, task, background?)`** runs it. Foreground
  returns the full output. `background=true` returns a task id at once, and the
  result arrives later as a message in the conversation, so the model
  needn't poll.
- **`check_task(task_id, offset?)`** returns status and a page of output.
- **`cancel_task(task_id)`** stops a running background task.

A subagent **does not see the conversation**. The task must carry everything
it needs, and it is told to prefer file paths over pasted content. Each gets
the same scoped file tools as its parent, and every result is saved as a
Markdown file under `data/processes/<agent>/`, so a finished task can be
re-read.

### Defining one

A subagent is a spec in `specs/sub_agents/`, named in the parent's roster:

```yaml
# specs/sub_agents/coding_subagent.yaml
name: coding_subagent
model: openrouter:deepseek/deepseek-v4-flash-0731
description: General coding sub-agent for small, well-scoped engineering tasks.
profile: coding              # its tools: codebase map + scoped file tools
subagent:
  name: coding
  timeout_seconds: 300
  usage_limits:
    request_limit: 30
    total_tokens_limit: 400000
  background: required       # allowed | required | never
instructions: |
  ...
```

```yaml
# in the parent's spec
subagents:
  agents: [writer_subagent, code_reviewer_subagent, coding_subagent]
  max_concurrent: 5
```

| `subagent:` key | Meaning |
| --- | --- |
| `timeout_seconds` | Wall-clock limit for one run (default 300). |
| `usage_limits` | `request_limit` (default 30) and `total_tokens_limit` for one run. |
| `background` | `allowed`, `required` (only runs in the background, for long work), or `never`. |

paf ships four: `writer`, `code_reviewer`, `coding` and `researcher`. A leaf's
`profile` picks its tools from the leaf profiles (`researcher`: web and files;
`coding`: codebase map and files). Leaves never get persistence, memory,
summaries or subagents of their own, so delegation is one level deep.

### Limits

- **A subagent doesn't inherit the parent's guards.** It is built from its own
  spec and profile, so the parent's tool gating, shell command policy and
  injection screening don't apply to it unless the leaf spec sets up its own.
  Check what a leaf's profile and spec give it before adding one to the roster
  of a locked-down agent. `secure` has no `Subagents` capability for this
  reason.
- **Approvals aren't routed from subagents.** Sending a subagent's approval
  requests to you is not built yet, so don't rely on an `approve` rule inside a
  leaf.

## Planning

On a long task the model keeps a plan: a list of tasks with statuses,
optionally with subtasks and dependencies.

- `write_plan` replaces the plan, `read_plan` shows it.
- `add_task`, `update_task_status`, `update_task_statuses`, `remove_task`.
- `add_subtask`, `set_dependency` and `get_available_tasks` for work with
  structure: a parent completes by itself when its last subtask does.

The current plan is shown back to the model with every request as a reminder
that sits after the cache breakpoint, so editing the plan doesn't invalidate the
prompt cache. Plans are saved per conversation at
`data/plans/<conversation_id>/plan.json`, which survives restarts. You can read
the plan with `/plan` in `paf chat`, and the web UI shows it in a sidebar.

## Skills

A **skill** is a directory with a `SKILL.md` (a name, a description, and
instructions for doing one kind of task well) plus optional resources in
`references/`, `scripts/`, `templates/` and `assets/`. Skills keep the system
prompt short: the model sees the list and loads only what the task needs.

- **`list_skills`** shows the library as a category tree with one-line
  descriptions.
- **`load_skill`** reads a skill's instructions; `list_skill_files` and
  `read_skill_file` reach its resources.
- **`create_skill`** and **`edit_skill`** let the model write and refine skills
  in the library, so know-how it works out can be kept.

```yaml
skills:
  libraries:
    - ~/.hermes/skills
    - ~/projects/skill-library
```

Skills are indexed when the agent is built, so a skill added while it runs
appears after a restart. Because the model can write skills that later turns
load as instructions, point `libraries` only at directories you want it to
change.

## KeywordCurator

Tells the model which skills and notes are relevant to what you just said,
without a model call on the turn.

A background job reads your skills and vault notes and builds a vocabulary:
canonical entities and their aliases, each pointing at a source. Each turn, the
last couple of your messages are matched against it, and the best few matches
are injected as a short `<relevance-hints>` block of pointers, which the
model can follow with `load_skill` or `read_note`. No match means nothing is
injected.

```yaml
keywords:
  inject: true
  top_k: 3
  curate: true       # opt in: build the vocabulary in the background
```

`curate` is **off by default** because a curation pass sends digests of your
skills and notes to the model provider. Matching and injection don't call a
model. Knobs are in the [spec reference](../spec-reference.md#keywords).

## ScheduledTasks

The agent can put work on a schedule. A task belongs to the conversation that
created it.

- **`schedule_task(schedule, name, description, instructions | script, notify?)`**
  takes a cron expression (`0 9 * * *` is 9am daily) and **exactly one** of:
  - `instructions`: an **agent task**. At each firing the agent runs a turn in
    the conversation and follows the instructions.
  - `script`: a **script task**. A script runs with no model, and the agent is
    brought in depending on `notify`.
- **`list_scheduled_tasks`**, **`update_scheduled_task`**,
  **`remove_scheduled_task`**, and **`run_scheduled_task`** to fire one now.

For script tasks, `notify: always` has the agent review every run. This is the
way to start, while you build trust in the script. `notify: on_trigger` brings
the agent in only when the script **exits non-zero**: a cheap check that costs
tokens only when it finds something. Pair it with [Push](#push): "check X, and
tell me only if Y".

Jobs live in `data/scheduled_tasks/`, with a run history beside them. **The
scheduler runs inside the SSE server** (`paf-sse`), so jobs fire only while it
is up. They are not available in a bare `paf chat`.

### Safety

- **Script tasks run on the host.** Not in the sandbox, not through the
  [shell command policy](../security/shell-command-policy.md), with a 60-second
  limit. A script task is as powerful as the shell, and a model that can
  schedule one can run code on your machine later. Don't give a locked-down
  profile this capability.
- **Nobody is attached when a task fires**, so an agent task that reaches a
  call needing [approval](../security/approvals.md) is refused rather than
  waiting.

## Push

`send_push(title, body, urgent?)` sends a notification to your subscribed
devices (the web UI, over Web Push). The model is told to use it sparingly:
when you'd want to be interrupted, and not to repeat what it just said or report
that there's nothing to report. Tapping the notification opens the
conversation, and the link is built by the runtime, not chosen by the model.

The runtime enforces the sparing use: at most **10 per hour per agent**, an
identical title and body suppressed for **10 minutes**, a title cut to 60
characters and a body to 200. Only pushes that reach a device count against
the limit.

The tool exists only where delivery is wired, which means the SSE server with
VAPID keys set (`scripts/gen_vapid.py --write`); in `paf chat` the model is
never offered a tool that can't work.

## MCP

Connects [Model Context Protocol](https://modelcontextprotocol.io) servers, so
their tools appear as the agent's own. Available in the `demo` profile.

```yaml
mcp:
  config: specs/mcp/demo_servers.json
```

The file is the Claude Desktop `mcpServers` format:

```json
{
  "mcpServers": {
    "time": { "command": "uvx", "args": ["mcp-server-time@2026.10.19"] },
    "deepwiki": { "type": "http", "url": "https://mcp.deepwiki.com/mcp" }
  }
}
```

- Each server's tools are prefixed with its name.
- `${VAR}` and `${VAR:-default}` in the file expand from the environment, so
  credentials stay in `.env`, not in the config.
- Servers connect on first use. A missing config file is an error at build time.

An MCP server runs outside paf's sandbox and policies: a `command` server is a
process on your machine, and an `http` server sends what it's given to that
host. Add only servers you trust, and pin versions as the example does.
