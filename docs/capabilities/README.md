# Capabilities

A **capability** is a unit of agent behaviour: tools the model can call,
instructions that explain them, and hooks that run around each request or tool
call. Everything an agent can do is a capability, and a **profile** is the
allowlist of which ones an agent may have. A capability left out of the profile
is never built, so no prompt can reach it.

A spec then configures the ones its profile allows, in a block named for the
capability (`shell:`, `memory:`, …). Every key is in the
[spec reference](../spec-reference.md).

## The catalog

`assistant`, `demo` and `secure` are the three profiles you'd run an agent
under (`demo` is `assistant` plus MCP). `●` means the profile includes it.

**Conversations, memory and context.**
[Conversations and memory](conversations-and-memory.md) ·
[Context management](context.md)

| Capability | What it's for | assistant | demo | secure |
| --- | --- | :-: | :-: | :-: |
| StepPersistence | Saves every run, step by step | ● | ● | ● |
| ConversationSearch | Search and re-read the full history | ● | ● | ● |
| Memory | A durable notebook of facts, written in the background | ● | ● |  |
| ThreadSummarizer | The story of one conversation, as dated beats | ● | ● |  |
| Compaction | Trims old messages and tool results to fit the window | ● | ● | ● |
| ToolOutputLimits | Moves oversized tool results out of the window | ● | ● | ● |
| WarnOnCacheBusts | Warns when prompt caching stops working | ● | ● |  |

**Files, code and commands.** [Files and code](files-and-code.md)

| Capability | What it's for | assistant | demo | secure |
| --- | --- | :-: | :-: | :-: |
| FileSystem | Read, write, edit and search under one root | ● | ● | ● |
| Shell | Run commands, foreground or background | ● | ● |  |
| Python | Run model-written Python in a throwaway container | | | ● |
| Git | Inspect and checkpoint a repository, safely | ● | ● |  |
| CodebaseMap | Navigate a codebase without reading whole files | ● | ● |  |
| Vault | A folder of Markdown notes, with links | ● | ● |  |
| Workspace | Tells the model where each tool's paths start | ● | ● | ● |

**Web and vision.** [Web and vision](web-and-vision.md)

| Capability | What it's for | assistant | demo | secure |
| --- | --- | :-: | :-: | :-: |
| Web | Search, and read allowlisted pages | ● | ● |  |
| Browser | A real Chromium the agent can operate | ● | ● |  |
| Visual | Describe images through a vision model | ● | ● |  |

**Orchestration and extensions.** [Orchestration](orchestration.md)

| Capability | What it's for | assistant | demo | secure |
| --- | --- | :-: | :-: | :-: |
| Subagents | Delegate jobs to other agents | ● | ● |  |
| Planning | A task plan the model keeps up to date | ● | ● |  |
| Skills | Load instructions on demand | ● | ● |  |
| KeywordCurator | Per-turn pointers to relevant skills and notes | ● | ● |  |
| ScheduledTasks | Run work on a cron schedule | ● | ● |  |
| Push | Notify your phone | ● | ● |  |
| MCP | Connect Model Context Protocol servers | | ● |  |

**Security.** [Security](../security/README.md)

| Capability | What it's for | assistant | demo | secure |
| --- | --- | :-: | :-: | :-: |
| InjectionGuard | Screens tool results for prompt injection | | | ● |
| ToolGating | Per-call allow, approve or block rules | | | ● |
| Secrets | Credential handles in, scrubbed values out | | | ● |

The guards are off in `assistant` and `demo` by design: they are for agents you
want to constrain, and `secure` turns all of them on together. Approvals are
not in the table because they're installed on every agent (an approval broker
that denies when nobody is attached), and the shell command policy belongs to
Shell: a spec's `shell.command_policy` turns it on.

## Reading a page

Each page gives, per capability: its tools by name, what they do and don't do,
the spec block that configures it, where its data lives under `data/`, and its
limits. The tool names are the ones the model sees, so they match what you'll
read in traces and in `paf chat`.

## Adding one

A capability is a class with a toolset, instructions and hooks, registered in
`src/paf/capabilities/registry.py`, built in `build.py`, and named in a profile.
If it takes a spec block, add the parser in `specs.py` and a row in the
[spec reference](../spec-reference.md) (a test fails until you do).
