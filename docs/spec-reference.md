# Spec reference

An agent is one YAML file in `specs/`. `paf chat secure_assistant` loads
`specs/secure_assistant.yaml`; subagent and background-agent specs live in
subfolders and resolve by bare name the same way.

A spec has two kinds of keys:

- **pydantic-ai keys** (`name`, `model`, `instructions`, `model_settings`,
  `capabilities`, …) are read by pydantic-ai's `AgentSpec`.
- **paf keys** (everything else on this page) configure paf's capabilities.
  A block only takes effect when the agent's **profile** includes the
  capability it configures. A `shell:` block on a profile without `Shell` does
  nothing.

Every block is optional. An absent block or key falls back to the default listed
here. Unknown keys inside most blocks are ignored, so older specs keep loading.
The security policies are the exception: `gating`, `guardrails.injection` and
`shell.command_policy` **refuse to load** when malformed or when they contain an
unknown key, because a half-applied security policy is worse than none.

Paths accept `~`. Relative paths in blocks that take a root are relative to the
repository root.

Contents: [Identity and model](#identity-and-model) ·
[Run limits](#run-limits) · [Files and code](#files-and-code) ·
[Execution](#execution) · [Security](#security) · [Memory and recall](#memory-and-recall) ·
[Context](#context) · [Web and vision](#web-and-vision) ·
[Orchestration](#orchestration) · [Benchmark prompts](#benchmark-prompts)

---

## Identity and model

### `name`

The agent's name. Keep it equal to the file name.

### `profile`

**Required.** The capability allowlist for this agent: which capabilities may be
built at all. `build_agent` raises if a spec declares no profile, or names one
that isn't registered, so a typo can never widen what an agent can do.

| Profile | For | Includes |
| --- | --- | --- |
| `assistant` | everyday use | conversation, memory, files, shell, git, codebase map, vault, web, browser, vision, skills, subagents, planning, scheduled tasks, push |
| `demo` | trying new things | everything in `assistant`, plus MCP |
| `secure` | high-stakes work | persistence, conversation search, compaction, scoped files, sandboxed Python, workspace, output limits, and all three guards (InjectionGuard, ToolGating, Secrets). No shell, no network, no memory. |
| `contextbench`, `longmemeval` | benchmarks | the minimum each benchmark needs |

Subagent leaves use their own profiles (`researcher`, `coding`). The definitions
live in `src/paf/capabilities/profiles.py`.

### `model`

The model, as `provider:model`. `openrouter:` and `ollama:` models get paf's
bounded-timeout clients; anything pydantic-ai accepts works.

### `instructions`

The system prompt. A string or a list of strings. Capabilities append their own
instructions (how to use their tools, what the runtime will refuse) after it.

### `model_settings`

Passed to the model as-is (`temperature`, `max_tokens`, provider-specific keys
such as `openrouter_reasoning` or `extra_body`). Absent means pydantic-ai's
defaults.

### `capabilities`

pydantic-ai native capabilities declared directly in YAML (for example
`StepPersistence`, web search, thinking). Only types the profile allows will
resolve.

### Other pydantic-ai keys

`description`, `retries`, `end_strategy`, `tool_timeout`, `metadata`,
`output_schema`, `deps_schema` are pydantic-ai's own; see its `AgentSpec`.

---

## Run limits

### `usage_limits`

| Key | Default | Meaning |
| --- | --- | --- |
| `request_limit` | `50` (pydantic-ai's) | Most model requests one turn may make before it stops. |

### `approvals`

| Key | Default | Meaning |
| --- | --- | --- |
| `timeout` | `600` | Seconds an approval request waits for a person before the call is denied. |

See [Approvals](security/approvals.md).

---

## Files and code

### `filesystem`

The file tools (`read_file`, `write_file`, `edit_file`, `list_directory`,
`search_files`, `find_files`, `file_info`, `create_directory`). They are always
confined to `root_dir`: no `..` traversal, no escaping through symlinks, and
`.git/`, `.env` and key files are read-only.

| Key | Default | Meaning |
| --- | --- | --- |
| `root_dir` | repository root | The only directory the tools can reach. |
| `deny_secret_files` | `false` | Refuse to read, write, list or search credential files at any depth (`.env*`, keys, `.ssh/`, …; [the list](security/sandbox.md#secret-files)). |
| `denied_patterns` | `[]` | Extra globs (relative to the root) the tools refuse entirely. |

### `git`

| Key | Default | Meaning |
| --- | --- | --- |
| `root` | repository root | The repository the git tools operate on. Paths are confined to it. |

### `codebase_map`

| Key | Default | Meaning |
| --- | --- | --- |
| `scan_root` | repository root | The tree to index. The index is built on demand and kept fresh. |

### `vault`

A folder of Markdown notes (an Obsidian vault, say).

| Key | Default | Meaning |
| --- | --- | --- |
| `root` | `data/vault/` | The vault directory. Your real vault is reached only by naming it here. |
| `allow` | whole root | Glob subtrees the agent may use. |
| `deny` | none | Glob subtrees it may not. Deny wins over allow. |

### `workspace`

The `<workspace>` block the model sees every turn: where each tool family's
relative paths start, and the user's time zone.

| Key | Default | Meaning |
| --- | --- | --- |
| `expose_paths` | `true` | `false` hides host paths (your home directory, the tool roots) from the model. Use it for locked-down agents. |
| `notes` | `[]` | Extra lines added to the block (for example, "Projects live under ~/projects"). |

---

## Execution

### `shell`

`run_command` (foreground) and `start_command` (background).

| Key | Default | Meaning |
| --- | --- | --- |
| `cwd` | repository root | Where commands run. |
| `executor` | `backend: local` | Where commands execute: on the host, or in a Docker container. See [`executor`](#executor). |
| `command_policy` | none | What happens to each command before it runs: allow, ask for approval, or block. See [Shell command policy](security/shell-command-policy.md). |

With no `command_policy`, the shell falls back to a first-word deny-list:
`rm rmdir mkfs dd format shutdown reboot halt poweroff init`. That list only
checks the first word, so `ls; rm x` gets through it. Use `command_policy`
instead. A `command_policy` replaces the deny-list.

### `python`

`run_python`: model-written Python, run in a throwaway Docker container.

| Key | Default | Meaning |
| --- | --- | --- |
| `cwd` | repository root | The workspace mounted into the container. |
| `allow_unsandboxed` | `false` | `true` permits `executor.backend: local` (running on the host). Leave it off. |
| `default_timeout` | `30` | Seconds a call may run when the model doesn't ask for more. |
| `max_timeout` | `120` | The most a call may ask for. |
| `executor` | see below | Overrides the strict defaults, key by key. |

`run_python`'s executor starts from the strictest settings: `backend: docker`,
`mode: run` (a fresh container per call), `workspace_readonly: true`, a 64 MB
non-executable `/tmp`, and `max_writable_size: 100m`.

### `executor`

Used under `shell:` and `python:`. Docker keys are ignored with `backend: local`.
Asking for Docker on a host without it **fails at build time**; it never falls
back to running on the host.

| Key | Default | Meaning |
| --- | --- | --- |
| `backend` | `local` (shell), `docker` (python) | `local` runs on the host; `docker` runs in a locked-down container. |
| `mode` | `exec` (shell), `run` (python) | `exec`: one long-lived container, state persists between commands. `run`: a fresh container per command. |
| `image` | `python:3.12-slim` | Container image. |
| `network` | `none` | Docker network. `none` means no network at all. |
| `user` | your uid:gid | The container user. Never root. |
| `read_only` | `true` | Read-only root filesystem. |
| `memory` | `1g` | Memory limit. |
| `cpus` | `1.0` | CPU limit. |
| `pids_limit` | `256` | Process limit (stops fork bombs). |
| `workspace` | `/workspace` | Where `cwd` is mounted inside the container. |
| `workspace_readonly` | `false` | Mount the workspace read-only. |
| `writable` | `[]` | Workspace subpaths that stay writable when the workspace is read-only (for example `[out]`). |
| `max_writable_size` | none | Cap on what a call may leave in `writable` paths (`100m`, `1g`, or bytes). Needs `mode: run`. |
| `tmpfs` | `[]` | Extra tmpfs mounts (`/tmp:rw,size=64m,noexec`). |
| `exclude` | [secret files](security/sandbox.md#secret-files) | Files blanked inside the container: the real file stays on the host and an empty one is mounted over it. |

See [Sandbox](security/sandbox.md).

---

## Security

Each of these needs its capability in the profile; today only `secure` has them.

### `gating`

Per-call allow / approve / block rules for any tool, optionally conditioned on
a path argument. See [Tool gating](security/tool-gating.md).

### `guardrails`

| Key | Meaning |
| --- | --- |
| `injection` | Prompt-injection screening of tool results. See [Prompt injection](security/prompt-injection.md). |

### `secrets`

| Key | Default | Meaning |
| --- | --- | --- |
| `allowed` | unrestricted | Environment variables a `$SECRET:NAME` handle may resolve. `[]` resolves nothing. Absent means any variable may be named (the legacy behaviour). |

See [Secrets](security/secrets.md).

---

## Memory and recall

### `memory`

| Key | Default | Meaning |
| --- | --- | --- |
| `consolidate_every` | `30` | Minutes between background consolidation passes (turn count is the active trigger: every 5 turns). |
| `max_memory_size` | `65536` | Characters of `MEMORY.md` injected per request, at most. |
| `tier_caps` | `{t1: 75, t2: 25, t3: 15}` | How many records of each tier the injected index shows. `null` means unbounded. |
| `pairwise_review` | `true` | Run the pairwise reviewer (merge duplicates, apply updates) after each pass. |
| `render` | tiered | `mode: scoped` switches to the scoped render, with token budgets `global_tokens` (2000), `project_tokens` (2500), `history_tokens` (1000), `mentioned_tokens` (300), `history_days` (30), `mentioned_max` (5), `redundancy` (0.8). |

### `search`

`search_conversation_history`: BM25 over the full transcript.

| Key | Default | Meaning |
| --- | --- | --- |
| `max_matches` | `10` | Hits returned. |
| `context_lines` | `5` | Lines of context around each hit. |
| `bm25_k1` | `1.5` | BM25 term-frequency saturation. |
| `bm25_b` | `0.75` | BM25 length normalisation. |
| `scope` | `conversation` | `conversation` searches this conversation; `all` searches every conversation in the run store. |

### `keywords`

Relevance hints: short pointers to matching skills and notes, injected per turn
with no model call.

| Key | Default | Meaning |
| --- | --- | --- |
| `inject` | `true` | `false` builds the capability but injects nothing. |
| `top_k` | `3` | Most hints per turn. |
| `min_score` | `2.5` | Score floor for a hint. |
| `query_turns` | `2` | Recent user prompts the match reads. |
| `curate` | `false` | Maintain the vocabulary in the background. This sends skill and note digests to the model provider, so it's off unless you opt in. |
| `curate_limit` | `25` | Sources curated per pass. |
| `curate_cooldown_s` | `300` | Minimum seconds between passes. |

---

## Context

### `compact`

Keeps long conversations within budget. Compaction trims at turn boundaries and
never cuts into the turn in progress.

| Key | Default | Meaning |
| --- | --- | --- |
| `target_tokens` | `4000` | Live history size that triggers the emergency mid-turn ladder. Set high (a large share of the context window) for live use. |
| `turn_boundary` | `true` | Trim at the start of each turn, so the prompt prefix stays stable (and cached) for the whole turn. `false` is the old mid-turn behaviour. |
| `turn_boundary_target` | `target_tokens` | Routine budget for the turn-boundary trim. |
| `window_max_messages` | `10` | Messages examined before the oldest are evicted. |
| `window_keep_messages` | `20` | Recent messages kept after eviction. |
| `clear_tool_max_tokens` | `500` | Above this size, an old tool result may be collapsed to a placeholder. |
| `clear_tool_keep_pairs` | `4` | Most-recent tool call/result pairs that are never cleared. |
| `min_clear_tokens` | none | Skip a clear that reclaims fewer tokens than this. Clearing rewrites history and invalidates the prompt cache from that point, so a tiny reclamation costs more than it saves. |
| `clamp_max_part_tokens` | `50000` | A single message part larger than this is cut to head and tail. |
| `clamp_head_chars`, `clamp_tail_chars` | `2000` each | What a clamp keeps. |

---

## Web and vision

### `web`

| Key | Default | Meaning |
| --- | --- | --- |
| `search_domains` | a list of documentation sites | Domains `fetch_web_page` may pull into context. Search itself is never restricted. |

### `browser`

| Key | Default | Meaning |
| --- | --- | --- |
| `allowed_domains` | none | Hosts (and their subdomains) the browser may reach. |
| `block_private_addresses` | `true` | Refuse loopback, private and link-local addresses even when allowlisted. `false` lets an allowlisted private host through. Cloud metadata (`169.254.169.254`) is always refused. |

### `visual`

| Key | Default | Meaning |
| --- | --- | --- |
| `model` | none | The vision model that describes images. Absent means images go to the main model unchanged. A `visual:` block without `model` is an error. |
| `model_settings` | none | Settings for the vision calls. |

---

## Orchestration

### `subagents`

| Key | Default | Meaning |
| --- | --- | --- |
| `agents` | `[writer_subagent]` | Leaf specs the agent may delegate to. |
| `max_concurrent` | `5` | Background tasks per conversation. |

### `skills`

| Key | Default | Meaning |
| --- | --- | --- |
| `libraries` | `data/skills/` | Directories scanned for `SKILL.md` files. |

### `mcp`

| Key | Default | Meaning |
| --- | --- | --- |
| `config` | none | Path to a Claude-Desktop-style `mcpServers` JSON. Servers and their credentials live there, not in the spec. Needs the `demo` profile. |

---

## Benchmark prompts

Read only by the benchmark harness: `search_only_instructions`,
`search_only_no_readturn_instructions`, `thread_summary_instructions`. Each
replaces or extends `instructions` for one experimental arm. Background-agent
specs also use `retries` and `max_item_chars`.
