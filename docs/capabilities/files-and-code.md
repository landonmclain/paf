# Files and code

How an agent reads and changes files, runs commands, works with git, browses a
codebase, and keeps notes.

| Capability | Tools | Spec block |
| --- | --- | --- |
| [FileSystem](#filesystem) | `read_file`, `write_file`, `edit_file`, `list_directory`, `search_files`, `find_files`, `create_directory`, `file_info` | `filesystem` |
| [Shell](#shell) | `run_command`, `start_command`, `check_command`, `stop_command` | `shell` |
| [Python](#python) | `run_python` | `python` |
| [Git](#git) | `git_status`, `git_changed_files`, `git_diff`, `git_checkpoint`, `git_restore_paths` | `git` |
| [CodebaseMap](#codebasemap) | `get_overview`, `lookup_file`, `read_block`, `search_codebase`, `run_codebase_map` | `codebase_map` |
| [Vault](#vault) | `vault_overview`, `lookup_note`, `read_note`, `note_find`, `note_search`, `note_links`, `note_backlinks`, `note_create`, `note_rename`, `note_edit`, `note_append` | `vault` |
| [Workspace](#workspace) | none: adds a `<workspace>` block to the instructions | `workspace` |

Each tool family has its own root, and relative paths mean different things to
each. The Workspace block tells the model which is which.

## FileSystem

Read, write, edit and search files under one root directory.

- **`read_file(path, offset?, limit?)`** returns numbered lines, at most 2000
  per call.
- **`write_file(path, content, expected_hash?)`** creates or overwrites.
- **`edit_file(path, edits, expected_hash?)`** applies `{old_text, new_text}`
  pairs in one atomic write. Each `old_text` must appear exactly once, so the
  model includes surrounding lines to make it unique. A failed edit is an
  error, never a silent no-op.
- **`list_directory`**, **`find_files(pattern)`** (glob),
  **`search_files(pattern, include_glob?)`** (regex over contents),
  **`create_directory`**, **`file_info`**.

`expected_hash` is optimistic concurrency: pass the hash from the last read and
the write is rejected if the file changed since. Searches and globs have a time
budget, so a pattern over a huge tree stops instead of wedging the run.

**Confinement.** Paths resolve under `filesystem.root_dir`. Traversal above it
is rejected, and symlinks are resolved before the check. `.git/`, `.env`,
`.env.*`, `*.pem`, `*.key` and `**/secrets*` are **read-only**. Add
`deny_secret_files` to refuse reading credential files too, and
`denied_patterns` for anything else. See
[Sandbox](../security/sandbox.md#host-side-file-tools).

## Shell

Runs commands in a subprocess rooted at `shell.cwd`, on the host or in a Docker
container (`shell.executor`).

- **`run_command(command, timeout_seconds?)`** runs in the foreground. The
  default timeout is 30 seconds. Output is labelled stdout/stderr with the exit
  code, and cut at 50,000 characters.
- **`start_command(command, notify_on_completion?)`** starts a long-running
  command (a server, a watcher, a build) detached, and returns an id. When it
  finishes, a message arrives in the conversation with its exit code and the
  end of its output, so the model need not poll.
- **`check_command(command_id, offset?)`** returns status and output, by
  default the last part (progress, the error). `offset` pages from a position.
- **`stop_command(command_id)`** stops it and returns the final output.

Background commands are recorded under `data/processes/<agent>/`, so they are
found again after a turn ends or the server restarts. `/tasks` in `paf chat`
lists them.

Each command starts in `cwd`; a `cd` doesn't carry to the next call. Commands
that need a terminal (`vi`, `nano`, `ssh`, …) are refused.

**What decides whether a command runs.** With a `command_policy`, each command
is read and allowed, sent for approval, or refused
([Shell command policy](../security/shell-command-policy.md)). Without one, a
first-word deny-list applies (`rm`, `dd`, `shutdown`, …). Pair either with the
Docker backend for untrusted work.

**Environment.** On the host, commands inherit your whole environment,
including any API keys in it. If the agent runs commands you don't fully
trust, use the Docker backend, which has no network and none of your files
beyond `cwd`.

## Python

`run_python` runs model-written Python in a fresh Docker container per call:
no network, read-only workspace, resource limits, output only to paths you open.
It is the safe way to give an agent computation without a shell. It refuses to
run on the host unless the spec says `allow_unsandboxed: true`.

Details are in [Sandbox](../security/sandbox.md#run_python).

## Git

Safe, root-confined git. The tools are deliberately narrow: they inspect, and
they write only what is named.

- **`git_status`** (short form), **`git_changed_files`** (tracked changes, no
  diff), **`git_diff(paths?, start_line?, max_lines?)`** (unstaged, in numbered
  chunks).
- **`git_checkpoint(message, paths)`** commits **explicit paths only**. It
  refuses an empty list and never stages everything, so unrelated local changes
  can't be swept into a commit.
- **`git_restore_paths(paths)`** discards unstaged changes for explicit paths.
  It's destructive, and its description tells the model to use it only when
  you ask to revert.

There is no push, pull, branch or reset. For those, the model needs the shell,
where the [command policy](../security/shell-command-policy.md) asks you before
the risky ones.

## CodebaseMap

Browsing a codebase by **navigation** instead of search: the agent orients,
then narrows, then reads only the lines it needs. It is built for large repos,
where reading whole files wastes context.

```
get_overview ──▶ lookup_file ──▶ read_block ("path:L1-L2")
   orient          head → detail      the lines, and nothing more
                         ╲
                  search_codebase   an escape hatch, after orienting
```

- **`get_overview(path?, depth?)`**: the folder tree with file counts and line
  counts. Noise folders (tests, docs, vendored code, CI) fold into one summary
  line and open on request. When a call graph was built, files carry an
  importance score, so hubs and orchestrators stand out.
- **`lookup_file(path, symbol?, detail?)`**: a file's head (docstring and symbol
  names), then its symbols' signatures and spans, then one symbol's body. Never
  a whole large file.
- **`read_block("path")` or `"path:L1-L2"`**: the only tool that returns source.
  Capped at 64 KB, and returns line numbers, so the model can say exactly which
  lines matter.
- **`search_codebase(query, limit?)`**: ranked symbol search (name, then
  signature, then docstring). Each hit comes with a ready `read_block` pointer.
- **`run_codebase_map`**: rebuild the index.

The index lives in paf’s `data/.paf-codebase-index/<sha256(root)>/`, outside
the scanned repository, and is rebuilt when files change. Each scan root gets
its own directory. Old `.agent-codeindex/` directories are left untouched and
excluded from scans; the first query rebuilds the index at the new location.
The walk follows `.gitignore` and works for any language the extractors can
parse; other files are listed but unparsed.

## Vault

A folder of Markdown notes (an Obsidian vault, for example), navigated the same
way: outline first, then a note's head, then its body.

- **Reading:** `vault_overview(depth?)`, `lookup_note` (frontmatter, headings,
  links out), `read_note` (numbered lines plus a `version`), `note_find` (by
  title, tag, folder), `note_search` (text).
- **The graph:** `note_links`, `note_backlinks`.
- **Writing:** `note_create`, `note_append`, `note_edit` (exact-text pairs, with
  `expected_version` so a stale edit is rejected), and `note_rename`, which
  rewrites every `[[wikilink]]` that pointed at the old name.

A note can be named by path, by title or by `[[wikilink]]`. The vault is
confined to `vault.root`, narrowed by `allow` and `deny` subtrees (deny wins),
so you can give an agent your work notes and keep your journal out of reach.
The default root is a paf-owned scratch vault, never your real one.

## Workspace

Without help, a model reads `ui/chat.py` from the codebase map and hands it to a
file tool rooted somewhere else. Workspace adds one static `<workspace>` block
to the instructions: where each tool family's relative paths start, built from
the roots the other capabilities were actually given, plus your time zone and
any `workspace.notes`. The block is static for each built agent. When a project
root changes, the session rebuilds it from the effective tool roots on the next
turn; otherwise the block remains stable for prompt caching.

`workspace.expose_paths: false` hides the real paths: every root is described
as "the workspace root", your home directory isn't named, and the model is told
to use relative paths. Locked-down agents use it, which also keeps the text true
when a sandbox mounts the workspace somewhere other than its host path.

When `workspace.project_roots.within` is enabled, a conversation's project root
supplies the common root for file tools, shell, Python, git, and codebase map.
Vault and skill-library paths stay per agent. See the
[workspace configuration](../spec-reference.md#workspace) and
[project commands](conversations-and-memory.md#projects).
