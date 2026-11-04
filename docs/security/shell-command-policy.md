# Shell command policy

Decides, before a shell command runs, whether it **runs**, **asks** you
([Approvals](approvals.md)), or is **refused**. It's for agents that have a
shell and that you trust to use it, where some commands still deserve a second
look.

Without it, the shell has a first-word deny-list (`rm`, `dd`, `shutdown`, …).
The deny-list has two problems. It misses `ls; rm -rf x`, `sudo rm`,
`bash -c 'rm …'`, `xargs rm` and `find -delete`. And it teaches the model to
reach for those workarounds when `rm` is refused. The policy fixes both: it
understands the whole command, and a legitimate `rm` asks you rather than being
refused, so the model has no reason to dodge it.

## The lenient policy

This is what `demo_assistant` uses: anything irreversible asks, everything else
runs.

```yaml
shell:
  command_policy:
    default: allow
    unparseable: approve
    unknown_scripts: allow
    rules:
      - effects_any: [delete]
        targets_any: ["/", "/*", "~", "~/", "$HOME", "/home", "/etc", "/usr"]
        action: block
        why: deletes the root, your home, or a system directory
      - effects_any: [delete, discard, force_push, disk, overwrite, data]
        action: approve
      - commands: [chmod, chown]
        flags_any: [-R, --recursive]
        action: approve
```

| Command | Decision |
| --- | --- |
| `ls -la`, `pytest -q`, `git status`, `echo x > new.txt` | runs |
| `rm -rf build`, `ls; rm a.txt`, `sudo rm x`, `find . -name '*.pyc' -delete` | asks |
| `echo x > notes.txt` when `notes.txt` exists, `mv a b` over an existing `b` | asks |
| `git reset --hard`, `git push -f`, `git branch -D x` | asks |
| `psql -c 'DROP TABLE users'`, `redis-cli FLUSHALL` | asks |
| `python3 -c "import shutil; shutil.rmtree('x')"` | asks |
| `rm -rf ~`, `rm -rf /`, `sudo rm -rf /etc`, `find ~ -delete` | refused |

## How a command is read

To permit SSH only after approval, enable `allow_interactive` and add a command
rule to the shell policy:

```yaml
shell:
  allow_interactive: true
  command_policy:
    default: allow
    rules:
      - commands: [ssh]
        action: approve
        why: SSH requires approval for each invocation
```

Add that rule alongside your existing rules. The command policy checks both
`run_command` and `start_command`, including `/usr/bin/ssh` and supported wrappers
such as `env ssh` and `bash -c 'ssh …'`. Without `allow_interactive: true`, a plain
SSH command is still refused by the separate interactive-command check, even
after approval. Use individual remote commands (`ssh host 'uptime'`): approval
covers the whole tool call, not each command inside a remote session. This rule
does not inspect remote effects, enforce read-only access, or detect SSH hidden
inside script files. Restart `paf-sse` after changing a loaded agent's spec.

The command is parsed with the bash grammar (tree-sitter), not split on spaces.
Every simple command that would run is found and judged on its own:

- across `;`, `&&`, `||`, pipes, subshells, `$( )` and backticks;
- through wrappers: `sudo`, `env`, `nohup`, `time`, `timeout`, `nice`,
  `xargs`, `exec`, `command`;
- inside `bash -c '…'` / `sh -c '…'` strings and `find -exec …`.

The decision for the whole command is its strictest part: **block** beats
**approve**, and **approve** beats **allow**. So `echo hi && rm -rf ~` is
refused.

## Effects

Each simple command is tagged with what it does. Rules match on these tags, so
one rule covers every command with that effect, however it's spelled.

| Effect | Tagged on |
| --- | --- |
| `delete` | `rm`, `rmdir`, `unlink`, `shred`, `trash`; `find -delete`; `git rm`, `git clean`; `rsync --delete`; `docker`/`podman` `rm`, `rmi`, `prune`, `volume rm`, `compose down -v`; `crontab -r`; Python/Node/Perl/Ruby one-liners that call a delete API |
| `discard` | git work thrown away: `reset --hard`, `checkout -- .`, `restore`, `branch -D`, `stash drop`/`clear`, `tag -d` |
| `force_push` | `git push --force`, `-f`, `--force-with-lease`, `--mirror`, `--delete`, `+refspec`, `:ref` |
| `disk` | `dd`, `mkfs*`, `fdisk`, `parted`, `wipefs`, `mkswap` |
| `overwrite` | `>`, <code>&gt;&#124;</code>, `&>`, `mv`, `cp`, `install`, `ln -f`, `tee`, `truncate`, `rsync` onto a file that **already exists**. Writing a new file, appending (`>>`, `tee -a`) and `-n`/`--backup` don't count. |
| `data` | `DROP`, `TRUNCATE`, `DELETE FROM`, `UPDATE … SET`, `ALTER TABLE … DROP`, `FLUSHALL`/`FLUSHDB` given to `psql`, `mysql`, `sqlite3`, `duckdb`, `mongosh`, `redis-cli`, `clickhouse-client` |
| `permissions` | `chmod`, `chown`, `chgrp` |
| `privileged` | anything run through `sudo` or `doas` |
| `script` | something whose contents can't be seen: `./run.sh`, `bash deploy.sh`, `python cleanup.py`, `make`, `just`, `npm`/`yarn`/`pnpm`/`bun`/`uv`/`poetry` `run`, `psql -f file.sql` |

`overwrite` is the only effect that checks the disk. The policy keeps it only
when a target file exists. A destination computed at run time (`$VAR`) can't be
checked, so it counts.

## Policy keys

| Key | Default | Meaning |
| --- | --- | --- |
| `default` | `allow` | Decision for a command no rule matches. |
| `unparseable` | `approve` | Decision when the command can't be fully understood: a parse error, `eval`/`source`, or a command name computed at run time. |
| `unknown_scripts` | `allow` | Decision for a `script`-tagged command no rule matches. Set `approve` to be asked before any script, `make` target or `npm run`; expect a prompt on most builds and test runs. |
| `rules` | `[]` | Ordered; the **first matching rule** decides each simple command. |

## Rule keys

A rule matches when **all** of its conditions match. An omitted condition
matches everything.

| Key | Matches when |
| --- | --- |
| `commands` | The command name matches one of these globs (`rm`, `git`, `mkfs*`). |
| `subcommands` | The subcommand is one of these (`push` for `git push`). |
| `flags_any` | The command has any of these flags. Combined short flags are split, so `-rf` has `-r` and `-f`. |
| `effects_any` | The command has any of these [effects](#effects). |
| `targets_any` | A target path matches one of these patterns. A plain pattern (`/`, `~`, `$HOME`, `/etc`) matches that exact path, as written or resolved against the shell's directory. So `rm -rf .` run from your home directory hits `~`. A glob is matched one path component per `*`: `/*` is any top-level directory, never `/tmp/project/build`. |
| `action` | **Required.** `allow`, `approve` or `block`. |
| `why` | Shown to the person and the model instead of the generic description. |

Unknown keys, unknown effects and bad actions make the spec **fail to load**.

## `find`

For `find`, the starting directories count as what gets deleted only when
nothing narrows the match. `find ~ -name '*.pyc' -delete` searches your home
and asks. `find ~ -delete` and `find ~ -type f -delete` would empty it, so the
block rule refuses them. Negation (`!`, `-not`) and `-o` count as not narrowed,
because `! -name keep` matches nearly everything.

## What it can't see

The policy reads the command text. It is a seatbelt for mistakes, not a
sandbox.

- **Scripts and task runners.** What `./clean.sh` or `make clean` does is
  invisible; they're tagged `script` and follow `unknown_scripts`.
- **In-place edits.** `sed -i` and `perl -pi` aren't tagged. They are edits,
  like the `edit_file` tool.
- **Remote side effects.** `curl -X POST`, `ssh host 'rm …'`, `kubectl delete`
  and the like aren't recognised.
- **Variables, aliases and `eval`.** A command name computed at run time, or
  text run through `eval`, counts as `unparseable`.
- **The file tools.** `write_file` and `edit_file` can overwrite a file without
  going through this policy. To ask about those, use [Tool gating](tool-gating.md).

For work you don't trust, run the shell in the [Docker sandbox](sandbox.md), or
leave it out of the profile.

## Tracing

Every approve or block is traced as a `paf.shell.command_policy` span, with the
tool, the decision, the reasons and the command (secrets scrubbed).
