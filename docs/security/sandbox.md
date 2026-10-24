# Sandbox

Two things in paf run code: the shell (`run_command`, `start_command`) and
`run_python`. Either can run on the host or in a locked-down Docker container.
The file tools always run on the host, confined to one directory.

## Docker

```yaml
shell:
  cwd: ~/work
  executor:
    backend: docker
    mode: exec          # or run: a fresh container per command
```

What the container gets by default:

| | |
| --- | --- |
| Network | none (`network: none`) |
| Root filesystem | read-only |
| User | your own uid:gid, never root |
| Limits | 1 GB memory, 1 CPU, 256 processes |
| Workspace | `cwd`, mounted at `/workspace` |
| Docker socket | never mounted: a sandboxed `docker` command can't reach the host daemon |
| Credential files | blanked (see [below](#secret-files)) |

Every key is in the [spec reference](../spec-reference.md#executor).

**Asking for Docker on a host without it fails at build time.** It never falls
back to running on the host, so a missing daemon can't quietly remove the
sandbox.

**`exec` or `run`.** `exec` keeps one container for the session (fast, and
state such as installed packages persists between commands). `run` starts a
fresh container per command: slower, nothing carries over, and it's required
for `max_writable_size`.

**Cleanup.** Containers are labelled `paf.sandbox` with the owning process.
Containers left by a process that has exited are removed automatically.

## `run_python`

Stricter by default than the shell:

- Docker, always. The tool refuses to run on the host unless the spec sets
  `python.allow_unsandboxed: true`.
- `mode: run`: a fresh container per call.
- Read-only workspace, with a 64 MB non-executable `/tmp`.
- Writable output only where you open it, capped in size:

```yaml
python:
  cwd: ~/paf-secure-workspace
  executor:
    writable: [out]          # programs may write here, nowhere else
    max_writable_size: 100m
```

### Writable paths

A writable path is the one channel from sandboxed code back to the host, and
host-side tools read it afterwards. So after every call:

1. Anything that isn't a regular file or directory (symlinks, FIFOs, sockets,
   devices) is removed. A symlink could point where the sandbox couldn't reach;
   a FIFO would hang a later read.
2. If the total is over `max_writable_size`, files written **by that call** are
   deleted, largest first, until it fits. Earlier files are never deleted for
   being over.

The model is told what was removed.

## Secret files

One list of credential files, enforced on both sides of the boundary:

- **In Docker**, each is blanked: the real file stays on the host and an empty
  one is mounted over it inside the container.
- **Through the file tools** (`filesystem.deny_secret_files: true`), each is
  refused for read, write, list and search.

Matched at any depth (`backend/.env` counts as much as `.env`):

- **Files:** `.env`, `.env.*`, `*.env`, `.git-credentials`, `.netrc`, `.npmrc`,
  `.pypirc`, `.pgpass`, `id_rsa`, `id_dsa`, `id_ecdsa`, `id_ed25519`, `*.pem`,
  `*.key`, `*.p12`, `*.pfx`, `credentials.json`, `secrets.yaml`, `secrets.yml`,
  `secrets.json`, `secrets.toml`
- **Directories:** `.git`, `.ssh`, `.aws`, `.gnupg`

`.env.example` is covered too (it matches `.env.*`). The walk skips
`node_modules`, `.venv`, `__pycache__` and similar vendored trees, where `*.pem`
files are legitimate CA bundles.

The model is told that blanked or refused files tell it nothing about their
contents, so it doesn't report a real `.env` as "empty".

## Host-side file tools

The file tools run on the host. The Docker overlay doesn't cover them; their
own rules do:

- Confined to `filesystem.root_dir`: no `..`, no escape through symlinks.
- `.git/`, `.env` and key files are read-only even without `deny_secret_files`.
- `denied_patterns` adds anything else you want refused.
- `workspace.expose_paths: false` hides the host paths (your home directory,
  the tool roots) from the model entirely.

## `backend: local`

`local` is the default for the shell, and it is **not a sandbox**: commands run
as you, with your network, your files and your environment. Pair it with a
[command policy](shell-command-policy.md) for personal use, and use Docker (or
no shell at all) for anything you don't trust.
