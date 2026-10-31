# Security policy

## Reporting a vulnerability

Please report security issues privately, not in a public issue.

Use GitHub's private reporting: **Security → Report a vulnerability** on
[this repository](https://github.com/landonmclain/paf/security/advisories/new).
Include what you found, how to reproduce it, and the version or commit.

I'll acknowledge a report within a few days and keep you updated while it is
fixed. This is a one-person project, so please allow for that.

## What counts

paf runs model-written code and reads untrusted content, so the interesting
reports are failures of the defenses described in [docs/security](docs/security/README.md):

- Escaping the Docker sandbox (`run_python`, or `shell` with `executor: docker`),
  or reaching the host, network or credentials from inside it.
- Reading or writing outside a configured root with the file tools, including
  through symlinks or `..`.
- Bypassing the shell command policy, tool gating or approvals so an
  irreversible call runs without the check that should have applied.
- Reaching private, loopback or cloud-metadata addresses through `fetch_web_page`
  or the browser despite the SSRF guard, including through redirects.
- A credential or secret reaching the model's context, a trace or a log.
- Prompt-injection text in a tool result getting past the guard in a way the
  documented policy says it should not.

Ordinary model mistakes (a wrong answer, an unwise but allowed command) are not
vulnerabilities. Neither is anything that needs you to turn a guard off: some
are off by default and documented as such, such as running `shell` on the host.

## Supported versions

Only the latest commit on `main` is supported. There are no released versions yet.

## Running paf safely

- Keep `.env` out of version control (it is git-ignored) and out of any
  directory an agent can read; set `filesystem.deny_secret_files: true`.
- For untrusted input, use the `secure_assistant` spec: no shell, no network,
  sandboxed Python, every guard on. It needs Docker.
- Treat `shell` with `executor: local` as running as you.
