# Secrets

Credentials should never enter the model's context. Anything in the context
can end up in a reply, a log, a trace, or a tool call an injection talked the
model into. The `Secrets` capability keeps them out in two directions.

Needs `Secrets` in the profile (`secure` has it).

## Handles in, values never

The model refers to a credential by name, as a handle: `$SECRET:GITHUB_TOKEN`.
When a tool runs, the runtime swaps the handle in its arguments for the real
value, read from the environment. The model proposed the handle and never sees
the value.

```yaml
secrets:
  allowed: [GITHUB_TOKEN]   # the only variables a handle may name
```

| `allowed` | Effect |
| --- | --- |
| a list | Only these variables resolve. A handle naming anything else gets "not available", which doesn't reveal whether the variable is set. |
| `[]` | Nothing resolves. `secure_assistant` uses this. |
| absent | Any environment variable may be named. That includes your model API keys, so set a list for anything but personal use. |

## Values out

Every tool result is scrubbed before the model sees it. A value from your
environment is replaced when it is long enough to be meaningful (8 characters
or more), isn't a filesystem path, and either:

- its variable name contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`,
  `CREDENTIAL`, `PRIVATE`, `COOKIE`, `DSN`, `DATABASE_URL` or
  `CONNECTION_STRING`; or
- the value has a credential's shape: `sk-…`, `ghp_…`, `github_pat_…`,
  `xoxb-…`, `AKIA…`, `AIza…`, or a JWT (`eyJ…`).

Allowlisted variables are replaced by their `$SECRET:NAME` handle, so the model
can still use them. Everything else becomes `[redacted]`. Ordinary
configuration (`HOME`, `PATH`, `USER`, `SHELL`) is left alone: scrubbing it
corrupts output and gains nothing.

The same scrubbing is applied to approval requests and the approval audit log,
so a credential in a tool's arguments isn't shown or stored.

## Limits

- **Only values from the environment are known.** A credential that lives only
  in a file the model reads is not recognised by value. Keep credential files
  out of reach instead: `filesystem.deny_secret_files` and the sandbox's
  blanked files ([Sandbox](sandbox.md#secret-files)).
- **A tool can still misuse a value.** A handle resolved into `curl` is sent
  wherever `curl` sends it. `allowed` limits which secrets the agent can use,
  not what it does with them.
