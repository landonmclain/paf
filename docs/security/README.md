# Security

paf assumes the model will sometimes be wrong, and that some of what it reads
was written by someone trying to steer it. The defenses are layered so that no
single one has to be perfect, and each is enforced by the runtime, outside the
model, where the model cannot talk its way past it.

| Layer | Stops | Configured by | Page |
| --- | --- | --- | --- |
| **Profile** | The agent having a tool at all | `profile:` | [spec reference](../spec-reference.md#profile) |
| **Sandbox** | Code touching the host, the network, or your credentials | `shell.executor`, `python.executor` | [Sandbox](sandbox.md) |
| **Scoped file tools** | Reads and writes outside one directory; reading credential files | `filesystem` | [Sandbox](sandbox.md#host-side-file-tools) |
| **Shell command policy** | Irreversible shell commands running without a person agreeing | `shell.command_policy` | [Shell command policy](shell-command-policy.md) |
| **Tool gating** | A tool call that is fine in one place and not another (writes outside `out/`) | `gating` | [Tool gating](tool-gating.md) |
| **Prompt-injection screening** | Injected instructions in tool results reaching the model, and the model acting on them | `guardrails.injection` | [Prompt injection](prompt-injection.md) |
| **Secret handles** | Credentials entering the model's context | `secrets` | [Secrets](secrets.md) |
| **Approvals** | All of the above being all-or-nothing: a person can say yes to one call | `approvals` | [Approvals](approvals.md) |

## How they fit together

**Refuse what isn't needed, ask about what's risky, contain the rest.**

1. The profile decides which capabilities exist. A tool the profile leaves out
   is never built, so no prompt can reach it.
2. Before a call runs, the policies decide: **allow**, **approve** (ask the
   person), or **block**. The policies are tool gating (any tool) and the
   command policy (shell commands).
3. What runs is contained. Programs and shell commands can run in a Docker
   container with no network, a read-only root and resource limits. The file
   tools are confined to their root, and credential files are blanked or
   refused.
4. What comes back is screened. Tool results are scored for prompt injection.
   A hit can be withheld, redacted, or delivered marked as untrusted. A hit also
   tightens the rest of the run: listed tools then need approval.
5. Credentials never enter the context. The model refers to them by
   `$SECRET:NAME` and the runtime substitutes the value at execution time.
   Credential-looking values in tool output are scrubbed.

Every **approve** goes to whoever is attached to the turn: the terminal running
`paf chat`, or the browser showing it. With nobody attached (a piped run, a
scheduled task, a background turn), approve means **refused**. Nothing waits on
a question no one will see.

## Choosing a setup

**Personal use, lenient** (`demo_assistant`): every capability, shell on the
host, and a command policy that asks before anything irreversible (deletes,
overwrites, discarded git work, force pushes, destructive database statements).
You stay in the loop for what you can't undo and aren't interrupted otherwise.

**High stakes, locked down** (`secure_assistant`, `profile: secure`): no shell,
no network, no memory. File tools are confined to one workspace with credential
files denied. Python runs only in a throwaway container. Writes outside `out/`
need approval. Injection screening redacts hits and requires approval for
writes after one. Secret handles resolve nothing. Host paths are hidden from
the model. Needs Docker.

## What this does not do

- **It is not a guarantee against a determined attacker with shell access.**
  The command policy reads command *text*. It can't see inside scripts,
  `make` targets, aliases or `eval`, and it flags those rather than
  understanding them. For untrusted work, use the Docker sandbox and leave the
  shell out of the profile, as `secure` does.
- **Injection screening is a classifier.** It misses some attacks and flags
  some harmless text. The tightening step (approval after a hit) is what limits
  the damage of a miss elsewhere in the run.
- **`backend: local` is not a sandbox.** Commands run as you, with your
  network and your files.

The policies are checked by tests under `tests/` (`test_shell_command_policy.py`,
`test_approvals.py`, `test_secure_profile.py`, the injection tests), and the
injection scorer's threshold comes from a calibration eval in
`tests/defender_eval/`.
