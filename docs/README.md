# paf documentation

Start with the [README](../README.md) for what paf is and how to run it.

- **[Spec reference](spec-reference.md)**: every key an agent spec can set,
  with its default.
- **[Security](security/README.md)**: how the defenses fit together and how to
  choose a setup.
  - [Approvals](security/approvals.md): a person says yes or no to one call.
  - [Shell command policy](security/shell-command-policy.md): allow, ask or
    refuse each shell command.
  - [Tool gating](security/tool-gating.md): allow, ask or refuse each tool
    call, by path.
  - [Prompt injection](security/prompt-injection.md): screening tool results,
    and tightening the run after a hit.
  - [Secrets](security/secrets.md): handles in, values never.
  - [Sandbox](security/sandbox.md): Docker, `run_python`, credential files,
    and the file tools.
