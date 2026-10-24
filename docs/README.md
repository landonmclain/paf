# paf

**paf** is a small, opinionated agent framework built on
[pydantic-ai](https://ai.pydantic.dev). You describe an agent in a YAML spec,
choose which capabilities it may use, and paf assembles the agent and runs it
with durable conversations, long-term memory, context management, and
runtime-enforced security.

It ships with a terminal chat (`paf chat`), an SSE server for web front ends,
and a Python API (`Session`) that both are built on. The
[repository README](https://github.com/landonmclain/paf#readme) covers
installing and running it.

## Documentation

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
