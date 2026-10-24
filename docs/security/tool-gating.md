# Tool gating

The profile decides which tools exist. Gating decides, per call, whether a
call to one of them **runs**, **asks** you ([Approvals](approvals.md)), or is
**refused**, optionally depending on the path it touches. That covers what the
profile can't express: "writing is fine under `out/`, anywhere else needs a
person".

Needs `ToolGating` in the profile (`secure` has it).

## Example

From `secure_assistant`: read anything, write freely under `out/`, ask before
writing anywhere else, refuse everything not listed.

```yaml
gating:
  default: block
  rules:
    - tools: [read_file, list_directory, search_files, find_files, file_info]
      action: allow
    - tools: [run_python]
      action: allow
    - tools: [write_file, edit_file, create_directory]
      action: allow
      inside: [out]
    - tools: [write_file, edit_file, create_directory]
      action: approve
    - tools: [search_conversation_history, read_turn]
      action: allow
```

`default: block` makes the rule list an allowlist: a tool added to the profile
later stays refused until a rule names it.

## Keys

| Key | Default | Meaning |
| --- | --- | --- |
| `default` | `allow` | Decision for a call no rule matches. |
| `rules` | `[]` | Ordered; the **first matching rule** decides. |

| Rule key | Meaning |
| --- | --- |
| `tools` | **Required.** Tool names this rule covers (exact names). |
| `action` | **Required.** `allow`, `approve` or `block`. |
| `inside` | Match only when the path is inside one of these directories (relative to the tool's root). |
| `outside` | Match only when the path is **not** inside any of these. |
| `path_arg` | The argument holding the path. Default `path`. |
| `message` | What the model is told when the call is refused, instead of the default. |

A rule with neither `inside` nor `outside` matches every call to its tools.

**Ambiguity fails closed.** A path the policy can't interpret (missing, not a
string, absolute, starting with `~`, containing `..` or a backslash) counts as
outside every directory. So `inside: [out]` never matches `../out/x` or
`/etc/out`.

A malformed block makes the spec **fail to load**.

## Approve without a person

With nobody attached to the turn (a piped run, a scheduled task), `approve` is
enforced as `block`, with a message telling the model approval wasn't
available and not to retry.

## Without a `gating:` block

If the profile includes `ToolGating` but the spec has no `gating:` block, a
built-in list applies. That list is empty today, so every call is allowed.

## Gating and the command policy

Gating sees a tool call; it can't see inside a shell command. For the shell,
use the [command policy](shell-command-policy.md), which understands the
command. The two compose, and the stricter one wins: a gating `block` on
`run_command` refuses every command, whatever the command policy says.
