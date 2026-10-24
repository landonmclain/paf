# Approvals

Some calls should run only if a person agrees: deleting a directory, writing
outside the agent's output folder, writing anything after the run has read
something that looks like a prompt injection. Approvals let a policy say
**approve** instead of choosing between always allowing and always refusing.

The model calls the tool normally. The runtime pauses that call, shows it to the
person, and runs it only on a yes. The model is told not to dodge the prompt
with a workaround; the workarounds ask too.

## What asks

Three policies can return **approve**:

- [Shell command policy](shell-command-policy.md): a command that matches an
  `approve` rule, such as `rm -rf build`.
- [Tool gating](tool-gating.md): a call that matches an `approve` rule, such as
  `write_file` outside `out/`.
- [Prompt injection](prompt-injection.md): after a hit, any tool listed under
  `on_flag.require_approval`, for the rest of the run.

## Who is asked

The person attached to the turn.

| Where the turn runs | How it asks |
| --- | --- |
| `paf chat` in a terminal | A prompt under the tool call. `y` approves. `n` or Enter denies. `n <text>` denies with a note the agent sees, for example `n use the trash instead`. |
| The web UI (paf-ui over `ui/sse.py`) | A card in the chat with Approve / Deny and an optional note. The card survives a page reload: a client that reconnects mid-turn fetches what is pending. |
| Piped `paf chat`, scheduled tasks, background turns | Nobody is attached, so the call is **refused** with a message telling the model so. |

Each request shows:

- the tool and its arguments, with secrets scrubbed;
- why it needs approval, in words (the rule's `why`, or a description such as
  "deletes files");
- a preview when one can be worked out without running anything, for example
  what `rm -rf build dist` would delete (file count, size, how many files git
  tracks, paths outside the working directory), or what an overwrite would
  replace;
- the time left before it expires.

## How it resolves

- **First decision wins.** The terminal and the browser can both see a request.
  Whichever answers first settles it, and a later answer gets a conflict.
- **Timeout.** A request nobody answers is denied after `approvals.timeout`
  seconds (default 600, ten minutes).
- **Stop.** Stopping the turn cancels its pending requests; those calls don't
  run.
- **Denied calls never run.** The model sees that the call was denied, plus
  your note if you left one, and is told not to retry it or work around it.

| Outcome | Meaning |
| --- | --- |
| `approved` | A person said yes; the call ran. |
| `denied` | A person said no. |
| `timeout` | Nobody answered in time. |
| `cancelled` | The turn was stopped. |
| `unattended` | Nobody was attached to ask. |

## Configuration

```yaml
approvals:
  timeout: 600   # seconds before an unanswered request is denied
```

Nothing else turns approvals on. The approval broker is always installed, and a
request only happens when one of the policies above returns **approve**.

## Audit trail

Every request and every decision is appended to
`data/approvals/<agent>.jsonl`: the tool, its scrubbed arguments, the reason,
the outcome, who decided (`cli`, `web` or `system`), the note, and timestamps.
Requests and decisions are also traced as `paf.approval.request` and
`paf.approval.decision` spans.

## HTTP API

The SSE server exposes the approval flow to any front end:

| | |
| --- | --- |
| `approval_requested` frame | Sent on the turn's stream: `approval_id`, `step_id` (the tool call), `tool_name`, `args`, `reason`, `preview`, `created_at`, `expires_at`, `conversation_id`. |
| `approval_resolved` frame | `approval_id`, `step_id`, `tool_name`, `outcome`, `actor`, `note`. |
| `GET /api/conversations/{id}/approvals?agent=` | What is pending for a conversation. Used to rebuild cards after a reload. |
| `POST /api/approvals/{approval_id}` | Body `{"decision": "approve" \| "deny", "note"?: str}`. Returns 404 when nothing with that id is pending, 409 when someone else already decided, 422 for a bad body. |

The SSE server has no authentication of its own. It is meant to be reached over
a private network (paf's deployment puts it behind Tailscale).

## Limits

- **One call at a time.** There is no "approve everything like this for the
  rest of the turn" yet.
- **History.** Reopening an old conversation shows the tool calls but not which
  ones were approved or denied; the audit file has that.
- **Subagents.** Routing a subagent's approval requests to the person is not
  built yet. Plan for a subagent's approve calls to be refused.
