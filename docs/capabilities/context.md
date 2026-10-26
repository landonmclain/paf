# Context management

A long conversation outgrows the model's context window, and a big tool result
can fill it in one step. Three capabilities keep the window bounded without
losing what the agent needs.

| Capability | What it does | Spec block |
| --- | --- | --- |
| [Compaction](#compaction) | Trims old messages and old tool results | `compact` |
| [ToolOutputLimits](#tooloutputlimits) | Moves oversized tool results out of the window | none |
| [WarnOnCacheBusts](#warnoncachebusts) | Warns when the prompt cache stops working | none |

None of them summarises. A summary is a second, lossy copy of the conversation.
paf's long-term recall is instead the saved history
([ConversationSearch](conversations-and-memory.md#conversationsearch)), the
[memory](conversations-and-memory.md#memory) notebook and the
[thread summary](conversations-and-memory.md#threadsummarizer), which are
separate from the window. Compaction only ever *shrinks* the live context.

## Compaction

A ladder of three steps, tried in order, each stopping as soon as the context
fits its budget:

1. **Clamp** any single oversized message part to its head and tail.
2. **Clear** old tool results to a placeholder, keeping the most recent
   call/result pairs intact. Tool output is the usual bulk, and the old ones are
   the least useful.
3. **Slide** the window: drop the oldest *settled* messages.

**It never touches the turn in progress.** The window only evicts messages
before the latest real user prompt, so the model never loses part of the task it
is working on.

### When it runs

By default compaction runs at the **start of each turn** (`turn_boundary: true`),
trimming to `turn_boundary_target`. That keeps the prompt prefix identical for
the whole turn, which matters because providers cache the prefix: every edit to
earlier messages invalidates the cache from that point on, and you pay to
re-read it. A trim at the turn boundary costs one cache miss per turn instead of
one per request.

A second, emergency pass runs *during* a turn only if the context passes
`target_tokens` anyway.

### Tuning

```yaml
compact:
  turn_boundary: true
  turn_boundary_target: 100000   # routine budget at the start of each turn
  target_tokens: 200000          # emergency ceiling during a turn
  window_max_messages: 100
  window_keep_messages: 20
  clear_tool_max_tokens: 5000    # older tool results above this may be cleared
  clear_tool_keep_pairs: 6       # always keep this many recent pairs
  min_clear_tokens: 10000        # skip a clear that reclaims less than this
```

`min_clear_tokens` is the cache guard: clearing a small result rewrites history
(a cache miss) to save a few tokens, which costs more than it saves. Raise
`clear_tool_keep_pairs` for work that keeps referring back to recent results,
and lower it to compact less often.

The code's built-in defaults are tiny (a 4,000-token target) because they were
tuned for a benchmark that forces evidence out of context. Any agent used for
real work should set its own. Every key and default is in the
[spec reference](../spec-reference.md#compact).

## ToolOutputLimits

A single `cat` of a big file or a wide search can drop tens of thousands of
tokens into the window, and every later request re-sends them. ToolOutputLimits
catches results over **20,000 characters** before the model sees them:

- the full result is saved losslessly to a temporary file store;
- the model gets a truncated version, with the `read_tool_result` tool to page
  through the rest on demand.

It costs no model call and loses nothing. It's configured in code rather than
the spec.

## WarnOnCacheBusts

Prompt caching is the largest lever on cost and latency, and the easiest to
break by accident: a reordered tool list, a timestamp injected into the prefix,
or a provider-side expiry all make the model re-read tokens a stable prefix
would have served from cache.

This capability watches each response's `cache_read_tokens`. If an established
cached prefix (at least 1,024 tokens) drops below half its size mid-run, it
raises a `CacheBustWarning`, which shows up as a red span in traces. It only
observes: no tools, no instructions, no change to behaviour. The thresholds are
deliberately conservative, so ordinary partial misses stay quiet.
