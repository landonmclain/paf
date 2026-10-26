# Conversations and memory

Four capabilities that give an agent a past: every turn is saved, searchable,
distilled into durable facts, and summarised. All of it is plain files under
`data/`, so you can read, grep and diff what the agent knows.

| Capability | Gives the model | Profiles |
| --- | --- | --- |
| **StepPersistence** | nothing visible: it saves every run, step by step | assistant, demo, secure |
| **ConversationSearch** | `search_conversation_history`, `read_turn` | assistant, demo, secure |
| **Memory** | an injected `<memory>` block, `memory_archive_suggestion`, and with the scoped render `search_memory`, `list_memory` | assistant, demo |
| **ThreadSummarizer** | an injected `<thread-summary>` block, `list_turn_summaries` | assistant, demo |

`secure` keeps persistence and search (the audit trail) and leaves out memory
and summaries: nothing carries over between conversations.

## StepPersistence

Writes every run to `data/paf/` as it happens: JSONL events plus a JSON
snapshot per run, one directory each. It is the single source for everything
else on this page. Search reads exactly what persistence wrote, and memory
facts point back into it.

## ConversationSearch

The model's way back into the past, in a fixed order: what is in context, then
`search_conversation_history`, then `read_turn`.

- **`search_conversation_history(query, run_id?)`** ranks persisted history
  with BM25. A multi-word query searches each word independently, and rare
  terms score higher. `run_id` narrows it to one run.
- **`read_turn(ref, full?)`** expands a hit (`run_id:turn_id`) into the whole
  turn, tool calls and results included. `full=true` returns it byte for byte.
  A memory id such as `#42` resolves to the turn(s) that fact came from.

Search never indexes the injected `<memory>` or summary blocks, so it can't
find its own notes. An agent that keeps searching without reading a hit is
nudged to read one.

```yaml
search:
  scope: conversation   # or `all`
  max_matches: 10
```

`scope: conversation` (the default) limits results to the current
conversation and fails closed if there isn't one. `all` searches every
conversation in the store. All keys are in the
[spec reference](../spec-reference.md#search).

## Memory

A durable notebook of facts about you and your work, injected into the agent's
context. The agent **reads** it; a separate background process **writes** it.

```
conversation turns ──▶ owner agent ──▶ journal ──▶ index ──▶ MEMORY.md
                       (background)    (source)    (derived)   (what the agent sees)
```

**Journal** (`data/memory/<agent>/memory-journal.jsonl`) is the source of
truth: an append-only log, one record per fact, never edited or deleted. Each
record has a type (`identity`, `rule`, `preference`, `decision`, `context`,
`event`), the date it became true, and a reference to the turn it came from.
Facts you contradict later are not overwritten; a newer record wins when the
view is built.

**Index** (`memory-index.json`) is derived from the journal and rebuilt from it
alone: which line is the live copy of each fact, and which older lines it
replaced. It can't drift into a second source of truth.

**View** (`MEMORY.md`) is rendered from the index. Read it with `/memory` in
`paf chat`.

**The owner** is a background agent (`specs/background_agents/memory_extractor.yaml`)
that reads new turns and appends facts. Memory belongs to the agent, not to
one conversation: it runs once 5 new turns have accumulated across all of the
agent's conversations, out of band, so it never slows a reply. A second background agent
(`memory_pair_reviewer`) runs inside the same job and merges duplicates and
applies updates, by writing overrides beside the journal rather than editing
it.

**What the agent can do.** It never edits memory. If it sees a fact that is
stale or wrong, it calls `memory_archive_suggestion(journal_id, reason)`. That
records a *pending* request; the fact stays visible, marked
*(archive pending review)*, until the reviewer decides.

### What is injected

Each turn the memory is rebuilt for this conversation: **About you**, **this
project**, any **mentioned project**, **recent history**, and a line naming
other projects. Sections are filled to a token budget, ranked by BM25 against
your latest messages and then by recency. The agent also gets
`search_memory(query, project?)` and `list_memory(project, type?, offset?)`.

Every fact shown ends with its id (`#N`), so `read_turn("#N")` shows where it
came from.

```yaml
memory:
  render:
    global_tokens: 2000
    project_tokens: 2500
```

### Projects

A project is a label on a conversation. In the scoped render it decides which
facts count as "this project's" and which as global. Manage them in `paf chat`:

| Command | |
| --- | --- |
| `/projects` | list projects |
| `/projects add <name> [-- description]` | create one |
| `/projects rename <slug> <name>`, `archive <slug>`, `unarchive <slug>` | edit |
| `/project` | show this conversation's project |
| `/project <slug>`, `/project none` | assign or clear |

Tagging a conversation after the fact is fine: its earlier facts are filed
under the project at the next render.

### Files

```
data/memory/<agent>/
  memory-journal.jsonl     the journal
  memory-index.json        derived index
  memory-overrides.jsonl   pending and decided archive/supersede overrides
  MEMORY.md                the rendered view
```

Memory settings (`consolidate_turns`, `max_memory_size`,
`pairwise_review`, `render`) are in the [spec reference](../spec-reference.md#memory).

### Limits

- Memory is background knowledge, not instruction. The agent is told that your
  latest statement wins over a remembered fact.
- A fact missing from the view may still exist: old `t3` entries age out of the
  tiered view. The agent is told so, and told to search before saying it
  doesn't know.
- Extraction is a model call per pass, so memory costs tokens in the
  background and can miss or mis-state a fact. Provenance is what makes that
  checkable: every fact links to its turn.

## ThreadSummarizer

Where memory holds facts that outlive a conversation, the thread summarizer
holds the story of *this* one: what it was doing and how it evolved, as dated
beats, each anchored to a turn. A long conversation keeps its thread even after
compaction trims the old messages.

- A background agent (`thread_summarizer`) writes summaries to
  `data/summaries/<agent>/conversations/`, per conversation, on its own
  cadence.
- Each real user turn, a bounded `<thread-summary>` block (oldest to newest,
  with recency headers) is injected. The injection costs nothing per turn; only
  the background writer calls a model.
- **`list_turn_summaries(offset?, limit?)`** pages through them. It only reads
  the current conversation.

Memory and the summarizer are independent: a profile can have either.
