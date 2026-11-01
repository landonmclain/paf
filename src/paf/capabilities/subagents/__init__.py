"""paf-owned subagent delegation capability.

The design is a fixed roster of spec-driven leaf agents, one execution path that
traces every
child run and persists every result (task JSONL + ``.md`` output files).
"""

from paf.capabilities.subagents._capability import Subagents
from paf.capabilities.subagents._roster import Leaf, LeafPolicy, load_leaf
from paf.capabilities.subagents._store import SubagentTaskStore

__all__ = ["Leaf", "LeafPolicy", "Subagents", "SubagentTaskStore", "load_leaf"]
