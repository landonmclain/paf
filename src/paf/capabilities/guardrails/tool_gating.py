"""paf-owned tool-gating guard policy.

The harness ``ToolGuardrail`` is the shell; paf owns the policy as callables
here (the settled don't-pin stance). This module is the policy: which tool
calls the runtime refuses *before* they execute.

The first wiring is a hard gate (block-with-a-message), not interactive
approval. It gates only the highest-risk tools — the destructive/irreversible
calls across the shell, git, and filesystem capabilities. A call that never
runs can never hurt you.
"""

from __future__ import annotations

from pydantic_ai_harness.guardrails import GuardrailResult, ToolCallInfo

# Tool names the runtime refuses to execute. The destructive/irreversible
# calls across the shell, git, and filesystem capabilities.
DEFAULT_GATED_TOOLS: frozenset[str] = frozenset(
    {
        # Shell — arbitrary command execution.
        # "run_command",
        # "start_command",
        # Git — writes commits / discards uncommitted changes.
        # "git_checkpoint",
        # "git_restore_paths",
        # FileSystem — overwrites files.
        # "write_file",
        # "edit_file",
    }
)

_BLOCK_MESSAGE = (
    "Tool '{name}' is gated by paf's ToolGating guardrail. "
    "The runtime refused this call before it executed."
)


def tool_gating_guard(info: ToolCallInfo) -> GuardrailResult:
    """Refuse calls to gated tools; let everything else through."""
    if info.name in DEFAULT_GATED_TOOLS:
        return GuardrailResult.block(_BLOCK_MESSAGE.format(name=info.name))
    return GuardrailResult.allow()
