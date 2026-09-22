"""Shell capability: gives agents configurable command execution.

Originally copied from pydantic-ai-harness, MIT.
"""

from paf.capabilities.shell._capability import LLM_API_KEY_ENV_PATTERNS, Shell
from paf.capabilities.shell._events import (
    EventCallback,
    ProcessEvent,
    ProcessEventEmitter,
    ProcessEventType,
)
from paf.capabilities.shell._toolset import ShellToolset

__all__ = [
    "EventCallback",
    "LLM_API_KEY_ENV_PATTERNS",
    "ProcessEvent",
    "ProcessEventEmitter",
    "ProcessEventType",
    "Shell",
    "ShellToolset",
]
