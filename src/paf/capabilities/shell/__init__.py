"""Shell capability: gives agents configurable command execution.

paf-owned copy (verbatim from pydantic-ai-harness, MIT).
"""

from paf.capabilities.shell._capability import LLM_API_KEY_ENV_PATTERNS, Shell
from paf.capabilities.shell._toolset import ShellToolset

__all__ = ["LLM_API_KEY_ENV_PATTERNS", "Shell", "ShellToolset"]
