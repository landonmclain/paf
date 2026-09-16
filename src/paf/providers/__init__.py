"""Provider-specific model construction and defaults.

Each module owns the model construction and defaults for one model provider
(OpenRouter, OpenAI, Anthropic, ...). The generic spec loader dispatches here
based on the model string prefix; provider-specific concerns never leak into
``paf.specs``.
"""

from paf.providers.openrouter import model_with_client

__all__ = ["model_with_client"]
