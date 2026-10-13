"""paf-owned Push capability: lets the agent notify the user's phone.

One tool, ``send_push``. The agent decides when a notification is worth the
interruption (most usefully in scheduled tasks: "check X, tell me only if Y");
the framework supplies the delivery, the deep link to the current conversation,
and the guardrails (rate limit, dedupe, size caps). Gated by the profile
allowlist: an agent whose profile lacks ``"Push"`` has no tool and cannot push.
See the vault spec ``PWA-Push-Notifications-Spec.md`` ("Agent-chosen push").
"""

from __future__ import annotations

from ._capability import Push
from ._toolset import PushToolset

__all__ = ["Push", "PushToolset"]
