"""The ``send_push`` tool. Its docstring is the model's guidance on WHEN to push."""

from __future__ import annotations

from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from paf.push import PushResult

_MESSAGES: dict[str, str] = {
    "rate_limited": (
        "Not sent: the notification limit for this hour was reached. Don't "
        "retry; put it in your reply instead."
    ),
    "duplicate": "Not sent: an identical notification was sent a few minutes ago.",
    "no_devices": "Not sent: no device is subscribed to notifications.",
    "watching": (
        "Not sent: the user is looking at this conversation right now, so they "
        "will see your reply. No notification is needed."
    ),
    "not_configured": "Not sent: push notifications aren't configured on the server.",
    "failed": "Not sent: the push service didn't accept it. Mention it in your reply.",
}


class PushToolset(FunctionToolset[AgentDepsT]):
    """One tool: notify the user's phone."""

    def __init__(self, notifier) -> None:
        super().__init__()
        self._notifier = notifier
        self.add_function(self.send_push, name="send_push")

    async def send_push(self, title: str, body: str, urgent: bool = False) -> str:
        """Send the user a push notification on their phone.

        Use this sparingly: only when the user would genuinely want to be
        interrupted right now, such as a scheduled check that found something
        they asked to be told about, or long work they're waiting on that has
        just finished or hit a problem. Never use it to repeat what your reply
        already says to someone who is reading it, for routine "nothing to
        report" results, or for progress updates. Notifications are rate-limited
        and tapping one opens this conversation, so your reply here can carry
        the detail; keep the notification itself to one short line.

        Args:
            title: A few words naming what happened (max 60 characters).
            body: One short sentence with the key fact (max 200 characters).
            urgent: True only for something that needs attention soon.
        """
        result: PushResult = await self._notifier(title, body, urgent)
        if result.status == "sent":
            return f"Notification sent to {result.sent} device(s)."
        return _MESSAGES.get(result.status, "Not sent.")
