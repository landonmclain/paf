"""In-process scratch store for captured screenshots.

When the main agent model is text-only, screenshots taken by a harness browser
tool (`screenshot` / `navigate`) arrive as raw `BinaryContent` that a text-only
model cannot consume. Instead of crashing (or lossily generic-summarizing), the
`Visual` capability's ``before_model_request`` hook intercepts each captured
image, saves its bytes here under a short ``shot_<n>`` reference, and replaces
the pixels in the outgoing message with a text stub telling the model it can
call ``describe_screenshot(ref=..., prompt=...)`` to ask a *targeted* question
about that image through the vision model.

This store is intentionally trivial and transient:
  - in-memory dict, one process, no persistence — screenshots are turn-local;
  - the refs are validated and the dict is bounded so an unbounded history of
    captures can't grow without limit;
  - it is the ONLY coupling between the capture side (hook) and the consume
    side (``describe_screenshot`` tool), both in the same capability.

Refs survive for the lifetime of the process. A turn keeps a handful of
screenshots, so a simple bound on total entries is plenty; once consumed or
dropped the bytes can be released by ``drop``.
"""

from __future__ import annotations

import secrets
from threading import Lock
from typing import Final

_MAX_IMAGES: Final = 64
"""Upper bound on retained screenshots — far beyond any realistic turn."""


class ScreenshotStore:
    """Bounded ref → PNG-bytes scratch store with prefix-validated refs."""

    _PREFIX = "shot_"

    def __init__(self, max_images: int = _MAX_IMAGES) -> None:
        self._max = max_images
        self._images: dict[str, bytes] = {}
        self._lock = Lock()

    def save(self, data: bytes) -> str:
        """Persist `data` and return a fresh ref the model can reference.

        Evicts the oldest entry when at capacity so a long-running agent cannot
        grow the store without bound.
        """
        with self._lock:
            if len(self._images) >= self._max:
                self._images.pop(next(iter(self._images)))
            ref = f"{self._PREFIX}{secrets.token_hex(6)}"
            self._images[ref] = data
            return ref

    def get(self, ref: str) -> bytes | None:
        """Return the bytes for `ref`, validating it is one of ours."""
        if not isinstance(ref, str) or not ref.startswith(self._PREFIX):
            return None
        with self._lock:
            return self._images.get(ref)

    def drop(self, ref: str) -> None:
        """Release the entry for `ref` if present."""
        with self._lock:
            self._images.pop(ref, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._images)


# Shared per-process store. One instance is enough — the store is just
# process-scoped scratch.
_store = ScreenshotStore()
