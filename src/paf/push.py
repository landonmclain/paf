"""Web Push: VAPID config, the subscription store, and the sender.

The backend owns push (PWA-Push-Notifications-Spec): paf-ui only subscribes a
device and displays what arrives. This module is transport-agnostic — ``ui/sse.py``
exposes it over HTTP and (Phase 4) calls :meth:`PushService.broadcast` when a
turn settles.

* **Config** — VAPID keys live in the environment (sourced from ``.env`` by
  ``scripts/paf-sse-start.sh``): ``PAF_VAPID_PRIVATE_KEY`` / ``PAF_VAPID_PUBLIC_KEY``
  (url-safe base64, raw P-256) and ``PAF_VAPID_SUBJECT`` (``mailto:`` / https
  claim). Generate with ``scripts/gen_vapid.py``. Unconfigured → push is simply
  disabled (``PushService.enabled`` is False), never an import-time error.
* **Store** — one record per device, keyed by the push-service ``endpoint``
  (re-subscribing the same endpoint replaces it), persisted through the
  ``FlatFileStorage`` seam so it survives restarts.
* **Sender** — ``pywebpush.webpush_async`` (RFC 8030/8291/8292). A 404/410 from the
  push service means the subscription is gone and is deleted; other failures
  are logged and the subscription kept.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from paf.capabilities._storage import FlatFileStorage

log = logging.getLogger(__name__)

# Hosts of the push services browsers hand out endpoints for. The server POSTs
# to whatever endpoint a client registers, so an unrestricted allowlist would be
# an SSRF primitive — only accept the real push services (https only).
_PUSH_HOST_SUFFIXES = (
    "fcm.googleapis.com",  # Chrome / Android (and Chromium)
    "android.googleapis.com",  # legacy GCM endpoints
    "push.services.mozilla.com",  # Firefox
    "push.apple.com",  # Safari / iOS web.push.apple.com
    "notify.windows.com",  # Edge / WNS
)

# Push services cap the encrypted payload (~4 KB); stay clear of it.
MAX_PAYLOAD_BYTES = 3500

Outcome = Literal["sent", "gone", "failed"]


class PushNotConfigured(RuntimeError):
    """VAPID keys are not set; push is disabled."""


class InvalidSubscription(ValueError):
    """The submitted subscription is malformed or points at a non-push host."""


def b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


@dataclass(frozen=True)
class VapidConfig:
    private_key: str  # url-safe base64 raw P-256 private scalar
    public_key: (
        str  # url-safe base64 uncompressed point (the browser's applicationServerKey)
    )
    subject: str  # mailto:… or https://… (required VAPID claim)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> VapidConfig | None:
        env = os.environ if env is None else env
        private = env.get("PAF_VAPID_PRIVATE_KEY", "").strip()
        public = env.get("PAF_VAPID_PUBLIC_KEY", "").strip()
        subject = env.get("PAF_VAPID_SUBJECT", "").strip()
        if not (private and public and subject):
            return None
        return cls(private, public, subject)


def validate_subscription(sub: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized ``{endpoint, keys:{p256dh, auth}}`` or raise.

    Rejects non-https endpoints and hosts that aren't a known push service.
    """
    endpoint = sub.get("endpoint")
    keys = sub.get("keys")
    if not isinstance(endpoint, str) or not isinstance(keys, Mapping):
        raise InvalidSubscription("subscription needs an endpoint and keys")
    parsed = urlparse(endpoint)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not any(
        host == s or host.endswith("." + s) for s in _PUSH_HOST_SUFFIXES
    ):
        raise InvalidSubscription("endpoint is not a recognised push service")
    p256dh, auth = keys.get("p256dh"), keys.get("auth")
    if not (isinstance(p256dh, str) and isinstance(auth, str) and p256dh and auth):
        raise InvalidSubscription("keys.p256dh and keys.auth are required")
    try:
        b64url_decode(p256dh)
        b64url_decode(auth)
    except Exception as exc:  # noqa: BLE001 - any decode failure is a bad key
        raise InvalidSubscription("keys are not valid base64url") from exc
    return {"endpoint": endpoint, "keys": {"p256dh": p256dh, "auth": auth}}


class PushStore:
    """Device subscriptions, one record per endpoint, through the storage seam."""

    KIND = "subscriptions"
    KEY = "devices"

    def __init__(self, storage: FlatFileStorage) -> None:
        self._storage = storage

    def _load(self) -> dict[str, dict[str, Any]]:
        data = self._storage.read(self.KIND, self.KEY)
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self._storage.write(self.KIND, self.KEY, data)

    # Read-modify-write is synchronous (no await between load and save), so
    # concurrent requests on the event loop cannot interleave and lose updates.
    def add(
        self, subscription: Mapping[str, Any], *, user_agent: str = ""
    ) -> dict[str, Any]:
        sub = validate_subscription(subscription)
        data = self._load()
        record = {
            **sub,
            "user_agent": user_agent[:200],
            "created": data.get(sub["endpoint"], {}).get("created", time.time()),
            "updated": time.time(),
        }
        data[sub["endpoint"]] = record
        self._save(data)
        return record

    def remove(self, endpoint: str) -> bool:
        data = self._load()
        if endpoint not in data:
            return False
        del data[endpoint]
        self._save(data)
        return True

    def get(self, endpoint: str) -> dict[str, Any] | None:
        return self._load().get(endpoint)

    def all(self) -> list[dict[str, Any]]:
        return list(self._load().values())


# (subscription, payload-json, vapid-private-key, claims, ttl, headers) -> awaited
SendFn = Callable[..., Awaitable[Any]]


async def _default_send(**kwargs: Any) -> Any:
    from py_vapid import Vapid
    from pywebpush import webpush_async

    # Our env key is the raw base64url P-256 scalar (what gen_vapid.py emits);
    # pywebpush's string loader expects DER, so build the Vapid object ourselves.
    kwargs["vapid_private_key"] = Vapid.from_raw(kwargs["vapid_private_key"].encode())
    return await webpush_async(**kwargs)


def _status_of(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None)


class PushService:
    """Sends payloads to stored subscriptions and prunes dead ones."""

    def __init__(
        self,
        store: PushStore,
        config: VapidConfig | None,
        *,
        send: SendFn | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self._send = send or _default_send

    @property
    def enabled(self) -> bool:
        return self.config is not None

    @property
    def public_key(self) -> str:
        if self.config is None:
            raise PushNotConfigured("VAPID keys are not configured")
        return self.config.public_key

    @staticmethod
    def encode_payload(payload: Mapping[str, Any]) -> str:
        """JSON-encode, trimming ``body`` if the result would exceed the push cap."""
        data = dict(payload)
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        body = data.get("body")
        while (
            len(text.encode("utf-8")) > MAX_PAYLOAD_BYTES
            and isinstance(body, str)
            and body
        ):
            body = body[: max(0, len(body) // 2)]
            data["body"] = body
            text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        return text

    async def send_one(
        self,
        record: Mapping[str, Any],
        payload: Mapping[str, Any],
        *,
        ttl: int = 3600,
        urgency: Literal["very-low", "low", "normal", "high"] = "normal",
    ) -> Outcome:
        """Push to one device. 404/410 → delete the subscription, return ``gone``."""
        if self.config is None:
            raise PushNotConfigured("VAPID keys are not configured")
        endpoint = record["endpoint"]
        try:
            await self._send(
                subscription_info={"endpoint": endpoint, "keys": dict(record["keys"])},
                data=self.encode_payload(payload),
                vapid_private_key=self.config.private_key,
                vapid_claims={"sub": self.config.subject},
                ttl=ttl,
                headers={"Urgency": urgency},
                timeout=10,
            )
            return "sent"
        except Exception as exc:  # noqa: BLE001 - one bad device must not break the rest
            status = _status_of(exc)
            if status in (404, 410):
                self.store.remove(endpoint)
                log.info("push subscription gone (%s); removed", status)
                return "gone"
            log.warning("push send failed (status=%s): %s", status, exc)
            return "failed"

    async def send_to(
        self, endpoint: str, payload: Mapping[str, Any], **kw: Any
    ) -> Outcome | None:
        """Push to one registered endpoint; ``None`` if it isn't registered."""
        record = self.store.get(endpoint)
        if record is None:
            return None
        return await self.send_one(record, payload, **kw)

    async def broadcast(self, payload: Mapping[str, Any], **kw: Any) -> dict[str, int]:
        """Push to every subscription; returns ``{sent, gone, failed}`` counts."""
        counts = {"sent": 0, "gone": 0, "failed": 0}
        if not self.enabled:
            return counts
        import asyncio

        outcomes = await asyncio.gather(
            *(self.send_one(r, payload, **kw) for r in self.store.all())
        )
        for o in outcomes:
            counts[o] += 1
        return counts


def default_push_service(data_dir: Path) -> PushService:
    """The production service: store under ``<data_dir>/push``, config from env."""
    return PushService(
        PushStore(FlatFileStorage(data_dir / "push")), VapidConfig.from_env()
    )
