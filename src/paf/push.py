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
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote, urlparse

from paf.capabilities._storage import FlatFileStorage

if TYPE_CHECKING:
    from paf.runtime import TurnSettled

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


class PresenceTracker:
    """Which devices are looking at which conversation RIGHT NOW (in memory).

    Each device's page reports ``(endpoint, agent, cid, watching)`` as a heartbeat
    while it is visible and focused; a report expires after ``TTL`` seconds, so a
    backgrounded, locked or closed app drops out on its own with no goodbye needed.
    The sender skips a device that is watching the conversation a push is about:
    suppressing on the server (rather than the service worker hiding a push the
    phone already received) is reliable on iOS, which also expects every push it
    receives to show a notification and can revoke subscriptions that stay silent.
    """

    TTL = 30.0  # seconds; heartbeats come every ~10s

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._seen: dict[str, tuple[str, str, float]] = {}

    def report(self, endpoint: str, agent: str, cid: str, watching: bool) -> None:
        if watching:
            self._seen[endpoint] = (agent, cid, self._clock())
        else:
            self._seen.pop(endpoint, None)

    def watching(self, agent: str, cid: str | None) -> set[str]:
        """Endpoints currently watching this conversation (empty if ``cid`` is None)."""
        now = self._clock()
        self._seen = {e: v for e, v in self._seen.items() if now - v[2] < self.TTL}
        if not cid:
            return set()
        return {e for e, (a, c, _) in self._seen.items() if a == agent and c == cid}


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
        presence: PresenceTracker | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.presence = presence or PresenceTracker()
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

    async def broadcast(
        self,
        payload: Mapping[str, Any],
        *,
        exclude: set[str] | None = None,
        **kw: Any,
    ) -> dict[str, int]:
        """Push to every subscription except ``exclude`` (endpoints that are
        watching this conversation); returns ``{sent, gone, failed}`` counts."""
        counts = {"sent": 0, "gone": 0, "failed": 0}
        if not self.enabled:
            return counts
        import asyncio

        outcomes = await asyncio.gather(
            *(
                self.send_one(r, payload, **kw)
                for r in self.store.all()
                if r["endpoint"] not in (exclude or ())
            )
        )
        for o in outcomes:
            counts[o] += 1
        return counts


# --- settle policy -----------------------------------------------------------
_SNIPPET_CHARS = 140


def _snippet(text: str, limit: int = _SNIPPET_CHARS) -> str:
    """One tidy line from model text: strip markdown noise, collapse whitespace."""
    import re

    t = re.sub(r"[`*_>#]+", "", text)  # markdown emphasis / headings / quotes
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def settled_payload(
    settled: TurnSettled, *, title: str = "paf"
) -> dict[str, str] | None:
    """The push for a settled turn, or ``None`` when it should stay quiet.

    Hybrid policy (PWA-Push-Notifications-Spec, "Agent-chosen push"):

    * the agent already notified during the turn (``pushed``) → nothing (no double);
    * a ``user`` turn → push when it finishes ("answer ready") or fails; stay quiet
      if they pressed Stop themselves;
    * a ``notify`` turn (process / subagent / scheduled task) → quiet when it
      finishes (the agent decides via its own push tool), but always push a failure
      so a scheduled task can't die silently;
    * a hard abort (idle reaper / close) → push either way, since the user would
      otherwise wait on work that is never coming.

    Whether the user is *looking* is decided on the device (service worker), so
    this never tries to guess it.
    """
    if settled.pushed:
        return None
    background = settled.origin == "notify"
    outcome = settled.outcome
    if outcome == "ok":
        if background:
            return None
        body = _snippet(settled.text) or "Your answer is ready."
    elif outcome == "stopped":
        return None
    elif outcome == "aborted":
        body = "The turn was stopped before it finished (idle timeout)."
    else:  # error
        what = "Background task failed" if background else "The turn failed"
        detail = _snippet(settled.error or "", 100)
        body = f"{what}: {detail}" if detail else f"{what}."
    payload: dict[str, str] = {
        "title": _snippet(title, 60) or "paf",
        "body": body,
        "agent": settled.agent,
        "outcome": outcome,
        "url": "/",
    }
    cid = settled.conversation_id
    if cid:
        payload["cid"] = cid
        payload["tag"] = cid
        payload["url"] = (
            f"/?agent={quote(settled.agent, safe='')}&c={quote(cid, safe='')}"
        )
    return payload


# --- agent-chosen push ---------------------------------------------------------
PushStatus = Literal[
    "sent",
    "rate_limited",
    "duplicate",
    "no_devices",
    "watching",  # every device is looking at this conversation right now
    "not_configured",
    "failed",
]


@dataclass(frozen=True)
class PushRequest:
    """A notification the AGENT chose to send (the ``send_push`` tool)."""

    agent: str
    conversation_id: str | None
    title: str
    body: str
    urgent: bool = False


@dataclass(frozen=True)
class PushResult:
    status: PushStatus
    sent: int = 0  # devices reached (status == "sent")


class AgentPusher:
    """Applies the guardrails to agent-chosen pushes, then broadcasts.

    Fixed, in-memory limits (they reset on restart; promote to spec config when
    you know what you want): at most ``max_per_hour`` pushes per agent per rolling
    hour, and an identical (title, body) from one agent is suppressed for
    ``dedupe_seconds``. Only pushes that actually reached a device count, so
    "no devices" or a push-service outage never burns the budget. The link is
    built here from the conversation the agent is in; the agent can't choose it.
    """

    TITLE_MAX = 60
    BODY_MAX = 200

    def __init__(
        self,
        service: Callable[[], PushService],
        *,
        max_per_hour: int = 10,
        dedupe_seconds: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._service = service
        self._max_per_hour = max_per_hour
        self._dedupe_seconds = dedupe_seconds
        self._clock = clock
        self._sent: dict[str, list[float]] = {}
        self._recent: dict[tuple[str, str, str], float] = {}

    async def __call__(self, req: PushRequest) -> PushResult:
        result = await self._push(req)
        # One line per attempt so "why did my phone buzz (or not)" is answerable.
        log.info(
            "agent push %s:%s %s sent=%d title=%r",
            req.agent,
            req.conversation_id,
            result.status,
            result.sent,
            req.title[:60],
        )
        return result

    async def _push(self, req: PushRequest) -> PushResult:
        svc = self._service()
        if not svc.enabled:
            return PushResult("not_configured")
        title = _snippet(req.title, self.TITLE_MAX) or "paf"
        body = _snippet(req.body, self.BODY_MAX)
        now = self._clock()

        key = (req.agent, title.lower(), body.lower())
        self._recent = {
            k: t for k, t in self._recent.items() if now - t < self._dedupe_seconds
        }
        if key in self._recent:
            return PushResult("duplicate")
        window = [t for t in self._sent.get(req.agent, []) if now - t < 3600]
        self._sent[req.agent] = window
        if len(window) >= self._max_per_hour:
            return PushResult("rate_limited")
        devices = svc.store.all()
        if not devices:
            return PushResult("no_devices")
        # A device that is looking at this conversation sees the reply: don't buzz
        # it. If that's every device, nothing is sent (and no budget is spent).
        watching = svc.presence.watching(req.agent, req.conversation_id)
        if all(d["endpoint"] in watching for d in devices):
            return PushResult("watching")

        payload: dict[str, str] = {
            "title": title,
            "body": body,
            "agent": req.agent,
            "url": "/",
        }
        if req.conversation_id:
            cid = req.conversation_id
            payload.update(
                cid=cid,
                tag=cid,
                url=f"/?agent={quote(req.agent, safe='')}&c={quote(cid, safe='')}",
            )
        counts = await svc.broadcast(
            payload,
            exclude=watching,
            ttl=6 * 3600,
            urgency="high" if req.urgent else "normal",
        )
        if counts["sent"] == 0:
            return PushResult("failed")
        self._sent[req.agent].append(now)
        self._recent[key] = now
        return PushResult("sent", counts["sent"])


def default_push_service(data_dir: Path) -> PushService:
    """The production service: store under ``<data_dir>/push``, config from env."""
    return PushService(
        PushStore(FlatFileStorage(data_dir / "push")), VapidConfig.from_env()
    )
