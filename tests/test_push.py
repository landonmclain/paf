"""Tests for Web Push (src/paf/push.py + the /api/push/* routes in ui/sse.py).

Network-free: the sender is injected, so nothing here talks to a push service.
The VAPID key handling is exercised for real (cryptography, no network).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from paf import push
from paf.capabilities._storage import FlatFileStorage
from paf.push import (
    InvalidSubscription,
    PushService,
    PushStore,
    VapidConfig,
    validate_subscription,
)
from ui import sse

# A syntactically valid subscription (keys are just base64url; never decrypted here).
P256DH = push.b64url_encode(b"\x04" + b"a" * 64)
AUTH = push.b64url_encode(b"b" * 16)


def _sub(n: int = 1, host: str = "fcm.googleapis.com") -> dict[str, Any]:
    return {
        "endpoint": f"https://{host}/fcm/send/device-{n}",
        "keys": {"p256dh": P256DH, "auth": AUTH},
    }


CONFIG = VapidConfig("priv", "pub-key", "mailto:me@example.com")


def _store(tmp_path) -> PushStore:
    return PushStore(FlatFileStorage(tmp_path))


class _WebPushError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"push failed {status_code}")
        self.status_code = status_code


class _FakeSender:
    """Records calls; raises per-endpoint statuses to simulate push-service errors."""

    def __init__(self, fail: dict[str, int] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail or {}

    async def __call__(self, **kw: Any) -> None:
        self.calls.append(kw)
        status = self.fail.get(kw["subscription_info"]["endpoint"])
        if status:
            raise _WebPushError(status)


# --- validation ------------------------------------------------------------
@pytest.mark.parametrize(
    "host",
    [
        "fcm.googleapis.com",
        "updates.push.services.mozilla.com",
        "web.push.apple.com",
        "wns2-par02p.notify.windows.com",
    ],
)
def test_validate_accepts_known_push_services(host: str) -> None:
    assert validate_subscription(_sub(host=host))["endpoint"].startswith("https://")


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://fcm.googleapis.com/x",  # not https
        "https://evil.example.com/x",  # not a push service
        "https://fcm.googleapis.com.evil.com/x",  # suffix spoof
        "https://169.254.169.254/latest/meta-data",  # SSRF target
        "not a url",
    ],
)
def test_validate_rejects_non_push_endpoints(endpoint: str) -> None:
    sub = _sub()
    sub["endpoint"] = endpoint
    with pytest.raises(InvalidSubscription):
        validate_subscription(sub)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.pop("keys"),
        lambda s: s["keys"].pop("auth"),
        lambda s: s["keys"].update(p256dh=""),
        lambda s: s["keys"].update(auth="***not base64***"),
    ],
)
def test_validate_rejects_bad_keys(mutate) -> None:
    sub = _sub()
    mutate(sub)
    with pytest.raises(InvalidSubscription):
        validate_subscription(sub)


# --- store -----------------------------------------------------------------
def test_store_dedupes_by_endpoint_and_persists(tmp_path) -> None:
    store = _store(tmp_path)
    first = store.add(_sub(1))
    store.add(_sub(2))
    again = store.add(
        _sub(1), user_agent="iPhone"
    )  # re-subscribe: replace, not duplicate
    assert len(store.all()) == 2
    assert again["created"] == first["created"]
    assert store.get(_sub(1)["endpoint"])["user_agent"] == "iPhone"
    # A fresh store over the same dir sees the data (survives restart).
    assert len(_store(tmp_path).all()) == 2


def test_store_remove(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_sub(1))
    assert store.remove(_sub(1)["endpoint"]) is True
    assert store.remove(_sub(1)["endpoint"]) is False
    assert store.all() == []


def test_store_tolerates_corrupt_file(tmp_path) -> None:
    (tmp_path / PushStore.KEY).mkdir(parents=True)
    (tmp_path / PushStore.KEY / f"{PushStore.KIND}.json").write_text("{not json")
    assert _store(tmp_path).all() == []


# --- config ----------------------------------------------------------------
def test_vapid_config_from_env() -> None:
    env = {
        "PAF_VAPID_PRIVATE_KEY": "a",
        "PAF_VAPID_PUBLIC_KEY": "b",
        "PAF_VAPID_SUBJECT": "mailto:x@y.z",
    }
    assert VapidConfig.from_env(env) == VapidConfig("a", "b", "mailto:x@y.z")
    assert VapidConfig.from_env({**env, "PAF_VAPID_SUBJECT": ""}) is None
    assert VapidConfig.from_env({}) is None


def test_generated_keys_round_trip_through_pywebpush() -> None:
    """The env-key format gen_vapid.py emits must load via Vapid.from_raw and sign."""
    from py_vapid import Vapid

    from scripts.gen_vapid import generate

    keys = generate()
    vv = Vapid.from_raw(keys["PAF_VAPID_PRIVATE_KEY"].encode())
    headers = vv.sign(
        {"sub": keys["PAF_VAPID_SUBJECT"], "aud": "https://fcm.googleapis.com"}
    )
    assert "vapid" in headers["Authorization"].lower()
    # The public key we serve to the browser is the signing key's public point.
    assert len(push.b64url_decode(keys["PAF_VAPID_PUBLIC_KEY"])) == 65


def test_default_send_builds_vapid_object(monkeypatch) -> None:
    from py_vapid import Vapid

    from scripts.gen_vapid import generate

    seen: dict[str, Any] = {}

    async def fake_webpush_async(**kw: Any) -> None:
        seen.update(kw)

    monkeypatch.setattr("pywebpush.webpush_async", fake_webpush_async)
    key = generate()["PAF_VAPID_PRIVATE_KEY"]
    asyncio.run(push._default_send(vapid_private_key=key, data="x"))
    assert isinstance(seen["vapid_private_key"], Vapid)


# --- sender ----------------------------------------------------------------
def test_send_one_passes_vapid_ttl_urgency_and_payload(tmp_path) -> None:
    store = _store(tmp_path)
    rec = store.add(_sub(1))
    sender = _FakeSender()
    svc = PushService(store, CONFIG, send=sender)

    out = asyncio.run(
        svc.send_one(rec, {"title": "t", "body": "b"}, ttl=60, urgency="high")
    )

    assert out == "sent"
    (call,) = sender.calls
    assert call["vapid_private_key"] == "priv"
    assert call["vapid_claims"] == {"sub": "mailto:me@example.com"}
    assert call["ttl"] == 60 and call["headers"] == {"Urgency": "high"}
    assert json.loads(call["data"]) == {"title": "t", "body": "b"}
    assert call["subscription_info"]["endpoint"] == _sub(1)["endpoint"]


@pytest.mark.parametrize("status", [404, 410])
def test_gone_subscription_is_pruned(tmp_path, status: int) -> None:
    store = _store(tmp_path)
    rec = store.add(_sub(1))
    svc = PushService(store, CONFIG, send=_FakeSender({rec["endpoint"]: status}))
    assert asyncio.run(svc.send_one(rec, {"title": "t"})) == "gone"
    assert store.all() == []


@pytest.mark.parametrize("status", [429, 500, 503, None])
def test_transient_failure_keeps_subscription(tmp_path, status) -> None:
    store = _store(tmp_path)
    rec = store.add(_sub(1))

    async def boom(**kw: Any) -> None:
        raise _WebPushError(status) if status else RuntimeError("network down")

    svc = PushService(store, CONFIG, send=boom)
    assert asyncio.run(svc.send_one(rec, {"title": "t"})) == "failed"
    assert len(store.all()) == 1


def test_broadcast_isolates_failures_and_prunes(tmp_path) -> None:
    store = _store(tmp_path)
    for n in (1, 2, 3):
        store.add(_sub(n))
    sender = _FakeSender({_sub(2)["endpoint"]: 410, _sub(3)["endpoint"]: 500})
    svc = PushService(store, CONFIG, send=sender)

    counts = asyncio.run(svc.broadcast({"title": "t"}))

    assert counts == {"sent": 1, "gone": 1, "failed": 1}
    assert len(sender.calls) == 3  # one bad device didn't stop the others
    assert sorted(r["endpoint"] for r in store.all()) == [
        _sub(1)["endpoint"],
        _sub(3)["endpoint"],
    ]


def test_broadcast_noop_when_unconfigured(tmp_path) -> None:
    store = _store(tmp_path)
    store.add(_sub(1))
    sender = _FakeSender()
    svc = PushService(store, None, send=sender)
    assert asyncio.run(svc.broadcast({"title": "t"})) == {
        "sent": 0,
        "gone": 0,
        "failed": 0,
    }
    assert sender.calls == []


def test_encode_payload_trims_oversize_body() -> None:
    text = PushService.encode_payload({"title": "t", "body": "x" * 10_000, "url": "/"})
    assert len(text.encode()) <= push.MAX_PAYLOAD_BYTES
    data = json.loads(text)
    assert data["title"] == "t" and data["url"] == "/" and data["body"].startswith("x")


# --- HTTP routes -----------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    sender = _FakeSender()
    svc = PushService(_store(tmp_path), CONFIG, send=sender)
    monkeypatch.setattr(sse, "_push_service", lambda: svc)
    c = TestClient(sse.app)
    c.svc, c.sender = svc, sender  # type: ignore[attr-defined]
    return c


def test_route_public_key(client) -> None:
    assert client.get("/api/push/vapid-public-key").json() == {"public_key": "pub-key"}


def test_routes_503_when_unconfigured(tmp_path, monkeypatch) -> None:
    svc = PushService(_store(tmp_path), None)
    monkeypatch.setattr(sse, "_push_service", lambda: svc)
    c = TestClient(sse.app)
    assert c.get("/api/push/vapid-public-key").status_code == 503
    assert c.post("/api/push/subscribe", json=_sub()).status_code == 503
    assert c.post("/api/push/test", json=_sub()).status_code == 503


def test_route_subscribe_accepts_raw_and_wrapped(client) -> None:
    assert client.post("/api/push/subscribe", json=_sub(1)).json() == {
        "ok": True,
        "subscriptions": 1,
    }
    r = client.post("/api/push/subscribe", json={"subscription": _sub(2)})
    assert r.json() == {"ok": True, "subscriptions": 2}


def test_route_subscribe_rejects_bad_endpoint(client) -> None:
    bad = _sub()
    bad["endpoint"] = "https://evil.example.com/x"
    assert client.post("/api/push/subscribe", json=bad).status_code == 422
    assert client.svc.store.all() == []


def test_route_unsubscribe(client) -> None:
    client.post("/api/push/subscribe", json=_sub(1))
    ep = _sub(1)["endpoint"]
    assert client.post("/api/push/unsubscribe", json={"endpoint": ep}).json()["removed"]
    assert not client.post("/api/push/unsubscribe", json={"endpoint": ep}).json()[
        "removed"
    ]
    assert client.post("/api/push/unsubscribe", json={}).status_code == 422


def test_route_test_pushes_only_the_callers_endpoint(client) -> None:
    client.post("/api/push/subscribe", json=_sub(1))
    client.post("/api/push/subscribe", json=_sub(2))

    r = client.post("/api/push/test", json={"endpoint": _sub(1)["endpoint"]})

    assert r.json() == {"ok": True, "outcome": "sent"}
    assert [c["subscription_info"]["endpoint"] for c in client.sender.calls] == [
        _sub(1)["endpoint"]
    ]  # device 2 was NOT pinged


def test_route_test_unknown_endpoint_404(client) -> None:
    r = client.post("/api/push/test", json={"endpoint": _sub(9)["endpoint"]})
    assert r.status_code == 404
    assert client.sender.calls == []


# --- settle policy (hybrid push) --------------------------------------------
from paf.runtime import TurnSettled  # noqa: E402


def _settled(**kw: Any) -> TurnSettled:
    base: dict[str, Any] = dict(
        agent="assistant",
        conversation_id="c-1",
        origin="user",
        outcome="ok",
        text="All done. Here is the summary.",
        error=None,
        pushed=False,
    )
    base.update(kw)
    return TurnSettled(**base)


def test_policy_user_ok_pushes_answer_snippet_with_deeplink() -> None:
    p = push.settled_payload(_settled(), title="Quarterly review")
    assert p == {
        "title": "Quarterly review",
        "body": "All done. Here is the summary.",
        "agent": "assistant",
        "outcome": "ok",
        "cid": "c-1",
        "tag": "c-1",
        "url": "/?agent=assistant&c=c-1",
    }


@pytest.mark.parametrize(
    ("kw", "expect_push"),
    [
        ({"origin": "user", "outcome": "ok"}, True),
        ({"origin": "user", "outcome": "error", "error": "boom"}, True),
        ({"origin": "user", "outcome": "stopped"}, False),  # you pressed Stop
        ({"origin": "user", "outcome": "aborted"}, True),
        ({"origin": "notify", "outcome": "ok"}, False),  # agent decides via its tool
        ({"origin": "notify", "outcome": "stopped"}, False),
        ({"origin": "notify", "outcome": "error", "error": "boom"}, True),
        ({"origin": "notify", "outcome": "aborted"}, True),
        ({"origin": "user", "outcome": "ok", "pushed": True}, False),  # no double
        ({"origin": "notify", "outcome": "error", "pushed": True}, False),
    ],
)
def test_policy_table(kw: dict[str, Any], expect_push: bool) -> None:
    assert (push.settled_payload(_settled(**kw)) is not None) is expect_push


def test_policy_wording_and_escaping() -> None:
    err = push.settled_payload(
        _settled(origin="notify", outcome="error", error="disk full")
    )
    assert err["body"] == "Background task failed: disk full"
    assert "idle timeout" in push.settled_payload(_settled(outcome="aborted"))["body"]
    # Empty answer text falls back to a generic line; markdown noise is stripped.
    assert push.settled_payload(_settled(text=""))["body"] == "Your answer is ready."
    assert (
        push.settled_payload(_settled(text="## Title\n\n**bold** `code`"))["body"]
        == "Title bold code"
    )
    # Long text is truncated to one tidy line; ids are URL-escaped; no cid -> "/".
    long = push.settled_payload(_settled(text="word " * 200))["body"]
    assert len(long) <= 140 and long.endswith("…")
    p = push.settled_payload(_settled(agent="a b", conversation_id="x/y?z"))
    assert p["url"] == "/?agent=a%20b&c=x%2Fy%3Fz"
    q = push.settled_payload(_settled(conversation_id=None))
    assert q["url"] == "/" and "cid" not in q and "tag" not in q


# --- sse wiring: settled turn -> broadcast -------------------------------------
def _sse_with(tmp_path, monkeypatch, config=CONFIG):
    sender = _FakeSender()
    svc = PushService(_store(tmp_path), config, send=sender)
    svc.store.add(_sub(1))
    monkeypatch.setattr(sse, "_push_service", lambda: svc)
    monkeypatch.setattr(sse, "_meta_store", lambda agent: _Meta())
    return svc, sender


class _Meta:
    def get(self, cid: str) -> dict[str, Any]:
        return {"title": "My chat"} if cid == "c-1" else {}


def test_sse_callback_broadcasts_user_turn(tmp_path, monkeypatch) -> None:
    _, sender = _sse_with(tmp_path, monkeypatch)
    asyncio.run(sse._on_turn_settled("assistant", _settled()))
    (call,) = sender.calls
    data = json.loads(call["data"])
    assert data["title"] == "My chat" and data["url"] == "/?agent=assistant&c=c-1"
    assert call["ttl"] == 6 * 3600


def test_sse_callback_quiet_cases(tmp_path, monkeypatch) -> None:
    _, sender = _sse_with(tmp_path, monkeypatch)
    asyncio.run(sse._on_turn_settled("assistant", _settled(origin="notify")))
    assert sender.calls == []
    # Unconfigured push: never touches the store or sender.
    _, sender2 = _sse_with(tmp_path, monkeypatch, config=None)
    asyncio.run(sse._on_turn_settled("assistant", _settled()))
    assert sender2.calls == []
    # Aborted by server shutdown: stay quiet.
    _, sender3 = _sse_with(tmp_path, monkeypatch)
    monkeypatch.setattr(sse, "_shutting_down", True)
    asyncio.run(sse._on_turn_settled("assistant", _settled(outcome="aborted")))
    assert sender3.calls == []
