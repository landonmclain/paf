"""Model prices from the Langfuse project's own model definitions.

Langfuse prices each model call at ingestion, but that cost lives only on the
child observation. To show a cost on the ROOT span we price the same calls
ourselves, from the SAME definitions (``GET /api/public/models``), so the root
total matches the sum of its children. Prices load once in a daemon thread and
refresh hourly; lookups never touch the network (they run on span end).
"""

from __future__ import annotations

import base64
import re
import threading
import time
from typing import Any

_REFRESH_S = 3600.0
_lock = threading.Lock()
# (compiled match pattern, default-tier prices) — custom definitions first so
# they win over Langfuse-managed ones, mirroring Langfuse's own precedence.
_models: list[tuple[re.Pattern[str], dict[str, float]]] = []
_started = False


def _fetch(base: str, auth: str) -> list[tuple[re.Pattern[str], dict[str, float]]]:
    import httpx

    out: list[tuple[bool, re.Pattern[str], dict[str, float]]] = []
    page = 1
    while True:
        r = httpx.get(
            f"{base}/api/public/models",
            params={"page": page, "limit": 100},
            headers={"Authorization": auth},
            timeout=10,
        )
        r.raise_for_status()
        body = r.json()
        for m in body["data"]:
            tiers = m.get("pricingTiers") or []
            tier = next((t for t in tiers if t.get("isDefault")), None)
            prices = (tier or {}).get("prices") or {
                k: v["price"] for k, v in (m.get("prices") or {}).items()
            }
            try:
                pattern = re.compile(m["matchPattern"])
            except re.error:
                continue
            out.append((bool(m.get("isLangfuseManaged")), pattern, prices))
        if page >= body["meta"]["totalPages"]:
            break
        page += 1
    out.sort(key=lambda t: t[0])  # stable: custom (False) before managed (True)
    return [(p, pr) for _, p, pr in out]


def start_price_refresh(otlp_endpoint: str, public_key: str, secret_key: str) -> None:
    """Begin loading/refreshing prices in the background (idempotent)."""
    global _started
    with _lock:
        if _started or "/api/public" not in otlp_endpoint:
            return
        _started = True
    base = otlp_endpoint.split("/api/public", 1)[0]
    auth = "Basic " + base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

    def _loop() -> None:
        global _models
        while True:
            try:
                loaded = _fetch(base, auth)
                with _lock:
                    _models = loaded
            except Exception:  # noqa: BLE001 - keep last good prices, retry later
                pass
            time.sleep(_REFRESH_S)

    threading.Thread(target=_loop, name="paf-langfuse-prices", daemon=True).start()


def price_usage(model: str | None, usage: dict[str, Any]) -> float:
    """Cost of ``usage`` (Langfuse usage keys) for ``model``; 0.0 if unpriced."""
    if not model:
        return 0.0
    with _lock:
        models = _models
    for pattern, prices in models:
        if pattern.search(model):
            return sum(float(prices.get(k, 0.0)) * float(v) for k, v in usage.items())
    return 0.0
