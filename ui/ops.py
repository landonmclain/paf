"""paf ops transport — tiny systemd-control surface for the SSE server.

A browser cannot run ``systemctl`` directly, so restarting the SSE transport
(``paf-sse.service``) requires a small, separate, long-lived process that
survives the SSE server going down. This is that process: a minimal FastAPI
service (own port, systemd ``paf-ops.service``) exposing:

    POST /restart   ->  systemctl --user restart paf-sse

It is deliberately SEPARATE from ``ui/sse.py`` so that restarting the SSE
server never takes down the thing that answered the restart request — the
HTTP response returns even though the SSE process is being bounced.

Authorization: THE ENDPOINT IS AUTHENTICATED. It is reachable from the whole
Tailnet (bound 0.0.0.0 like the other services), so a request must carry the
ops token in an ``Authorization: Bearer <token>`` header. The token lives in
``.env`` as ``PAF_OPS_TOKEN`` (also mirrored to the frontend's
``NEXT_PUBLIC_PAF_OPS_TOKEN`` so the UI can send it). Compare with a
constant-time check. This is the personal/Tailnet trust model — the token is
only as secret as the built frontend, which is acceptable for a single-user
Tailnet but would NOT be for a public multi-tenant deployment.

No runtime changes, no pydantic-ai imports beyond what's needed to run.
"""

from __future__ import annotations

import asyncio
import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


# Token used to authorize restart requests. Read from PAF_OPS_TOKEN (from
# `.env` in the host process). If unset, the service still starts but refuses
# every restart (fail-closed) — it must never be open.
def _load_token() -> str:
    tok = os.environ.get("PAF_OPS_TOKEN", "").strip()
    if not tok:
        # Fall back to a token file next to the project root, chmod 600.
        try:
            p = Path(__file__).resolve().parents[1] / "data" / ".paf-ops.token"
            if p.exists():
                return p.read_text().strip()
        except OSError:
            pass
        return ""
    return tok


# The systemd unit to restart. Restarting the SSE server kills the very
# connection the browser used to talk to it, but this ops process is separate
# and survives, so the request completes and the frontend can show "done".
SSE_UNIT = os.environ.get("PAF_SSE_UNIT", "paf-sse.service")

# 127.0.0.1-only default for local safety; override via PAF_OPS_HOST to bind
# the Tailscale-reachable interface when remote restart is wanted.
OPS_HOST = os.environ.get("PAF_OPS_HOST", "0.0.0.0")
OPS_PORT = int(os.environ.get("PAF_OPS_PORT", "8091"))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # No long-lived state; this service is intentionally stateless.
    yield


app = FastAPI(title="paf ops transport", lifespan=_lifespan)

# Narrow CORS for the paf-ui dev server (localhost:3000/3001) + the Tailscale
# host, mirroring ui/sse.py so the frontend can call it cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        # Same-origin HTTPS via `tailscale serve` (the phone/PWA origin). The UI
        # and /api + /ops share it, so these requests are not cross-origin; kept for
        # tooling that sends an Origin header.
        "https://paf.example.com",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _authorized(authorization: str | None) -> bool:
    """Constant-time check of the bearer token. Fail-closed: no token on the
    host, or a missing/wrong header, both mean denied."""
    expected = _load_token()
    if not expected:
        return False
    if not authorization or not authorization.startswith("Bearer "):
        return False
    provided = authorization[len("Bearer ") :].strip()
    return hmac.compare_digest(provided, expected)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "paf-ops"}


@app.post("/api/restart")
async def restart(authorization: str | None = Header(default=None)) -> dict[str, str]:
    """Restart the SSE transport via systemd.

    Runs ``systemctl --user restart <SSE_UNIT>`` as the host user. This is
    intentionally an on-demand, unconditional restart (the operator accepted
    that an in-progress turn may be killed). The command is fire-and-forget:
    we run it, then return; systemd handles the bounce.
    """
    if not _authorized(authorization):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="unauthorized",
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            "restart",
            SSE_UNIT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError:
        raise HTTPException(502, detail="systemctl timed out") from None
    except FileNotFoundError:
        raise HTTPException(500, detail="systemctl not found on host") from None

    if proc.returncode != 0:
        msg = (stderr or b"").decode(errors="replace").strip()
        raise HTTPException(500, detail=f"systemctl failed: {msg}")

    return {"status": "restarting", "unit": SSE_UNIT}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(port: int = OPS_PORT) -> None:
    import uvicorn

    uvicorn.run(app, host=OPS_HOST, port=port, log_level="info")


if __name__ == "__main__":
    main()
