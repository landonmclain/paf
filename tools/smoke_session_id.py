"""Cheap smoke: prove langfuse.session.id now lands RAW (fix 63fb796).

Fires ONE real agent turn wrapped in root_span with a fresh session id, then
queries Langfuse for traces?sessionId=<clean> and asserts the id matches the
clean value (no JSON quotes). Mirrors exactly what the harness `_run_turn`
does, but for a single cheap turn.

Usage: PYTHONPATH=. .venv/bin/python tools/smoke_session_id.py
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
import urllib.request

from dotenv import load_dotenv

load_dotenv()

from pydantic_ai import Agent  # noqa: E402

from paf.observability.instrumentation import (  # noqa: E402
    root_span,
    setup_observability,
    shutdown_observability,
)


def main() -> None:
    traced = setup_observability()
    if not traced:
        raise SystemExit("tracing not configured -- cannot verify session id")

    # A clean, unique session id (hash of a timestamp).
    sid = "smoke-" + hashlib.sha1(str(time.time()).encode()).hexdigest()[:10]
    print("session id (clean):", sid)

    agent = Agent(
        "openrouter:deepseek/deepseek-v4-flash-0731",
    )
    with root_span(
        "smoke.session.verify",
        **{
            "langfuse.session.id": sid,
            "langfuse.observation.input": _json_dumps_d({"prompt": "Reply OK"}),
        },
    ) as span:
        result = agent.run_sync("Reply with exactly: OK")
        # Note: pydantic-ai does not emit the output attribute here; we set it
        # manually the way runtime.stream does, to keep the trace well-formed.
        from paf.observability.instrumentation import _json_dumps

        span.set_attribute("langfuse.observation.output", _json_dumps({"output": result.output}))

    shutdown_observability()  # flush before we query
    time.sleep(2)  # allow OTLP batch flush + Langfuse ingestion

    # Get Langfuse credentials from paf settings (same source the exporter uses).
    from paf.config import settings

    host = settings.otlp_endpoint.removesuffix("/api/public/otel/v1/traces")
    auth = base64.b64encode(
        f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
    ).decode()
    req = urllib.request.Request(
        f"{host}/api/public/traces?sessionId={sid}&limit=10",
        headers={"Authorization": f"Basic {auth}"},
    )
    data = json.load(urllib.request.urlopen(req)).get("data", [])
    print(f"traces under CLEAN sessionId '{sid}': {len(data)}")
    stored = [t.get("sessionId") for t in data]
    print("stored ids:", stored)
    ok = any(s == sid for s in stored)
    print("RESULT:", "PASS -- session id lands raw and groups" if ok
          else "FAIL -- session id did not match clean value")
    raise SystemExit(0 if ok else 1)


def _json_dumps_d(value) -> str:
    from paf.observability.instrumentation import _json_dumps

    return _json_dumps(value)


if __name__ == "__main__":
    main()
