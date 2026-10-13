"""Generate a VAPID keypair for Web Push and print the .env lines.

    PYTHONPATH= .venv/bin/python scripts/gen_vapid.py            # print
    PYTHONPATH= .venv/bin/python scripts/gen_vapid.py --write    # append to .env

Refuses to overwrite existing PAF_VAPID_* keys: rotating the keypair invalidates
every existing browser subscription (they are bound to the public key).
"""

from __future__ import annotations

import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from py_vapid import Vapid

from paf.push import b64url_encode

SUBJECT = "mailto:you@example.com"
ENV = Path(__file__).resolve().parent.parent / ".env"


def generate() -> dict[str, str]:
    v = Vapid()
    v.generate_keys()
    private = v.private_key.private_numbers().private_value.to_bytes(32, "big")
    public = v.public_key.public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return {
        "PAF_VAPID_PRIVATE_KEY": b64url_encode(private),
        "PAF_VAPID_PUBLIC_KEY": b64url_encode(public),
        "PAF_VAPID_SUBJECT": SUBJECT,
    }


def main() -> None:
    lines = [f"{k}={v}" for k, v in generate().items()]
    if "--write" not in sys.argv:
        print("\n".join(lines))
        return
    existing = ENV.read_text() if ENV.exists() else ""
    if "PAF_VAPID_PRIVATE_KEY=" in existing:
        sys.exit("PAF_VAPID_* already set in .env — refusing to rotate the keypair.")
    sep = "" if existing.endswith("\n") or not existing else "\n"
    ENV.write_text(
        existing
        + sep
        + "\n# Web Push (VAPID) — scripts/gen_vapid.py. Do NOT rotate: it\n"
        "# invalidates every device's push subscription.\n" + "\n".join(lines) + "\n"
    )
    print(f"wrote PAF_VAPID_* to {ENV} (private key not shown)")


if __name__ == "__main__":
    main()
