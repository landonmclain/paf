"""Config + secrets loading via pydantic-settings (env + .env)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchored to the repo root (…/src/paf/config.py → …/), not the process CWD.
# This keeps runtime state in one canonical place no matter where the CLI is
# launched from, and stays overridable via STORAGE_ROOT env var / .env.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """App configuration. Reads from environment + .env (gitignored)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Model to use for agents (provider-qualified, e.g. "openai:gpt-4o").
    # M1 placeholder (D5) — transient; specs/*.yaml hold per-agent overrides.
    model_name: str = "openai:gpt-5.4-nano"

    # Observability / Langfuse OTLP endpoint + keys (empty = tracing disabled).
    otlp_endpoint: str = "http://localhost:3000/api/public/otel/v1/traces"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # Storage root for flat-file state. Default anchors to the repo root
    # regardless of CWD (see PROJECT_ROOT); overridable via STORAGE_ROOT.
    storage_root: Path = PROJECT_ROOT / "data" / "state"


settings = Settings()
