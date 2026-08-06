"""Config + secrets loading via pydantic-settings (env + .env)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchored to the repo root (…/src/paf/config.py → …/), not the process CWD.
# Keeps runtime state (e.g. the persistence DB) in one canonical place no matter
# where the CLI is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical spec location: declarative agent config (model, instructions,
# capabilities). Loaded via Agent.from_file with harness capabilities
# registered as custom_capability_types.
SPECS_DIR = PROJECT_ROOT / "specs"
ASSISTANT_SPEC = SPECS_DIR / "assistant.yaml"


class Settings(BaseSettings):
    """App configuration. Reads from environment + .env (gitignored)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Model to use for agents (provider-qualified, e.g. "openai:gpt-4o").
    # M1 placeholder (D5) — transient; specs/*.yaml hold per-agent overrides.
    model_name: str = "openai:gpt-5.4-nano"

    # Persistence DB path. Relative paths resolve against PROJECT_ROOT so the
    # store lands in one canonical place regardless of launch CWD.
    db_path: str = "data/paf.db"

    # Observability / Langfuse OTLP endpoint + keys (empty = tracing disabled).
    otlp_endpoint: str = "http://localhost:3000/api/public/otel/v1/traces"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


settings = Settings()
