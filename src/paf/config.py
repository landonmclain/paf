"""Config + secrets loading via pydantic-settings (env + .env)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchored to the repo root (…/src/paf/config.py → …/), not the process CWD.
# Keeps runtime state and spec files in one canonical place no matter where the
# CLI is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical spec location: declarative agent config (model, instructions,
# capabilities). Loaded via Agent.from_file with harness capabilities
# registered as custom_capability_types.
SPECS_DIR = PROJECT_ROOT / "specs"


class Settings(BaseSettings):
    """App configuration. Reads from environment + .env (gitignored)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Persistence store root (a DIRECTORY of flat files). Relative paths resolve
    # against PROJECT_ROOT so the store lands in one canonical place regardless
    # of launch CWD.
    persistence_dir: str = "data/paf"

    # Observability (Langfuse OTLP). Provide these in .env / the environment —
    # they are secrets + host config, NOT meant to be entered here. All three
    # blank = tracing off entirely (setup_observability no-ops); providing keys
    # (and an endpoint) turns traces on. pydantic-settings merges .env values
    # over these empty defaults at import time.
    otlp_endpoint: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


settings = Settings()
