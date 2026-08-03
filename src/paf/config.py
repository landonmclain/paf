"""Config + secrets loading via pydantic-settings (env + .env)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration. Reads from environment + .env (gitignored)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Model to use for agents (provider-qualified, e.g. "openai:gpt-4o").
    model_name: str = "openai:gpt-4o-mini"

    # Observability / Langfuse OTLP endpoint + keys (empty = tracing disabled).
    otlp_endpoint: str = "http://localhost:3000/api/public/otel/v1/traces"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # Storage root for flat-file state.
    storage_root: str = "data/state"


settings = Settings()
