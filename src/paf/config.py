"""Config + secrets loading via pydantic-settings (env + .env)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchored to the repo root (…/src/paf/config.py → …/), not the process CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Declarative agent config (model, instructions, capabilities).
SPECS_DIR = PROJECT_ROOT / "specs"


class Settings(BaseSettings):
    """App configuration. Reads from environment + .env (gitignored)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Persistence store root (a directory of flat files). Relative paths
    # resolve against PROJECT_ROOT regardless of launch CWD.
    persistence_dir: str = "data/paf"

    # Where the Memory capability + consolidation owner write their readable
    # `.md` bodies + `.memory-state.json` CAS sidecar.
    memory_dir: str = "data/memory"

    # Observability (Langfuse OTLP). Provide in .env / environment — secrets +
    # host config, not meant to be entered here. All blank = tracing off.
    otlp_endpoint: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # The user's time zone (IANA name, e.g. America/Chicago). Times the model
    # reads are shown in this zone; storage stays UTC (see paf.clock). Blank =
    # the machine's zone.
    timezone: str = ""


settings = Settings()
