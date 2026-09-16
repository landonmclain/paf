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

    # Model for the consolidation owner sub-agent. Blank → a cheap default
    # (see paf.capabilities.memory._consolidate). Not the spec'd subject model.
    consol_model: str = ""

    # OpenRouter routing pin for the consolidation owner (provider order +
    # quantization hard filter). Empty list disables the pin.
    consol_provider_order: list[str] = ["deepinfra", "parasail", "coreweave"]
    consol_provider_quantizations: list[str] = ["fp8"]
    consol_seed: int | None = None

    # Observability (Langfuse OTLP). Provide in .env / environment — secrets +
    # host config, not meant to be entered here. All blank = tracing off.
    otlp_endpoint: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


settings = Settings()
