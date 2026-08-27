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

    # Where the Memory capability + pre-eviction consolidation owner write their
    # readable `.md` bodies + `.memory-state.json` CAS sidecar. Relative paths
    # resolve against PROJECT_ROOT like persistence_dir. Overridable via env /
    # .env (MEMORY_DIR).
    memory_dir: str = "data/memory"

    # Pre-eviction memory consolidation: when true (CONSOLIDATE / .env), the
    # sliding-window compaction tier folds the about-to-be-evicted slice into
    # durable memory via the writable owner sub-agent before dropping it.
    consolidate: bool = False

    # Model for the consolidation owner sub-agent. Blank (default) → resolved to
    # a cheap default constant (see paf.capabilities.memory._consolidate);
    # set via CONSOL_MODEL / .env to override per run. Not the spec'd subject
    # model — the owner is a separate leaf agent.
    consol_model: str = ""

    # OpenRouter routing pin for the consolidation owner (provider-order +
    # quantization hard filter). Mirrors the benchmark's SUBJECT_*/JUDGE_* knobs
    # so the owner resolves off deepseek fp8 boxes deterministically instead of
    # OpenRouter's default throughput sort (which bounces across providers and
    # gives run-to-run nondeterminism). Defaults to the subject's provider order
    # + fp8; set empty (CONSOL_PROVIDER_ORDER='') to disable the routing pin.
    # Set via CONSOL_PROVIDER_ORDER / CONSOL_PROVIDER_QUANTIZATIONS /
    # CONSOL_SEED / .env.
    consol_provider_order: list[str] = ["deepinfra", "parasail", "coreweave"]
    consol_provider_quantizations: list[str] = ["fp8"]
    consol_seed: int | None = None

    # Model for the memory-organizer sub-agent (index maintenance). Blank
    # (default) → resolved to the SAME model as the consolidation owner
    # (`consol_model` / its cheap default) — the organizer is a separate leaf
    # agent that can later move to a higher-level model via ORGANIZE_MODEL /
    # .env. Not the spec'd subject model, and NOT hardcoded in source.
    organize_model: str = ""

    # Egress allowlist for the Browser capability (build.py
    # `_build_browser` → `allowed_domains`). Local dev hosts by default so
    # the headless browser can reach loopback (paf web UI / Vite); add an
    # external staging/known host here (PLAYWRIGHT_ALLOWED_DOMAINS / .env, a
    # JSON-encoded list, e.g. ["localhost","example.com"]) to let the browser
    # navigate it too.
    # `block_private_addresses=True` stays on regardless, so only these named
    # hosts are reachable — keep the list as small as SSRF exposure allows.
    playwright_allowed_domains: list[str] = ["localhost", "127.0.0.1", "::1"]

    # OpenRouter routing pin for the memory organizer, mirroring the `consol_*`
    # knobs (provider-order + quantization hard filter). Defaults to the owner's
    # provider order + fp8; set empty (ORGANIZE_PROVIDER_ORDER='') to disable
    # the routing pin. Set via ORGANIZE_PROVIDER_ORDER /
    # ORGANIZE_PROVIDER_QUANTIZATIONS / ORGANIZE_SEED / .env.
    organize_provider_order: list[str] = ["deepinfra", "parasail", "coreweave"]
    organize_provider_quantizations: list[str] = ["fp8"]
    organize_seed: int | None = None

    # The derived one-line-per-turn index lives NEXT TO the step store, as its own
    # append-only JSONL surface. It is now backed by the agent-foundation
    # `StorageService` seam (M2): this value is the seam's *base directory* for the
    # turn-index `FlatFileStorage`, and the journal lands at
    # `data/.turn-index/<hash(.turn-index)>/turn_index.jsonl`. Relative paths
    # resolve against PROJECT_ROOT like persistence_dir. Overridable via env / .env
    # (TURN_INDEX_DIR), consistent with persistence_dir.
    turn_index_dir: str = "data/.turn-index"

    # Vault capability flat-file index root. A DIRECTORY of per-vault-root JSON
    # artifacts (`<this dir>/<sha256(vault_root)>/{index,snapshot}.json`), keyed
    # by the vault root's normalized absolute path so every distinct vault —
    # the playground, a per-project temp vault, or a pointed-at Obsidian
    # subfolder — gets its own stable index area while the artifacts stay in the
    # paf repo, NOT co-located inside the vault. Relative paths resolve against
    # PROJECT_ROOT like persistence_dir; gitignored via the `data` entry.
    # Overridable via env / .env (VAULT_INDEX_DIR). An explicit
    # `FlatFileVaultStorage(index_root=...)` still wins over this default.
    vault_index_dir: str = "data/.paf-vault-index"

    # Observability (Langfuse OTLP). Provide these in .env / the environment —
    # they are secrets + host config, NOT meant to be entered here. All three
    # blank = tracing off entirely (setup_observability no-ops); providing keys
    # (and an endpoint) turns traces on. pydantic-settings merges .env values
    # over these empty defaults at import time.
    otlp_endpoint: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


settings = Settings()
