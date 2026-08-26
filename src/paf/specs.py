"""Spec loader — resolve agent names to YAML specs and build agents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic_ai import Agent, AgentSpec
from pydantic_ai.settings import ModelSettings

from paf.capabilities import ASSISTANT_PROFILE, AgentProfile
from paf.capabilities.build import build_runtime_capabilities
from paf.capabilities.compaction import CompactionConfig
from paf.config import SPECS_DIR

DEFAULT_SPEC_DIRECTORY = SPECS_DIR


# ---------------------------------------------------------------------------
# OpenRouter determinism pin (mirrors tests/framework.py `_openrouter_model_settings`).
# Prod shares the benchmark's reproducibility contract: temp 0, top_p 1.0, a fixed
# sampler seed, and a pinned provider chain with hard quantization filter. Without
# the pin, `:nitro`-style throughput routing silently bounces across upstream
# providers that serve the same checkpoint at different quantizations (fp8 vs fp4)
# -> real output nondeterminism at temp 0. Defaults are the DeepSeek subject chain;
# all env-overridable (empty string disables the pin). See vault
# OpenRouter-Provider-Quantization-Nondeterminism.md.
# ---------------------------------------------------------------------------
SUBJECT_PROVIDER_ORDER = [
    p for p in os.environ.get(
        "LME_SUBJECT_PROVIDER_ORDER", "deepinfra,parasail,coreweave"
    ).split(",") if p
]
SUBJECT_PROVIDER_QUANTIZATIONS = [
    q for q in os.environ.get(
        "LME_SUBJECT_PROVIDER_QUANTIZATIONS", "fp8"
    ).split(",") if q
]
LME_SAMPLER_SEED = int(os.environ.get("LME_SAMPLER_SEED", "20260815"))


def _openrouter_model_settings(
    provider_order: list[str] | None = SUBJECT_PROVIDER_ORDER,
    quantizations: list[str] | None = SUBJECT_PROVIDER_QUANTIZATIONS,
    seed: int | None = LME_SAMPLER_SEED,
) -> ModelSettings:
    """Build `model_settings` that pin OpenRouter routing + sampler determinism.

    Returns a dict combining sampling determinism (`temperature: 0`, `top_p: 1.0`,
    fixed seed) with the OpenRouter `provider` object passed via `extra_body`.
    When `provider_order` is empty/None, no routing pin is applied (plain request).
    pydantic-ai's `openai_chat` passes `extra_body` verbatim to the HTTP body.
    `Agent.from_spec` shallow-merges these over the spec YAML's `model_settings`,
    so the YAML's `openrouter_cache_*` keys survive alongside `extra_body`.
    """
    out: dict = {"temperature": 0}
    out["top_p"] = 1.0
    if seed is not None:
        out["seed"] = seed
    pinned_quantizations = [q for q in (quantizations or []) if q]
    if provider_order:
        provider: dict = {"order": list(provider_order)}
        if pinned_quantizations:
            provider["quantizations"] = pinned_quantizations
        extra_body: dict = dict(out.get("extra_body") or {})
        extra_body["provider"] = provider
        out["extra_body"] = extra_body
    return cast(ModelSettings, out)


def _spec_compaction_config(spec_path: Path) -> CompactionConfig:
    """Read the spec's top-level ``compact:`` block into a ``CompactionConfig``.

    pydantic-ai's ``AgentSpec`` drops unknown top-level keys, so `build_agent`
    reads the raw YAML separately for the per-agent compaction tuning block.

    Absent block (or unreadable YAML) -> ``CompactionConfig()`` (all-None),
    which makes ``build_compaction`` use its benchmark-tuned defaults. Unknown
    keys *inside* ``compact:`` are ignored so the schema can grow without
    breaking older spec files.
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return CompactionConfig()
    if not isinstance(data, dict):
        return CompactionConfig()
    block = data.get("compact")
    if not isinstance(block, dict):
        return CompactionConfig()
    return CompactionConfig.from_mapping(block)


def _spec_web_config(spec_path: Path) -> list[str] | None:
    """Read the spec's top-level ``web:`` block into an extract domain list.

    Mirrors ``_spec_compaction_config``: pydantic-ai's ``AgentSpec`` drops
    unknown top-level keys, so `build_agent` reads the raw YAML separately for
    the per-agent web trust boundary. The block shape:

        web:
          search_domains:
            - docs.pydantic.dev
            - github.com

    ``search_domains`` is the **extract** allowlist for ``fetch_web_page``
    (which domains may be pulled into context). Open web *search* is never
    restricted by it.

    Absent/unreadable block or empty list -> None, which makes
    ``build_runtime_capabilities`` -> ``_build_web`` fall back to the
    ``PREFERRED_DOC_DOMAINS`` default. Unknown keys inside ``web:`` are ignored
    so the schema can grow without breaking older spec files.
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    block = data.get("web")
    if not isinstance(block, dict):
        return None
    domains = block.get("search_domains")
    if not isinstance(domains, list) or not domains:
        return None
    return [str(d) for d in domains]


def _spec_browser_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``browser:`` block into a Browser config dict.

    Mirrors ``_spec_compaction_config`` / ``_spec_web_config``: pydantic-ai's
    ``AgentSpec`` drops unknown top-level keys, so `build_agent` reads the raw
    YAML separately for per-agent browser egress policy. The block shape:

        browser:
          allowed_domains:
            - localhost
            - staging.example.com

    ``allowed_domains`` is the Browser capability's egress allowlist — hosts
    (and their subdomains) the browser may reach. Only ``allowed_domains`` is
    exposed from the spec; every other Browser field (headless, mobile, sandbox,
    …) stays a code-level default. A non-empty present list **wins** (overrides
    the env-driven ``PLAYWRIGHT_ALLOWED_DOMAINS`` default); absent block, non-dict
    block, or empty list -> ``None``, so `_build_browser` falls back to the env
    default. Unknown keys inside ``browser:`` are ignored so the schema can grow
    without breaking older spec files.
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    block = data.get("browser")
    if not isinstance(block, dict):
        return None
    domains = block.get("allowed_domains")
    if not isinstance(domains, list) or not domains:
        return None
    return {"allowed_domains": [str(d) for d in domains]}


class AgentSpecNotFoundError(FileNotFoundError):
    """Raised when an agent specification cannot be found."""


def get_spec_path(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
) -> Path:
    """Resolve an agent name to a YAML/JSON specification path.

    Tries ``{name}.yaml``, ``{name}.yml``, then ``{name}.json`` in order.
    """
    for suffix in (".yaml", ".yml", ".json"):
        path = spec_directory / f"{agent_name}{suffix}"
        if path.is_file():
            return path
    raise AgentSpecNotFoundError(
        f"No agent specification found for {agent_name!r} in {spec_directory}"
    )


def load_agent_spec(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
) -> AgentSpec:
    """Load and validate an agent specification from disk."""
    path = get_spec_path(agent_name, spec_directory)
    return AgentSpec.from_file(path)


def build_agent(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
    profile: AgentProfile | None = None,
    **overrides: Any,
) -> Agent[Any, Any]:
    """Construct an agent from a named spec, with optional overrides.

    ```agent_name`` is the spec file base name (``assistant`` → ``assistant.yaml``).

    ``profile`` is the capability allowlist for this agent kind — resolved to
    ``custom_capability_types`` so the spec's ``capabilities:`` entries can only
    come from the allowlisted catalog. Defaults to ``ASSISTANT_PROFILE``, M1's
    one real agent kind.

    ``build_runtime_capabilities()`` assembles capabilities that cannot resolve
    from YAML (the tiered ``TieredCompaction``, whose nested strategy objects have
    no ``from_spec``) and passes them via ``Agent.from_spec(..., capabilities=)``,
    which merges them with any spec-declared capabilities (e.g. ``StepPersistence``).

    Overrides are passed as kwargs to ``Agent.from_spec``. Supported:
    ``model``, ``instructions``, ``model_settings``, ``name``, ``output_type``.
    Tools and spec-declarable capabilities must be declared in the spec YAML.
    If ``model_settings`` is not among the overrides, the OpenRouter determinism
    pin (``_openrouter_model_settings``) is applied by default and shallow-merged
    over the spec YAML's ``model_settings`` by ``Agent.from_spec``.
    """
    profile = profile or ASSISTANT_PROFILE
    spec = load_agent_spec(agent_name, spec_directory)
    spec_path = get_spec_path(agent_name, spec_directory)
    # Per-agent compaction tuning lives in the spec YAML under a top-level
    # `compact:` block (option B — diffable, reviewable, per-agent). pydantic-ai's
    # `AgentSpec` silently drops unknown top-level keys, so we read the raw YAML
    # off disk to extract `compact:` (lenient: unknown keys inside ignored;
    # absent block -> CompactionConfig() -> builder's benchmark defaults).
    compaction_config = _spec_compaction_config(spec_path)
    # Per-assistant web trust boundary: the spec's top-level `web:` block
    # declares the extract allowlist for `fetch_web_page`; absent -> the
    # PREFERRED_DOC_DOMAINS default (see _spec_web_config). Open search is open.
    web_config = _spec_web_config(spec_path)
    # Per-assistant browser egress allowlist: the spec's top-level `browser:`
    # block overrides the env-driven PLAYWRIGHT_ALLOWED_DOMAINS default when it
    # declares a non-empty `allowed_domains` (see _spec_browser_config).
    browser_config = _spec_browser_config(spec_path)
    # Default: apply the OpenRouter provider/quantization pin (temp 0,
    # top_p 1.0, seed, pinned provider chain) unless the caller supplies their own
    # `model_settings` override. Agent.from_spec shallow-merges this over the spec's
    # YAML model_settings, so the YAML's openrouter_cache_* keys are preserved.
    if "model_settings" not in overrides:
        overrides.setdefault("model_settings", _openrouter_model_settings())
    return Agent.from_spec(
        spec,
        # Resolve the agent kind's allowlist to the capability TYPES the loader
        # registers. Harness capabilities (StepPersistence) aren't in pydantic-ai's
        # default registry; allowlisting them here is what lets the YAML
        # `capabilities:` entry resolve.
        custom_capability_types=profile.capability_types(),
        # Capabilities that only Python can construct merge here with any
        # spec-declared ones via Agent.from_spec. The profile is the gatekeeper:
        # only capabilities in its allowlist are built (see build.py). The spec's
        # `compact:` block (if any) drives the compaction tiers.
        capabilities=build_runtime_capabilities(profile, compaction_config, web_config, browser_config),
        **overrides,
    )
