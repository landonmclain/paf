"""Spec loader — resolve agent names to YAML specs and build agents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import yaml
from openai import AsyncOpenAI
from openai import Timeout as OpenAITimeout
from pydantic_ai import Agent, AgentSpec
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

from paf.capabilities import ASSISTANT_PROFILE, AgentProfile
from paf.capabilities.build import build_runtime_capabilities
from paf.capabilities.compaction import CompactionConfig
from paf.config import SPECS_DIR

DEFAULT_SPEC_DIRECTORY = SPECS_DIR


# ---------------------------------------------------------------------------
# OpenRouter determinism pin — fallback defaults for specs that don't declare
# their own `model_settings.extra_body.provider`. Specs are the source of truth;
# this is used only when a spec's `model_settings` block omits routing. The
# contract: temp 0, top_p 1.0, a fixed sampler seed, and a pinned provider chain
# with hard quantization filter. Without the pin, `:nitro`-style throughput
# routing silently bounces across upstream providers that serve the same
# checkpoint at different quantizations (fp8 vs fp4) -> real nondeterminism at
# temp 0. Defaults are the DeepSeek subject chain; all env-overridable (empty
# string disables the pin). See vault OpenRouter-Provider-Quantization-Nondeterminism.md.
# ---------------------------------------------------------------------------
SUBJECT_PROVIDER_ORDER = [
    p for p in os.environ.get(
        "SUBJECT_PROVIDER_ORDER", "deepinfra,parasail,coreweave"
    ).split(",") if p
]
SUBJECT_PROVIDER_QUANTIZATIONS = [
    q for q in os.environ.get(
        "SUBJECT_PROVIDER_QUANTIZATIONS", "fp8"
    ).split(",") if q
]
SAMPLER_SEED = int(os.environ.get("SAMPLER_SEED", "20260815"))


def _openrouter_model_settings(
    provider_order: list[str] | None = SUBJECT_PROVIDER_ORDER,
    quantizations: list[str] | None = SUBJECT_PROVIDER_QUANTIZATIONS,
    seed: int | None = SAMPLER_SEED,
) -> ModelSettings:
    """Build `model_settings` that pin OpenRouter routing + sampler determinism.

    Returns a dict combining sampling determinism (`temperature: 0`, `top_p: 1.0`,
    fixed seed) with the OpenRouter `provider` object passed via `extra_body`.
    When `provider_order` is empty/None, no routing pin is applied (plain request).
    pydantic-ai's `openai_chat` passes `extra_body` verbatim to the HTTP body.
    This is a FALLBACK for specs that don't declare their own `model_settings.extra_body.provider`;
    a spec that declares one is the source of truth and wins (see `build_agent`).
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


# ---------------------------------------------------------------------------
# Bounded-timeout OpenRouter client — fix for the "stuck turn" hang.
#
# pydantic-ai resolves an ``openrouter:`` model string via the default provider,
# which builds an openai ``AsyncOpenAI`` client with SDK defaults: 600s read
# timeout and ``max_retries=2``. Against OpenRouter that is a trap: traffic is
# pinned to a provider chain (``extra_body.provider.order``, e.g. deepinfra
# first), and a provider that goes *slow-but-alive* (rather than erroring)
# makes OpenRouter hold the request indefinitely — its ``allow_fallbacks`` only
# fires on provider errors, not hangs. The SDK's 2 blind retries then route
# right back to the SAME stalled provider, so the effective worst case is
# ~600s x several attempts (observed 13m26s before a user cancelled).
#
# This builds an OpenRouterModel whose provider wraps a custom AsyncOpenAI
# client with a bounded read timeout and ``max_retries=0``: the stall now
# surfaces as a clean ModelAPIError after ~120s instead of a 10+ min freeze,
# and *we* own retry policy (currently: none at the SDK layer) rather than the
# SDK's same-provider blind retries.
# ---------------------------------------------------------------------------
OPENROUTER_CLIENT_TIMEOUT_S = 120.0
OPENROUTER_CLIENT_CONNECT_S = 5.0


def _openrouter_model_with_client(model_name: str) -> OpenRouterModel | None:
    """Build an ``OpenRouterModel`` wrapping a bounded-timeout, no-retry client.

    ``model_name`` is the bare OpenRouter model slug (without the
    ``openrouter:`` prefix, e.g. ``deepseek/deepseek-v4-flash-0731``). The
    returned model's provider reuses an ``AsyncOpenAI`` client pointed at
    OpenRouter with ``max_retries=0`` and a ~120s read timeout, so a pinned
    provider that goes slow-but-alive fails fast instead of freezing the turn.

    The OpenRouter routing pin (``extra_body.provider.order`` + quantizations)
    is NOT touched here — it continues to live in ``model_settings`` (see
    ``_openrouter_model_settings``) and flows through each request verbatim.

    If ``OPENROUTER_API_KEY`` is missing, this returns ``None`` so callers can
    fall back to the plain string path (whose own missing-key UserError then
    applies) rather than constructing a key-less AsyncOpenAI up front.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=OpenAITimeout(
            OPENROUTER_CLIENT_TIMEOUT_S, connect=OPENROUTER_CLIENT_CONNECT_S
        ),
        max_retries=0,
    )
    return OpenRouterModel(
        model_name, provider=OpenRouterProvider(openai_client=client)
    )


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
          block_private_addresses: false   # optional; default true

    ``allowed_domains`` is the Browser capability's egress allowlist — hosts
    (and their subdomains) the browser may reach. ``block_private_addresses``
    is the **mechanical** SSRF guard (refuses loopback/RFC-1918/link-local and
    cloud-metadata addresses whether or not an allowlist names them). It
    defaults to ``True`` — secure by default; only an explicit ``false`` here
    loosens it so an allowlisted *private* host (local dev boxes, staging LAN
    hosts) actually becomes reachable.

    These are two independent trust decisions and are deliberately **not**
    conflated: listing ``localhost`` in ``allowed_domains`` does **not**
    release the private-address guard on its own (the guard outranks the
    allowlist). To reach a local host the operator must BOTH name it in
    ``allowed_domains`` AND set ``block_private_addresses: false``. This keeps
    the secure default for every agent that merely inherits a ``browser:``
    block, and makes "reach my own internal network" a conscious, explicit
    choice.

    ``cloud-metadata`` (``169.254.169.254`` — the IAM-credential theft vector
    on cloud hosts) is **never** released by this flag and stays refused; see
    ``EgressPolicy``. Unknown keys inside ``browser:`` are ignored so the
    schema can grow without breaking older spec files.

    Returns a dict (or ``None`` if the block is absent or unreadable). The
    dict carries whatever was declared; absent members fall back to their
    code-level defaults in `_build_browser`.
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
    cfg: dict[str, object] = {}
    domains = block.get("allowed_domains")
    if isinstance(domains, list) and domains:
        cfg["allowed_domains"] = [str(d) for d in domains]
    block_private = block.get("block_private_addresses")
    if isinstance(block_private, bool):
        cfg["block_private_addresses"] = block_private
    if not cfg:
        return None
    return cfg


def _spec_vault_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``vault:`` block into a Vault config dict.

    Mirrors ``_spec_browser_config`` / ``_spec_web_config``: pydantic-ai's
    ``AgentSpec`` drops unknown top-level keys, so `build_agent` reads the raw
    YAML separately for per-agent vault config. The block shape:

        vault:
          root: ~/Documents/notes   # `~` expands via os.path.expanduser
          allow:            # optional; absent = whole root
            - Notes/**
          deny:             # optional; always wins over allow
            - "**/.trash/**"

    ``root`` is the vault directory the capability confines itself to; ``allow``
    /``deny`` are optional glob subtree patterns expressing a workspace-vs-personal
    boundary inside that root. Unknown keys inside ``vault:`` are ignored so the
    schema can grow without breaking older spec files.

    Returns a dict (or ``None`` when the block is absent/unreadable). Absent
    members fall back to their code-level defaults in ``_build_vault`` (whose
    default root is the paf-owned ``data/vault/`` playground — the real vault is
    ONLY reached via an explicit ``root`` here).
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    block = data.get("vault")
    if not isinstance(block, dict):
        return None
    cfg: dict[str, object] = {}
    root = block.get("root")
    if isinstance(root, str) and root:
        cfg["root"] = os.path.expanduser(root)
    for key in ("allow", "deny"):
        val = block.get(key)
        if isinstance(val, list) and val:
            cfg[key] = [str(v) for v in val]
    if not cfg:
        return None
    return cfg


def _spec_skills_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``skills:`` block into a Skills config dict.

    Mirrors ``_spec_vault_config`` / ``_spec_browser_config``: pydantic-ai's
    ``AgentSpec`` drops unknown top-level keys, so `build_agent` reads the raw
    YAML separately for per-agent skill-library config. The block shape:

        skills:
          libraries:        # one or more full paths (~ expands); absent -> the
            - ~/.hermes/skills          # paf-owned data/skills/ default
            - ~/projects/Personal/skill-library
          top_k: 3           # optional; default number of point_skills hints

    ``libraries`` are **full absolute paths** (``~`` expanded via
    ``os.path.expanduser``); each is a directory scanned for ``SKILL.md`` skills
    (a skill tree: ``<root>/<category>/<skill>/SKILL.md`` or flat). ``top_k`` is
    the default number of ``point_skills`` hint pointers returned per call.
    Unknown keys inside ``skills:`` are ignored so the schema can grow without
    breaking older spec files.

    Returns a dict (or ``None`` when the block is absent/unreadable). Absent
    members fall back to their code-level defaults in ``_build_skills`` (whose
    default library is the paf-owned ``data/skills/`` — no implicit disk
    scanning beyond what a spec names explicitly).
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    block = data.get("skills")
    if not isinstance(block, dict):
        return None
    cfg: dict[str, object] = {}
    libraries = block.get("libraries")
    if isinstance(libraries, list) and libraries:
        cfg["libraries"] = [os.path.expanduser(str(v)) for v in libraries if str(v)]
    top_k = block.get("top_k")
    if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0:
        cfg["top_k"] = top_k
    if not cfg:
        return None
    return cfg


def _spec_memory_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``memory:`` block into a Memory config dict.

    Mirrors ``_spec_vault_config`` / ``_spec_browser_config``: pydantic-ai's
    ``AgentSpec`` drops unknown top-level keys, so `build_agent` reads the raw
    YAML separately for per-agent memory tuning. The block shape:

        memory:
          consolidate_every: 10      # optional; background consolidation cadence
          max_memory_size: 65536     # optional; per-read char cap for MEMORY.md
          tier_caps:                 # optional; per-tier COUNT caps (render-owned)
            t1: 75
            t2: 25
            t3: 15

    ``consolidate_every`` is the background-consolidation cadence in **minutes**
    — a time metric (time-anchored watermark; monotonic across runs). It is
    decoupled from context eviction; the owner simply folds at least this often
    when new material has accumulated. ``max_memory_size`` bounds how much of
    the rendered journal is injected per read (see ``JournalViewStore.read``).
    ``tier_caps`` (a ``{t1/t2/t3: int|None}`` dict) are PER-TIER COUNT caps on
    the injected index render, owned by the render itself — see
    ``paf.specs.resolve_memory_tier_caps`` (read here, not in this scalar loop).

    ``consolidate_every`` / ``max_memory_size`` live on **Memory** — the
    capability owns every memory config variable — and are handed to the
    background queue (a dumb service) only as injected construction args, never
    read by the queue itself. The queue's ``write_render`` forwards its own
    ``agent_name`` so the render resolves ``tier_caps`` from the spec directly.

    Unknown keys inside ``memory:`` are ignored so the schema can grow without
    breaking older spec files.

    Returns a dict (or ``None`` when the block is absent/unreadable). Absent
    members fall back to their code-level defaults in ``_build_memory`` / the
    ``Memory`` capability (``consolidate_every`` defaults to
    ``DEFAULT_CONSOLIDATE_EVERY``; ``max_memory_size`` to ``DEFAULT_MEMORY_MAX_SIZE``;
    ``tier_caps`` to ``_render._TIER_CAPS``).
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    block = data.get("memory")
    if not isinstance(block, dict):
        return None
    cfg: dict[str, object] = {}
    for key in (
        "consolidate_every",
        "review_every",
        "max_memory_size",
    ):
        value = block.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            cfg[key] = value
    if not cfg:
        return None
    return cfg


def resolve_memory_config(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
) -> dict:
    """Resolve the spec's ``memory:`` block into the values a consumer needs.

    Public seam the runtime Session uses to reach the background-consolidation
    cadence WITHOUT poking pydantic-ai's private capability registry. It reads
    the exact same ``_spec_memory_config`` path ``_build_memory`` uses and
    applies the same code-level default, so the Session's trigger cadence can
    never drift from the ``Memory`` capability's ``consolidate_every`` — Memory
    stays the owner, this is just a non-invasive read of the resolved value.

    Returns ``{"consolidate_every": int, "review_every": int, "consolidate_turns":
    int, "review_turns": int}`` (always present, defaults applied). The minute
    cadences remain for explicitness/back-compat; the TURN-count values are the
    active gate (DECIDED 2026-09-14, see Designs/Turn-Based-Background-Cadence).
    ``review_turns`` defaults to ``DEFAULT_REVIEW_TURNS``.
    """
    from paf.capabilities.build import (
        DEFAULT_CONSOLIDATE_EVERY,
        DEFAULT_CONSOLIDATE_TURNS,
        DEFAULT_REVIEW_EVERY,
        DEFAULT_REVIEW_TURNS,
    )

    every = DEFAULT_CONSOLIDATE_EVERY
    review_every = DEFAULT_REVIEW_EVERY  # reviewer runs less often than the owner
    turns = DEFAULT_CONSOLIDATE_TURNS
    review_turns = DEFAULT_REVIEW_TURNS
    try:
        cfg = _spec_memory_config(get_spec_path(agent_name, spec_directory))
    except Exception:  # noqa: BLE001 - spec read is best-effort sugar
        cfg = None
    if isinstance(cfg, dict):
        candidate = cfg.get("consolidate_every")
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            every = candidate
        review_candidate = cfg.get("review_every")
        if (
            isinstance(review_candidate, int)
            and not isinstance(review_candidate, bool)
            and review_candidate > 0
        ):
            review_every = review_candidate
        turns_c = cfg.get("consolidate_turns")
        if isinstance(turns_c, int) and not isinstance(turns_c, bool) and turns_c > 0:
            turns = turns_c
        review_turns_c = cfg.get("review_turns")
        if (
            isinstance(review_turns_c, int)
            and not isinstance(review_turns_c, bool)
            and review_turns_c > 0
        ):
            review_turns = review_turns_c
    return {
        "consolidate_every": every,
        "review_every": review_every,
        "consolidate_turns": turns,
        "review_turns": review_turns,
    }


def resolve_memory_tier_caps(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
) -> dict[str, int | None] | None:
    """Best-effort read of the spec's ``memory: tier_caps`` block.

    ``tier_caps`` is a per-tier count bound for the injected index render
    (``{t1/t2/t3: int | None}``). It is NOT a single int like the cadence
    values, so it needs its own reader rather than riding inside
    :func:`resolve_memory_config`. The render (``_render.py``) owns resolving
    its own caps and calls this directly — neither the background queue nor
    the runtime Session carries ``tier_caps`` anywhere.

    Returns a **partial** dict of only the per-tier caps the spec declares
    (``None`` meaning unbounded for that tier), or ``None`` when the spec is
    absent/unreadable or declares no usable ``tier_caps`` block. The caller
    merges this over its built-in defaults (``_TIER_CAPS``), so unspecified
    tiers keep the code default — not a surprise unbounded.
    """
    try:
        with get_spec_path(agent_name, spec_directory).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception:  # noqa: BLE001 - spec read is best-effort sugar
        return None
    if not isinstance(data, dict):
        return None
    block = data.get("memory")
    if not isinstance(block, dict):
        return None
    raw = block.get("tier_caps")
    if not isinstance(raw, dict) or not raw:
        return None
    from paf.capabilities.memory._journal import MEMORY_TIERS

    out: dict[str, int | None] = {}
    for tier, cap in raw.items():
        if tier not in MEMORY_TIERS:
            continue  # ignore unknown tiers so schema can grow safely
        if cap is None:
            out[tier] = None  # explicit unbounded
        elif isinstance(cap, int) and not isinstance(cap, bool) and cap > 0:
            out[tier] = cap
    return out or None


def resolve_profile_name(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
) -> str | None:
    """Resolve a spec's declared ``profile:`` key to its profile name.

    The CLI entry points use this to choose the default profile for an agent
    kind when the user doesn't pass one explicitly: a spec declares the profile
    it belongs to (``profile: contextbench`` → ``PROFILES["contextbench"]``), so
    capability selection stays spec-declared and never depends on a hardcoded
    agent-name mapping in each `ui/*.py`.

    This mirrors ``resolve_memory_config`` — a best-effort raw-YAML read of one
    top-level key. pydantic-ai's ``AgentSpec`` drops unknown top-level keys, so
    the value is read off disk directly. Returns ``None`` when the spec is
    absent, unreadable, or declares no ``profile:`` key — callers fall back to
    their own default (specs without a declaration keep working today's way).
    """
    try:
        with get_spec_path(agent_name, spec_directory).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("profile")
    return value if isinstance(value, str) and value else None


def _spec_rooted_config(spec_path: Path, block_name: str, root_key: str) -> dict | None:
    """Read one per-capability root block from a spec into a config dict.

    Shared reader for the root-bearing capabilities (filesystem / shell / git /
    codebase_map). Mirrors ``_spec_vault_config`` / ``_spec_browser_config`` in
    shape and intent: pydantic-ai's ``AgentSpec`` drops unknown top-level keys,
    so `build_agent` reads the raw YAML separately for per-agent scoping.

    The block shape (per capability):

        filesystem:
          root_dir: ~/projects/Personal/landonmclain-site
        shell:
          cwd: ~/projects/Personal/landonmclain-site

    ``root_key`` is the member inside the block that carries the path
    (``root_dir`` / ``cwd`` / ``root`` / ``scan_root``). Only that key is
    validated and copied — unknown keys *inside* the block are ignored so the
    schema can grow (e.g. shell ``allowed_commands`` / ``denied_commands``)
    without breaking older spec files.

    Returns a dict (or ``None`` when the block is absent / unreadable / has no
    usable root). Absent members fall back to the builder's `PROJECT_ROOT`
    default — so a spec block is purely additive and never silently weakens a
    secure default.
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    block = data.get(block_name)
    if not isinstance(block, dict):
        return None
    root = block.get(root_key)
    if not isinstance(root, str) or not root.strip():
        return None
    return {root_key: root}


def _spec_filesystem_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``filesystem:`` block (root_dir)."""
    return _spec_rooted_config(spec_path, "filesystem", "root_dir")


def _spec_shell_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``shell:`` block (cwd + executor).

    Returns a dict with optional ``cwd`` and ``executor`` keys, or ``None``
    when the block is absent / unreadable. The ``executor`` block selects the
    execution backend:

        shell:
          cwd: ~/projects/Personal/landonmclain-site
          executor:
            backend: docker   # local (default) | docker
            mode: exec        # exec (default) | run
            image: python:3.12-slim

    ``backend`` defaults to ``local`` when the executor block is absent or
    names no backend. Unknown keys inside the block are ignored so the schema
    can grow without breaking older spec files.
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    block = data.get("shell")
    if not isinstance(block, dict):
        return None
    config: dict = {}
    cwd = block.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        config["cwd"] = cwd
    executor = block.get("executor")
    if isinstance(executor, dict):
        resolved: dict = {}
        backend = executor.get("backend")
        if isinstance(backend, str) and backend.strip():
            resolved["backend"] = backend
        mode = executor.get("mode")
        if isinstance(mode, str) and mode.strip():
            resolved["mode"] = mode
        image = executor.get("image")
        if isinstance(image, str) and image.strip():
            resolved["image"] = image
        if resolved:
            config["executor"] = resolved
    return config or None


def _spec_git_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``git:`` block (root)."""
    return _spec_rooted_config(spec_path, "git", "root")


def _spec_codebase_map_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``codebase_map:`` block (scan_root)."""
    return _spec_rooted_config(spec_path, "codebase_map", "scan_root")


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
    # Per-assistant vault config: the spec's top-level `vault:` block names the
    # vault root (+ optional allow/deny subtree patterns). Absent -> the paf-owned
    # data/vault/ playground default (see _spec_vault_config / _build_vault).
    vault_config = _spec_vault_config(spec_path)
    # Per-assistant skill library: the spec's top-level `skills:` block names
    # one or more full-path skill directories (+ optional top_k). Absent -> the
    # paf-owned data/skills/ default (see _spec_skills_config / _build_skills).
    skills_config = _spec_skills_config(spec_path)
    # Per-assistant memory config: the spec's top-level `memory:` block tunes
    # the Memory capability (e.g. `consolidate_every` — the background
    # consolidation cadence). Absent -> the Memory capability's code-level
    # default (see _spec_memory_config / _build_memory). All memory variables
    # live on Memory; anything the background queue needs is handed down as an
    # injected construction arg.
    memory_config = _spec_memory_config(spec_path)
    # Per-assistant root scoping: the spec's top-level `filesystem:`/`shell:`/
    # `git:`/`codebase_map:` blocks point each root-bearing capability at a
    # working directory other than PROJECT_ROOT (see _spec_rooted_config).
    # Absent block -> the builder's PROJECT_ROOT default. Lets a single agent
    # instance be pointed "at" another repo without touching a global default.
    filesystem_config = _spec_filesystem_config(spec_path)
    shell_config = _spec_shell_config(spec_path)
    git_config = _spec_git_config(spec_path)
    codebase_map_config = _spec_codebase_map_config(spec_path)
    # The spec is the source of truth for `model_settings`: if the spec YAML
    # declares its own `model_settings` (spec.model_settings), it is passed
    # through untouched and the caller's/injected pin is never applied over it.
    # We only inject the OpenRouter determinism pin (temp 0, top_p 1.0, seed,
    # pinned provider chain) when neither the spec nor the caller supplies a
    # `model_settings` override. Agent.from_spec still shallow-merges a caller
    # override over the spec's YAML settings (caller wins), but the *default*
    # pin no longer clobbers a spec-declared `model_settings`.
    if spec.model_settings is None and "model_settings" not in overrides:
        overrides.setdefault("model_settings", _openrouter_model_settings())
    # Bounded-timeout OpenRouter client (stuck-turn fix): when the spec's model
    # is an `openrouter:` string and the caller didn't supply their own model,
    # swap in an OpenRouterModel that wraps a client with a ~120s read timeout
    # and ``max_retries=0``. This is the only way to bound the SDK's
    # same-provider blind retries (``max_retries`` is not a ModelSettings
    # field), turning a 10+ min pinned-provider freeze into a clean
    # ModelAPIError after ~120s. The routing pin keeps living in
    # ``model_settings`` untouched. Falls back to the plain string (no change)
    # when ``OPENROUTER_API_KEY`` is absent or a custom ``model`` was passed.
    if (
        "model" not in overrides
        and isinstance(spec.model, str)
        and spec.model.startswith("openrouter:")
    ):
        custom_model = _openrouter_model_with_client(spec.model.split(":", 1)[1])
        if custom_model is not None:
            overrides["model"] = custom_model
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
        capabilities=build_runtime_capabilities(
            profile,
            compaction_config,
            web_config,
            browser_config,
            vault_config,
            memory_config=memory_config,
            filesystem_config=filesystem_config,
            shell_config=shell_config,
            git_config=git_config,
            codebase_map_config=codebase_map_config,
            skills_config=skills_config,
            agent_name=agent_name,
        ),
        **overrides,
    )
