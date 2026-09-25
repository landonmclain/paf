"""Spec loader — resolve agent names to YAML specs and build agents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic_ai import Agent, AgentSpec, UsageLimits

from paf.capabilities import ASSISTANT_PROFILE, AgentProfile
from paf.capabilities.build import build_runtime_capabilities
from paf.capabilities.compaction import CompactionConfig
from paf.config import SPECS_DIR
from paf.providers.openrouter import model_with_client

DEFAULT_SPEC_DIRECTORY = SPECS_DIR


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


def _spec_usage_limits(spec_path: Path) -> UsageLimits | None:
    """Read the spec's top-level ``usage_limits:`` block into a ``UsageLimits``.

    pydantic-ai's ``AgentSpec`` drops unknown top-level keys, so `build_agent`
    reads the raw YAML separately. The block shape:

        usage_limits:
          request_limit: 75

    Absent/unreadable block -> None, which makes the Session fall back to
    pydantic-ai's default ``UsageLimits()`` (request_limit=50). Only
    ``request_limit`` is read today; unknown keys are ignored so the schema can
    grow without breaking older spec files.
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    block = data.get("usage_limits")
    if not isinstance(block, dict):
        return None
    request_limit = block.get("request_limit")
    if not isinstance(request_limit, int):
        return None
    return UsageLimits(request_limit=request_limit)


def _spec_web_config(spec_path: Path) -> list[str] | None:
    """Read the spec's top-level ``web:`` block into an extract domain list.

    pydantic-ai's ``AgentSpec`` drops unknown top-level keys, so `build_agent`
    reads the raw YAML separately. The block shape:

        web:
          search_domains:
            - docs.pydantic.dev
            - github.com

    ``search_domains`` is the **extract** allowlist for ``fetch_web_page``
    (which domains may be pulled into context); open web *search* is never
    restricted by it. Absent/unreadable block or empty list -> None, which
    makes ``_build_web`` fall back to the ``PREFERRED_DOC_DOMAINS`` default.
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

    pydantic-ai's ``AgentSpec`` drops unknown top-level keys, so `build_agent`
    reads the raw YAML separately. The block shape:

        browser:
          allowed_domains:
            - localhost
            - staging.example.com
          block_private_addresses: false   # optional; default true

    ``allowed_domains`` is the Browser capability's egress allowlist — hosts
    (and their subdomains) the browser may reach. ``block_private_addresses``
    is the **mechanical** SSRF guard (refuses loopback/RFC-1918/link-local and
    cloud-metadata addresses whether or not an allowlist names them); it
    defaults to ``True``, and only an explicit ``false`` here loosens it so an
    allowlisted *private* host becomes reachable. The two are independent:
    listing a host in ``allowed_domains`` does **not** release the
    private-address guard on its own. ``cloud-metadata`` (``169.254.169.254``)
    is **never** released by this flag; see ``EgressPolicy``.

    Returns a dict (or ``None`` if the block is absent or unreadable); absent
    members fall back to their code-level defaults in `_build_browser`.
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

    pydantic-ai's ``AgentSpec`` drops unknown top-level keys, so `build_agent`
    reads the raw YAML separately. The block shape:

        vault:
          root: ~/Documents/notes   # `~` expands via os.path.expanduser
          allow:            # optional; absent = whole root
            - Notes/**
          deny:             # optional; always wins over allow
            - "**/.trash/**"

    ``root`` is the vault directory the capability confines itself to; ``allow``
    /``deny`` are optional glob subtree patterns expressing a workspace-vs-personal
    boundary inside that root. Returns a dict (or ``None`` when the block is
    absent/unreadable); absent members fall back to ``_build_vault`` defaults
    (whose default root is the paf-owned ``data/vault/`` playground — the real
    vault is ONLY reached via an explicit ``root`` here).
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

    pydantic-ai's ``AgentSpec`` drops unknown top-level keys, so `build_agent`
    reads the raw YAML separately. The block shape:

        skills:
          libraries:        # one or more full paths (~ expands); absent -> the
            - ~/.hermes/skills          # paf-owned data/skills/ default
            - ~/projects/Personal/skill-library
          top_k: 3           # optional; default number of point_skills hints

    ``libraries`` are **full absolute paths** (``~`` expanded via
    ``os.path.expanduser``); each is a directory scanned for ``SKILL.md`` skills
    (a skill tree: ``<root>/<category>/<skill>/SKILL.md`` or flat). ``top_k`` is
    the default number of ``point_skills`` hint pointers returned per call.
    Returns a dict (or ``None`` when the block is absent/unreadable); absent
    members fall back to ``_build_skills`` defaults (whose default library is
    the paf-owned ``data/skills/``).
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

    pydantic-ai's ``AgentSpec`` drops unknown top-level keys, so `build_agent`
    reads the raw YAML separately. The block shape:

        memory:
          consolidate_every: 10      # optional; background consolidation cadence
          max_memory_size: 65536     # optional; per-read char cap for MEMORY.md
          tier_caps:                 # optional; per-tier COUNT caps (render-owned)
            t1: 75
            t2: 25
            t3: 15

    ``consolidate_every`` is the background-consolidation cadence in **minutes**;
    ``max_memory_size`` bounds how much of the rendered journal is injected per
    read; ``tier_caps`` (a ``{t1/t2/t3: int|None}`` dict) are PER-TIER COUNT caps
    on the injected index render, resolved by ``resolve_memory_tier_caps``.

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
    cadence without poking pydantic-ai's private capability registry. It reads
    the same ``_spec_memory_config`` path ``_build_memory`` uses and applies the
    same code-level default, so the Session's trigger cadence can never drift
    from the ``Memory`` capability's ``consolidate_every``.

    Returns ``{"consolidate_every": int, "review_every": int, "consolidate_turns":
    int, "review_turns": int}`` (always present, defaults applied). The minute
    cadences remain for explicitness/back-compat; the TURN-count values are the
    active gate. ``review_turns`` defaults to ``DEFAULT_REVIEW_TURNS``.
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
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and candidate > 0
        ):
            every = candidate
        review_candidate = cfg.get("review_every")
        if (
            isinstance(review_candidate, int)
            and not isinstance(review_candidate, bool)
            and review_candidate > 0
        ):
            review_every = review_candidate
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
    (``{t1/t2/t3: int | None}``). It is not a single int like the cadence
    values, so it needs its own reader rather than riding inside
    :func:`resolve_memory_config`; the render (``_render.py``) calls this
    directly.

    Returns a **partial** dict of only the per-tier caps the spec declares
    (``None`` meaning unbounded for that tier), or ``None`` when the spec is
    absent/unreadable or declares no usable ``tier_caps`` block. The caller
    merges this over its built-in defaults (``_TIER_CAPS``), so unspecified
    tiers keep the code default.
    """
    try:
        with get_spec_path(agent_name, spec_directory).open(
            "r", encoding="utf-8"
        ) as fh:
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


def _spec_scalar(spec_path: Path, key: str) -> int | str | None:
    """Read one top-level scalar (e.g. ``max_item_chars``) from a spec.

    pydantic-ai's ``AgentSpec`` drops unknown top-level keys, so background-agent
    builders read paf-specific scalars (``retries``, ``max_item_chars``) from the
    raw YAML via the same best-effort pattern ``_spec_memory_config`` uses.
    Returns the raw value, or ``None`` when absent/unreadable.
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    return data.get(key)


def resolve_background_spec(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
) -> tuple[AgentSpec, dict[str, Any]]:
    """Resolve a background-agent leaf spec into ``(agent_spec, raw_dict)``.

    Background agents (``memory_extractor`` / ``memory_reviewer`` / …) are
    leaf-agent specs under ``specs/background_agents/``, resolved by bare name
    via the recursive ``get_spec_path``. The returned ``AgentSpec`` carries the
    pydantic-ai fields (``model`` / ``model_settings`` / ``instructions``); the
    raw dict additionally carries paf-specific top-level knobs that ``AgentSpec``
    would drop (``retries``, ``max_item_chars``). Builders read both.

    Raises :class:`AgentSpecNotFoundError` when no spec resolves.
    """
    spec_path = get_spec_path(agent_name, spec_directory)
    with spec_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raw = {}
    return load_agent_spec(agent_name, spec_directory), raw


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

    pydantic-ai's ``AgentSpec`` drops unknown top-level keys, so the value is
    read off disk directly. Returns ``None`` when the spec is absent,
    unreadable, or declares no ``profile:`` key — callers fall back to their
    own default.
    """
    try:
        with get_spec_path(agent_name, spec_directory).open(
            "r", encoding="utf-8"
        ) as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("profile")
    return value if isinstance(value, str) and value else None


def _spec_mcp_config(spec_path: Path) -> str | None:
    """Read the spec's top-level ``mcp:`` block into an MCP config path.

    The block just declares the path to a Claude-Desktop-style ``mcpServers``
    JSON; the servers + creds live there (env-expanded by the builtin), not in
    the spec. Shape:

        mcp:
          config: data/mcp_servers.json

    ``config`` is the path to the mcpServers JSON; ``_build_mcp`` resolves it
    against ``PROJECT_ROOT`` when relative and ``os.path.expanduser``-expands
    ``~``. Absent/unreadable block or missing ``config`` -> None (the
    profile-gated MCP capability stays absent).
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    block = data.get("mcp")
    if not isinstance(block, dict):
        return None
    config = block.get("config")
    if not isinstance(config, str) or not config.strip():
        return None
    return config


def _spec_rooted_config(spec_path: Path, block_name: str, root_key: str) -> dict | None:
    """Read one per-capability root block from a spec into a config dict.

    Shared reader for the root-bearing capabilities (filesystem / shell / git /
    codebase_map). pydantic-ai's ``AgentSpec`` drops unknown top-level keys, so
    `build_agent` reads the raw YAML separately. The block shape (per
    capability):

        filesystem:
          root_dir: ~/projects/Personal/landonmclain-site
        shell:
          cwd: ~/projects/Personal/landonmclain-site

    ``root_key`` is the member inside the block that carries the path
    (``root_dir`` / ``cwd`` / ``root`` / ``scan_root``). Only that key is
    validated and copied — unknown keys *inside* the block are ignored so the
    schema can grow without breaking older spec files.

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


def _spec_executor_block(executor: dict) -> dict:
    """Extract the known keys from a spec ``executor:`` block (shared by shell/fs).

    Returns only the keys that are present and well-typed (``backend`` /
    ``mode`` / ``image`` / ``exclude``). Unknown keys are ignored so the schema
    can grow without breaking older spec files.
    """
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
    exclude = executor.get("exclude")
    if isinstance(exclude, list):
        cleaned = [e for e in exclude if isinstance(e, str) and e.strip()]
        if cleaned:
            resolved["exclude"] = cleaned
    return resolved


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
            exclude: [.env, .env.*, .git-credentials, .git]

    ``backend`` defaults to ``local`` when the executor block is absent or
    names no backend.
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
        resolved = _spec_executor_block(executor)
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

    Tries ``{name}.yaml``, then ``{name}.yml``, then ``{name}.json`` in order
    (a root spec and a subfolder spec of the SAME suffix still resolve by that
    suffix precedence). Within each suffix it looks first at the spec root (the
    historical flat contract), then recursively beneath it, so leaf-agent specs
    can live in subfolders (``specs/sub_agents/writer_subagent.yaml``,
    ``specs/background_agents/memory_extractor.yaml``) and still resolve by bare
    name — no per-kind loader needed.

    Two specs of the SAME suffix in DIFFERENT directories (e.g. a root spec plus
    a same-named subfolder spec, or two subfolders sharing a name) is a genuine
    collision: we raise rather than silently pick, preserving determinism.
    Distinct suffixes in the same directory are NOT a collision — they follow
    ``.yaml`` > ``.yml`` > ``.json`` precedence (historical contract).
    """
    for suffix in _SPEC_SUFFIXES:
        matches = list(spec_directory.rglob(f"{agent_name}{suffix}"))
        if not matches:
            continue
        if len(matches) > 1:
            listed = ", ".join(str(p) for p in sorted(matches))
            raise AgentSpecNotFoundError(
                f"Ambiguous agent specification {agent_name!r}: {len(matches)} "
                f"'{suffix}' matches ({listed}); rename/dedupe so the name "
                "resolves once"
            )
        return matches[0]
    raise AgentSpecNotFoundError(
        f"No agent specification found for {agent_name!r} in {spec_directory}"
    )


_SPEC_SUFFIXES = (".yaml", ".yml", ".json")


def load_agent_spec(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
) -> AgentSpec:
    """Load and validate an agent specification from disk."""
    path = get_spec_path(agent_name, spec_directory)
    return AgentSpec.from_file(path)


def spec_usage_limits(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
) -> UsageLimits | None:
    """Read a named agent's spec ``usage_limits:`` block.

    Returns ``None`` when the spec has no ``usage_limits:`` block, so the
    Session falls back to pydantic-ai's default ``UsageLimits()``.
    """
    return _spec_usage_limits(get_spec_path(agent_name, spec_directory))


def build_agent(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
    profile: AgentProfile | None = None,
    on_complete=None,
    **overrides: Any,
) -> Agent[Any, Any]:
    """Construct an agent from a named spec, with optional overrides.

    ```agent_name`` is the spec file base name (``assistant`` → ``assistant.yaml``).

    ``profile`` is the capability allowlist for this agent kind — resolved to
    ``custom_capability_types`` so the spec's ``capabilities:`` entries can only
    come from the allowlisted catalog. Defaults to ``ASSISTANT_PROFILE``.

    ``build_runtime_capabilities()`` assembles capabilities that cannot resolve
    from YAML (the tiered ``TieredCompaction``, whose nested strategy objects have
    no ``from_spec``) and passes them via ``Agent.from_spec(..., capabilities=)``,
    which merges them with any spec-declared capabilities (e.g. ``StepPersistence``).

    Overrides are passed as kwargs to ``Agent.from_spec``. Supported:
    ``model``, ``instructions``, ``model_settings``, ``name``, ``output_type``.
    Tools and spec-declarable capabilities must be declared in the spec YAML.
    ``model_settings`` is the spec's to declare; when a spec omits it, the
    model runs on pydantic-ai's plain defaults.
    """
    profile = profile or ASSISTANT_PROFILE
    spec = load_agent_spec(agent_name, spec_directory)
    spec_path = get_spec_path(agent_name, spec_directory)
    # Per-agent compaction tuning from the spec's `compact:` block.
    compaction_config = _spec_compaction_config(spec_path)
    # Web extract allowlist from the spec's `web:` block.
    web_config = _spec_web_config(spec_path)
    # Browser egress allowlist from the spec's `browser:` block.
    browser_config = _spec_browser_config(spec_path)
    # Vault root + allow/deny from the spec's `vault:` block.
    vault_config = _spec_vault_config(spec_path)
    # Skill library dirs + top_k from the spec's `skills:` block.
    skills_config = _spec_skills_config(spec_path)
    # Memory tuning (e.g. consolidate_every) from the spec's `memory:` block.
    memory_config = _spec_memory_config(spec_path)
    # Root scoping for the root-bearing capabilities from their spec blocks.
    filesystem_config = _spec_filesystem_config(spec_path)
    shell_config = _spec_shell_config(spec_path)
    git_config = _spec_git_config(spec_path)
    codebase_map_config = _spec_codebase_map_config(spec_path)
    # mcpServers JSON path from the spec's `mcp:` block (builtin env expansion).
    mcp_config = _spec_mcp_config(spec_path)
    # Bounded-timeout OpenRouter client for `openrouter:` models (see
    # providers/openrouter.py); falls back to the plain string when no key.
    if (
        "model" not in overrides
        and isinstance(spec.model, str)
        and spec.model.startswith("openrouter:")
    ):
        custom_model = model_with_client(spec.model.split(":", 1)[1])
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
            mcp_config=mcp_config,
            agent_name=agent_name,
            on_complete=on_complete,
        ),
        **overrides,
    )
