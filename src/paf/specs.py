"""Spec loader — resolve agent names to YAML specs and build agents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic_ai import Agent, AgentSpec, UsageLimits

from paf.capabilities import PROFILES, AgentProfile
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

    ``libraries`` are **full absolute paths** (``~`` expanded via
    ``os.path.expanduser``); each is a directory scanned for ``SKILL.md`` skills
    (a skill tree: ``<root>/<category>/<skill>/SKILL.md`` or flat).
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
          pairwise_review: false     # optional; pairwise reviewer after each fold (default on)
          render:                    # optional; scoped render + its token knobs
            mode: scoped             #   scoped | tiers (default tiers)
            global_tokens: 2000      #   any RenderBudget field overrides its default
            project_tokens: 2500
            history_tokens: 1000
            history_days: 30

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
        "max_memory_size",
    ):
        value = block.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            cfg[key] = value
    if isinstance(block.get("pairwise_review"), bool):
        cfg["pairwise_review"] = block["pairwise_review"]
    if isinstance(block.get("render"), dict):
        cfg["render"] = dict(block["render"])
    if not cfg:
        return None
    return cfg


def _spec_keywords_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``keywords:`` block (relevance-hint knobs).

        keywords:
          inject: true       # false = build the capability but inject nothing
          top_k: 3           # at most this many hints per turn
          min_score: 2.5     # absolute floor on a hint's score
          query_turns: 2     # how many recent user prompts the match reads
          curate: false      # true = curate in the background (sends digests to the
                             # model provider; OFF unless you opt in)
          curate_limit: 25   # sources curated per pass (a backlog takes several)
          curate_cooldown_s: 300   # minimum seconds between passes

    Absent members fall back to the capability's defaults; unknown keys are
    ignored so the schema can grow. Returns ``None`` when the block is absent.
    """
    block = _spec_mapping_block(spec_path, "keywords")
    if not block:
        return None
    cfg: dict[str, object] = {}
    if isinstance(block.get("inject"), bool):
        cfg["inject"] = block["inject"]
    for key in ("top_k", "query_turns"):
        value = block.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            cfg[key] = value
    if isinstance(block.get("curate"), bool):
        cfg["curate"] = block["curate"]
    limit = block.get("curate_limit")
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        cfg["curate_limit"] = limit
    cooldown = block.get("curate_cooldown_s")
    if (
        isinstance(cooldown, int | float)
        and not isinstance(cooldown, bool)
        and cooldown >= 0
    ):
        cfg["curate_cooldown_s"] = float(cooldown)
    score = block.get("min_score")
    if isinstance(score, int | float) and not isinstance(score, bool) and score >= 0:
        cfg["min_score"] = float(score)
    return cfg or None


def _spec_search_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``search:`` block into a config dict.

    pydantic-ai's ``AgentSpec`` drops unknown top-level keys, so `build_agent`
    reads the raw YAML separately. Absent members fall back to their defaults
    in ``_build_conversation_search`` (max_matches=10, context_lines=5,
    bm25_k1=1.5, bm25_b=0.75, scope="conversation"), so a block is purely
    additive and never silently weakens a secure default. Unknown keys inside
    ``search:`` are ignored so the schema can grow without breaking older spec
    files.
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    block = data.get("search")
    if not isinstance(block, dict):
        return None
    cfg: dict[str, object] = {}
    for key in ("max_matches", "context_lines"):
        value = block.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            cfg[key] = value
    for key in ("bm25_k1", "bm25_b"):
        value = block.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            cfg[key] = value
    scope = block.get("scope")
    if isinstance(scope, str) and scope in ("all", "conversation"):
        cfg["scope"] = scope
    if not cfg:
        return None
    return cfg


def _spec_top_level_str(spec_path: Path, key: str) -> str | None:
    """Read a top-level ``<key>:`` string block from a spec's raw YAML.

    pydantic-ai's ``AgentSpec`` drops unknown top-level keys, so harness-prompt
    blocks are read from the raw YAML separately, the same way the ``search:``
    block is. Returns ``None`` when absent or not a non-empty string.
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    block = data.get(key)
    if isinstance(block, str) and block.strip():
        return block
    return None


def _spec_search_only_instructions(spec_path: Path) -> str | None:
    """Read the spec's top-level ``search_only_instructions:`` block.

    The search-only (no-memory, ``MEMORY=0``) arm's prompt is read from the raw
    YAML separately. Returns ``None`` when absent so a ``MEMORY=0`` run can
    hard-error rather than silently reuse the memory prompt.
    """
    return _spec_top_level_str(spec_path, "search_only_instructions")


def _spec_search_only_no_readturn_instructions(spec_path: Path) -> str | None:
    """Read the spec's top-level ``search_only_no_readturn_instructions:`` block.

    The pure-BM25-truncated floor arm (``NO_READ_TURN=1``): the search tool is
    exposed without ``read_turn``, so the subject must answer from truncated
    excerpts. Returns ``None`` when absent so a no-read-turn run can hard-error
    rather than silently reuse the read_turn prompt.
    """
    return _spec_top_level_str(spec_path, "search_only_no_readturn_instructions")


def _spec_thread_summary_instructions(spec_path: Path) -> str | None:
    """Read the spec's top-level ``thread_summary_instructions:`` block.

    The thread-summarizer arm's prompt section (describing the injected
    ``<thread-summary>`` block + the ``list_turn_summaries`` navigation tool),
    appended to the base instructions when ``THREAD_SUMMARY=1``. Returns ``None``
    when absent so a ``THREAD_SUMMARY=1`` run can hard-error rather than run
    without telling the subject the tool exists.
    """
    return _spec_top_level_str(spec_path, "thread_summary_instructions")


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

    Returns ``{"consolidate_every": int, "consolidate_turns": int,
    "pairwise_review": bool}`` (always present, defaults applied; the pairwise
    reviewer runs after each fold unless the spec sets
    ``memory.pairwise_review: false``). The minute cadence remains for
    explicitness/back-compat; the TURN count is the active gate.
    """
    from paf.capabilities.build import (
        DEFAULT_CONSOLIDATE_EVERY,
        DEFAULT_CONSOLIDATE_TURNS,
    )

    every = DEFAULT_CONSOLIDATE_EVERY
    turns = DEFAULT_CONSOLIDATE_TURNS
    pairwise_review = True
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
        if isinstance(cfg.get("pairwise_review"), bool):
            pairwise_review = cfg["pairwise_review"]
    return {
        "consolidate_every": every,
        "consolidate_turns": turns,
        "pairwise_review": pairwise_review,
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

    Background agents (``memory_extractor`` / ``memory_pair_reviewer`` / …) are
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
    for key in ("network", "user", "memory", "cpus", "workspace"):
        value = executor.get(key)
        if isinstance(value, str) and value.strip():
            resolved[key] = value
    pids_limit = executor.get("pids_limit")
    if isinstance(pids_limit, int) and not isinstance(pids_limit, bool):
        resolved["pids_limit"] = pids_limit
    for key in ("read_only", "workspace_readonly"):
        if isinstance(executor.get(key), bool):
            resolved[key] = executor[key]
    for key in ("writable", "tmpfs"):
        values = executor.get(key)
        if isinstance(values, list):
            cleaned = [v for v in values if isinstance(v, str) and v.strip()]
            if cleaned:
                resolved[key] = cleaned
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


def _spec_python_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``python:`` block (the ``run_python`` tool).

        python:
          cwd: ~/workspace
          allow_unsandboxed: false     # default; true permits the host backend
          default_timeout: 30
          max_timeout: 120
          executor:                    # overrides the strict Docker defaults
            writable: [out]            # workspace subpaths writable by programs
            memory: 512m

    Returns only well-typed keys, or ``None`` when the block is absent.
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    block = data.get("python")
    if not isinstance(block, dict):
        return None
    config: dict = {}
    cwd = block.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        config["cwd"] = cwd
    if isinstance(block.get("allow_unsandboxed"), bool):
        config["allow_unsandboxed"] = block["allow_unsandboxed"]
    for key in ("default_timeout", "max_timeout"):
        value = block.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            config[key] = value
    executor = block.get("executor")
    if isinstance(executor, dict):
        resolved = _spec_executor_block(executor)
        if resolved:
            config["executor"] = resolved
    return config or None


def _spec_gating_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``gating:`` block (declarative tool policy).

        gating:
          default: block          # allow (default) | block | approve
          rules:                  # first match wins
            - tools: [read_file, list_directory]
              action: allow
            - tools: [write_file, edit_file]
              action: allow
              inside: [out]       # only when the path is under out/
            - tools: [write_file, edit_file]
              action: approve     # everything else; enforced as block until
                                  # an approval transport exists

    Returned as written (a raw mapping); ``guardrails.gating_policy.parse_policy``
    validates it and raises on a malformed block, so a half-applied security
    policy cannot load. ``None`` when the block is absent or unreadable.
    """
    return _spec_mapping_block(spec_path, "gating")


def _spec_secrets_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``secrets:`` block.

        secrets:
          allowed: []             # env var names a $SECRET:name handle may resolve

    ``allowed`` is kept only when it is a list of strings; absent -> the guard
    stays unrestricted (legacy), so an explicit ``[]`` (resolve nothing) and a
    missing key are different on purpose.
    """
    block = _spec_mapping_block(spec_path, "secrets")
    if not block:
        return None
    allowed = block.get("allowed")
    if isinstance(allowed, list) and all(isinstance(n, str) for n in allowed):
        return {"allowed": allowed}
    return None


def _spec_mapping_block(spec_path: Path, key: str) -> dict | None:
    """The spec's top-level ``key:`` mapping, or ``None`` (absent/unreadable)."""
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    block = data.get(key)
    return block if isinstance(block, dict) else None


def _spec_git_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``git:`` block (root)."""
    return _spec_rooted_config(spec_path, "git", "root")


def _spec_codebase_map_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``codebase_map:`` block (scan_root)."""
    return _spec_rooted_config(spec_path, "codebase_map", "scan_root")


def _spec_workspace_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``workspace:`` block (extra always-present lines).

        workspace:
          expose_paths: false   # default true; false hides host paths (home,
                                # tool roots) from the model — locked-down agents
          notes:
            - Projects live under ~/projects/Personal.

    Returns ``{"notes": [...]?, "expose_paths": False?}`` with only the keys that
    are set, or ``None`` when absent / empty.
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    block = data.get("workspace") if isinstance(data, dict) else None
    if not isinstance(block, dict):
        return None
    config: dict = {}
    notes = [n for n in block.get("notes") or [] if isinstance(n, str) and n.strip()]
    if notes:
        config["notes"] = notes
    if block.get("expose_paths") is False:
        config["expose_paths"] = False
    return config or None


def _spec_subagents_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``subagents:`` block (roster + concurrency cap).

    Shape::

        subagents:
          agents: [writer_subagent, researcher_subagent]   # leaf spec names
          max_concurrent: 5                                # background cap / conversation

    Returns ``{"agents": [...]?, "max_concurrent": int?}`` with only the valid
    keys present, or ``None`` when the block is absent / unreadable. Absent ->
    the builder's default roster.
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    block = raw.get("subagents") if isinstance(raw, dict) else None
    if not isinstance(block, dict):
        return None
    config: dict[str, Any] = {}
    agents = block.get("agents")
    if isinstance(agents, list) and all(isinstance(a, str) and a for a in agents):
        config["agents"] = list(agents)
    max_concurrent = block.get("max_concurrent")
    if isinstance(max_concurrent, int) and max_concurrent > 0:
        config["max_concurrent"] = max_concurrent
    return config or None


def _spec_visual_config(spec_path: Path) -> dict | None:
    """Read the spec's top-level ``visual:`` block (vision model slug).

    Shape::

        visual:
          model: openrouter:xiaomi/mimo-v2.5
          model_settings:              # optional, passed to the vision calls
            openai_reasoning_effort: none

    Returns ``{"model": <str>}`` (plus ``"model_settings"`` when the block has
    them) when the block names a model, ``{"model": None}``
    when a `visual:` block is present but omits `model` (a misconfiguration the
    builder hard-errors on), or ``None`` when the block is absent / unreadable.
    Absent -> the Visual capability is inert: no separate vision model, images
    pass through to the main model untouched.
    """
    try:
        with spec_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    block = raw.get("visual") if isinstance(raw, dict) else None
    if not isinstance(block, dict):
        return None
    model = block.get("model")
    if isinstance(model, str) and model.strip():
        settings = block.get("model_settings")
        if isinstance(settings, dict) and settings:
            return {"model": model, "model_settings": settings}
        return {"model": model}
    return {"model": None}


def _spec_runtime_configs(spec_path: Path) -> dict[str, Any]:
    """Every per-capability config block a spec can carry, as build kwargs.

    Keys match :func:`paf.capabilities.build.build_runtime_capabilities`'s
    keyword parameters, so the subject agent (``build_agent``) and tooled
    subagent leaves read their capability config the same way.
    """
    return {
        # Per-agent compaction tuning from the spec's `compact:` block.
        "compaction_config": _spec_compaction_config(spec_path),
        # Web extract allowlist from the spec's `web:` block.
        "web_search_domains": _spec_web_config(spec_path),
        # Browser egress allowlist from the spec's `browser:` block.
        "browser_config": _spec_browser_config(spec_path),
        # Vault root + allow/deny from the spec's `vault:` block.
        "vault_config": _spec_vault_config(spec_path),
        # Memory tuning (e.g. consolidate_every) from the spec's `memory:` block.
        "memory_config": _spec_memory_config(spec_path),
        # Conversation-search knobs (max_matches/context_lines/bm25/scope) from
        # the spec's `search:` block.
        "search_config": _spec_search_config(spec_path),
        # Root scoping for the root-bearing capabilities from their spec blocks.
        "filesystem_config": _spec_filesystem_config(spec_path),
        "shell_config": _spec_shell_config(spec_path),
        "python_config": _spec_python_config(spec_path),
        "gating_config": _spec_gating_config(spec_path),
        "secrets_config": _spec_secrets_config(spec_path),
        "git_config": _spec_git_config(spec_path),
        "codebase_map_config": _spec_codebase_map_config(spec_path),
        # Skill library dirs from the spec's `skills:` block.
        "skills_config": _spec_skills_config(spec_path),
        # Relevance-hint knobs from the spec's `keywords:` block.
        "keywords_config": _spec_keywords_config(spec_path),
        # mcpServers JSON path from the spec's `mcp:` block (builtin env expansion).
        "mcp_config": _spec_mcp_config(spec_path),
        # Subagent roster + background cap from the spec's `subagents:` block.
        "subagents_config": _spec_subagents_config(spec_path),
        # Vision model slug from the spec's `visual:` block.
        "visual_config": _spec_visual_config(spec_path),
        # Extra <workspace> lines from the spec's `workspace:` block.
        "workspace_config": _spec_workspace_config(spec_path),
    }


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


def spec_visual_config(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
) -> dict[str, Any]:
    """The agent spec's vision config: ``{"model": str | None, "model_settings"?}``.

    ``model`` is None when the spec is missing or has no ``visual:`` model.
    """
    try:
        config = _spec_visual_config(get_spec_path(agent_name, spec_directory))
    except AgentSpecNotFoundError:
        return {"model": None}
    return config or {"model": None}


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


def _declared_profile(agent_name: str, spec_directory: Path) -> AgentProfile:
    """The profile the spec declares via ``profile:``; raises if it declares none.

    A profile is an agent's capability allowlist, so there is no safe default to
    fall back to: guessing one (the assistant's, say) hands a locked-down spec
    the full tool set, or a minimal one capabilities it never asked for. The
    caller must pass ``profile=`` or the spec must declare ``profile:``. A
    declared name that is not a registered profile also raises, so a typo can
    never widen an agent's capabilities.
    """
    declared = resolve_profile_name(agent_name, spec_directory)
    if declared is None:
        raise ValueError(
            f"No profile for agent {agent_name!r}: pass `profile=` to build_agent "
            f"or declare `profile:` in the spec. Known: {sorted(PROFILES)}"
        )
    if declared not in PROFILES:
        raise ValueError(
            f"Spec {agent_name!r} declares profile {declared!r}, which is not "
            f"registered. Known: {sorted(PROFILES)}"
        )
    return PROFILES[declared]


def build_agent(
    agent_name: str,
    spec_directory: Path = DEFAULT_SPEC_DIRECTORY,
    profile: AgentProfile | None = None,
    on_complete=None,
    scheduled_tasks_queue=None,
    **overrides: Any,
) -> Agent[Any, Any]:
    """Construct an agent from a named spec, with optional overrides.

    ```agent_name`` is the spec file base name (``assistant`` → ``assistant.yaml``).

    ``profile`` is the capability allowlist for this agent kind — resolved to
    ``custom_capability_types`` so the spec's ``capabilities:`` entries can only
    come from the allowlisted catalog. When omitted it is the profile the spec declares (``profile:``); with neither, ``build_agent`` raises.

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
    profile = profile or _declared_profile(agent_name, spec_directory)
    spec = load_agent_spec(agent_name, spec_directory)
    spec_path = get_spec_path(agent_name, spec_directory)
    # Every per-capability config block (compact/web/browser/vault/skills/
    # memory/filesystem/shell/git/codebase_map/mcp/subagents) from the spec.
    runtime_configs = _spec_runtime_configs(spec_path)
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
            **runtime_configs,
            agent_name=agent_name,
            on_complete=on_complete,
            scheduled_tasks_queue=scheduled_tasks_queue,
        ),
        **overrides,
    )
