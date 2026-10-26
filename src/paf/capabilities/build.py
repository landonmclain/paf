"""Runtime capability builders — the ``capabilities=`` seam for Agent.from_spec.

Spec files (`specs/*.yaml`) go through pydantic-ai's ``AgentSpec``/``Agent.from_spec``,
which resolves each top-level ``capabilities:`` entry via that class's ``from_spec()``
and — crucially — only ONE level deep. Capabilities that can't be expressed that way —
because a class doesn't override ``from_spec`` or needs real objects as arguments — are
assembled here and injected via the ``capabilities=`` kwarg, which ``Agent.from_spec``
MERGES with any spec-declared capabilities.

**The profile is the gatekeeper.** ``build_runtime_capabilities(profile)`` builds exactly
the capabilities the profile's ``capability_names`` allowlists, over one shared store.
This Python ``capabilities=`` seam is the ONLY thing that reaches the agent, so the
profile's allowlist is what decides "what this agent gets."

Persistent capabilities (StepPersistence, ConversationSearch) select their builders from
:data:`paf.capabilities.persistence.PERSISTENCE_BUILDERS`. Context compaction builds via
``build_compaction()`` **iff the profile allowlists ``Compaction``** — the ladder WITHOUT
the summarizing tier (the LLM rewrite is mutually exclusive with the recall-ladder
injection; see ``compaction.py``). A profile simply omits a name from its
``capability_names`` and that agent won't get it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from pydantic_ai.capabilities import MCP
from pydantic_ai.mcp import load_mcp_toolsets
from pydantic_ai_harness import ToolGuardrail
from pydantic_ai_harness.tool_output_limits import (
    Band,
    LocalFileStore,
    Spill,
    ToolOutputLimits,
    Truncate,
)
from pydantic_ai_harness.warn_on_cache_busts import WarnOnCacheBusts

from paf.approvals import ApprovalBroker, approvals_available
from paf.capabilities._sandbox import (
    DockerExecutor,
    LocalProcessExecutor,
    SandboxedExecutor,
    docker_available,
    parse_size,
)
from paf.capabilities._secret_files import fs_denied_patterns, sandbox_excludes
from paf.capabilities.browser import Browser as _Browser
from paf.capabilities.codebase_map import CodebaseMap as _CodebaseMap
from paf.capabilities.compaction import build_compaction
from paf.capabilities.fs import FileSystem as _FileSystem
from paf.capabilities.git import Git as _Git
from paf.capabilities.guardrails.gating_policy import make_gating_guard, parse_policy
from paf.capabilities.guardrails.secret_handles import (
    make_secret_handle_guard,
    make_secret_result_guard,
    secret_handle_guard,
)
from paf.capabilities.guardrails.tool_gating import tool_gating_guard
from paf.capabilities.injection import InjectionGuard as _InjectionGuard
from paf.capabilities.injection import parse_injection_policy
from paf.capabilities.keyword_curator import KeywordCurator
from paf.capabilities.memory import Memory
from paf.capabilities.persistence import PERSISTENCE_BUILDERS, _build_store
from paf.capabilities.planning import Planning
from paf.capabilities.profiles import AgentProfile
from paf.capabilities.push import Push
from paf.capabilities.python import Python as _Python
from paf.capabilities.scheduled_tasks import ScheduledTasks
from paf.capabilities.shell import Shell as _Shell
from paf.capabilities.shell._command_policy import parse_command_policy
from paf.capabilities.skills import Skills as _Skills
from paf.capabilities.subagents import Subagents
from paf.capabilities.thread_summarizer import ThreadSummarizer
from paf.capabilities.vault import Vault as _Vault
from paf.capabilities.visual import Visual as _Visual
from paf.capabilities.web import Web as _Web
from paf.config import DATA_DIR, PLANS_DIR, PROCESSES_DIR, PROJECT_ROOT, settings

# The profile name that gates context compaction in build_runtime_capabilities.
_COMPACTION_CAPABILITY = "Compaction"

# Leaf specs (bare names under specs/sub_agents/) the Subagents capability
# builds. The runtime delegate_task key comes from each spec's `subagent.name`.
_SUBAGENT_ROSTER: tuple[str, ...] = ("writer_subagent",)

# Where the Memory capability writes its readable `.md` bodies + `.memory-state.json`
# CAS sidecar, anchored to the repo root (same convention as persistence/_STORE_DIR).
# Directory comes from the shared Settings object (MEMORY_DIR / .env), so the
# consolidation owner and the assistant always target the same store.
_MEMORY_DIR = PROJECT_ROOT / Path(settings.memory_dir)

# Default cadence for background memory consolidation, in MINUTES. The owner
# runs on a **time** cadence (the watermark is time-anchored on the persisted
# `started_at` — monotonic across runs, unlike a per-run turn ordinal). Memory
# owns this value; the spec's `memory: {consolidate_every: N}` (minutes)
# overrides it per agent.
DEFAULT_CONSOLIDATE_EVERY = (
    30  # minutes; owner fires at least this often with new material
)

# Turn-count cadence (supersedes the minute-based default). Memory
# consolidation is cheap extract+append (5 turns), with the pairwise reviewer
# run inside the same job; the thread summarizer keeps its narrative block (10
# turns, thread_summarizer _background.py). The default when a spec does not
# override.
DEFAULT_CONSOLIDATE_TURNS = 5

# Default bounds for the assistant's Memory capability. `max_memory_size`
# remains the per-read char cap JournalViewStore truncates MEMORY.md to (a
# hard safety net, independent of tier counts). Per-tier COUNT caps
# (`memory: tier_caps`) are owned by the render (_render.py), which resolves
# them from the agent's spec itself — see `_render._resolve_tier_caps`.
# Memory owns every memory config variable; the spec overrides per agent.
DEFAULT_MEMORY_MAX_SIZE = 65_536

# Where the Planning capability writes its per-conversation plan files.
_PLAN_ROOT = PLANS_DIR


def _read_positive_int(memory_config: dict, key: str, default: int) -> int:
    """Read a positive-int knob from the spec's ``memory:`` block."""
    value = memory_config.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


def _build_memory(
    store, memory_config: dict | None = None, agent_name: str | None = None
) -> Memory:
    """Journal-view durable memory, INJECT-ONLY for the assistant.

    The model carries the rendered journal (MEMORY.md) injected into every
    request via before_model_request, and is exposed ZERO memory tools
    (expose_tools=False) -- not even read/search -- because a separate
    consolidation owner sub-agent is the sole writer.

    `store` is accepted for a uniform builder signature but unused.
    `memory_config` is the spec's top-level ``memory:`` block (see
    ``specs._spec_memory_config``): ``consolidate_every`` (background-
    consolidation cadence) and ``max_memory_size`` (a hard, non-evicting
    per-read char cap on the injected journal) are read here and overridable
    per agent. Per-tier COUNT caps (``memory: tier_caps``) are owned by the
    render (`_render._resolve_tier_caps`), which reads them from the agent's
    spec directly.
    """
    from paf.capabilities.memory._overrides import OverrideStore
    from paf.capabilities.memory._view import JournalViewStore
    from paf.capabilities.memory._window import memory_dir_for

    cfg = memory_config if isinstance(memory_config, dict) else {}

    max_memory_size = _read_positive_int(
        cfg, "max_memory_size", DEFAULT_MEMORY_MAX_SIZE
    )

    memory_dir = str(memory_dir_for(agent_name)) if agent_name else Path(_MEMORY_DIR)

    # `memory: render: {mode: scoped, ...}` injects the per-turn scoped render
    # (global / this project / mentioned / recent history) instead of the static
    # tier-capped MEMORY.md; the other `render` keys override RenderBudget.
    render = cfg.get("render")
    render_cfg = render if isinstance(render, dict) else {}
    render_budget = None
    if render_cfg.get("mode") == "scoped" and agent_name:
        from paf.capabilities.memory._scoped import RenderBudget

        render_budget = RenderBudget.from_config(render_cfg)

    return Memory(
        store=JournalViewStore(memory_dir),
        inject_memory=True,
        expose_tools=False,
        max_memory_size=max_memory_size,
        # None → `_DEFAULT_MEMORY_GUIDANCE` (the tiered, read-only notebook
        # guidance the assistant should always carry). A spec may opt out with
        # an explicit `guidance: ""`.
        guidance=None,
        # The assistant's narrow curation tool: propose archiving a stale fact.
        # It writes only to the override sidecar (never the journal/index).
        overrides=OverrideStore(memory_dir),
        render_budget=render_budget,
        memory_directory=str(memory_dir),
        render_agent=agent_name,
    )


def _build_keyword_curator(
    store, keywords_config: dict | None = None, agent_name: str | None = None
) -> KeywordCurator:
    """Build the KeywordCurator capability (the per-turn hints reader).

    ``store`` is accepted for a uniform builder signature but unused — the keyword
    store is its own flat-file directory (``data/keywords/<agent>/``), populated by
    the background curator. ``keywords_config`` is the spec's top-level
    ``keywords:`` block (``specs._spec_keywords_config``): ``top_k``, ``min_score``,
    ``inject``. With no store yet it injects nothing.
    """
    cfg = keywords_config or {}
    return KeywordCurator.from_spec(
        agent_name=agent_name,
        inject_hints=cfg.get("inject", True),
        **{k: cfg[k] for k in ("top_k", "min_score", "query_turns") if k in cfg},
    )


def _build_thread_summarizer(
    store, thread_config: dict | None = None, agent_name: str | None = None
) -> ThreadSummarizer:
    """Build the ThreadSummarizer capability.

    The capability is a pure on/off surface: it learns the active conversation
    from ``ctx.conversation_id`` and exposes the per-agent capture directory.
    Background thread-beats are driven by the runtime trigger + worker
    (cadence + per-conversation MemoryWatermark over the persisted step store),
    NOT by an eviction WAL — the capture-hook wiring is gone. ``store`` (shared
    persistence) and ``thread_config`` (a spec ``thread_summarizer:`` block)
    settle in later phases. ``agent_name`` scopes the per-agent summaries dir.
    Note: the window-hook attachment previously done here was removed with the
    capture path.
    """
    return ThreadSummarizer.from_spec(agent_name=agent_name, capture_directory=None)


def _build_filesystem(
    store, filesystem_config: dict | None = None, expose_root: bool = True
) -> _FileSystem:
    """Scoped filesystem access rooted at the project root (paf-owned copy).

    `store` is accepted for a uniform builder signature but unused — the
    filesystem is self-contained over its own root, not the persistence store.
    Rooting at PROJECT_ROOT lets the model work with repo files (read/write/
    edit/list/search/find) while the harness toolset's traversal prevention +
    protected-pattern defaults (`.git/`, `.env`, keys, secrets read-only) keep it
    confined and safe.

    `filesystem_config` is the spec's top-level ``filesystem:`` block (see
    ``specs._spec_filesystem_config``): ``root_dir`` overrides the default
    ``PROJECT_ROOT`` so an agent can be pointed at a specific working repo.
    Absent/``None`` -> ``PROJECT_ROOT``. Traversal/symlink/protected-pattern
    gates always apply, whichever root is in effect. ``denied_patterns`` adds
    paths the tools refuse to read OR write, list or search, and
    ``deny_secret_files: true`` adds the shared credential-file set
    (``_secret_files.py``, the same one the Docker sandbox blanks) at any depth.
    Both are opt-in per spec; the default policy is unchanged.
    """
    root_dir = PROJECT_ROOT
    if (
        isinstance(filesystem_config, dict)
        and isinstance(filesystem_config.get("root_dir"), str)
        and filesystem_config.get("root_dir")
    ):
        root_dir = Path(os.path.expanduser(filesystem_config["root_dir"]))
    denied: list[str] = []
    if isinstance(filesystem_config, dict):
        denied += [p for p in filesystem_config.get("denied_patterns") or [] if p]
        if filesystem_config.get("deny_secret_files") is True:
            denied += fs_denied_patterns()
    return _FileSystem(
        root_dir=root_dir, expose_root=expose_root, denied_patterns=denied
    )


# Credential-bearing files (matched at ANY depth; see _secret_files.py) that the
# Docker backend overlays with empty files so the sandbox never sees the real
# secret files. The host-side file tools deny the same set (`deny_secret_files`). ``~/.ssh`` is a
# host path outside the repo workspace and is not overlaid (the container cannot
# reach it anyway); it is listed for completeness when the workspace is the home
# dir.
DEFAULT_EXCLUDE: tuple[str, ...] = sandbox_excludes()


def _resolve_executor(executor_config: dict | None) -> SandboxedExecutor:
    """Resolve a spec ``executor`` block into a backend instance.

    Used by the shell capability. ``None`` or ``backend: local`` ->
    :class:`LocalProcessExecutor` (host, no sandbox). ``backend: docker`` ->
    :class:`DockerExecutor` with the block's options (``mode`` / ``image`` /
    ``network`` / ``user`` / ``read_only`` / ``memory`` / ``cpus`` /
    ``pids_limit`` / ``workspace`` / ``container_name`` / ``exclude`` /
    ``workspace_readonly`` / ``writable`` / ``tmpfs`` / ``max_writable_size``), each
    defaulting to the executor's secure default when omitted. ``exclude`` names
    workspace-relative paths (globs allowed) that the Docker backend overlays
    with empty files; it defaults to :data:`DEFAULT_EXCLUDE`.

    Docker is the security boundary, so a spec that *asks* for it but runs on a
    host without Docker hard-fails at build time rather than silently falling
    back to an unsandboxed local process (the model cannot weaken its own
    sandbox).
    """
    if not isinstance(executor_config, dict):
        return LocalProcessExecutor()
    backend = executor_config.get("backend", "local")
    if backend == "local":
        return LocalProcessExecutor()
    if backend == "docker":
        if not docker_available():
            raise RuntimeError(
                "executor.backend is 'docker' but Docker is not available "
                "on this host; refusing to fall back to an unsandboxed local "
                "process. Install/start Docker or set backend: local."
            )
        return DockerExecutor(
            mode=executor_config.get("mode", "exec"),
            image=executor_config.get("image", "python:3.12-slim"),
            network=executor_config.get("network", "none"),
            user=executor_config.get("user"),
            read_only=executor_config.get("read_only", True),
            memory=executor_config.get("memory", "1g"),
            cpus=executor_config.get("cpus", "1.0"),
            pids_limit=executor_config.get("pids_limit", 256),
            workspace=executor_config.get("workspace", "/workspace"),
            container_name=executor_config.get("container_name"),
            exclude=executor_config.get("exclude", DEFAULT_EXCLUDE),
            workspace_readonly=executor_config.get("workspace_readonly", False),
            writable=executor_config.get("writable", ()),
            tmpfs=executor_config.get("tmpfs", ()),
            max_writable_bytes=(
                parse_size(executor_config["max_writable_size"])
                if executor_config.get("max_writable_size") is not None
                else None
            ),
        )
    raise ValueError(f"executor.backend must be 'local' or 'docker', got {backend!r}")


def _resolve_shell_executor(executor_config: dict | None) -> SandboxedExecutor:
    """Resolve the spec's ``shell.executor`` block (see ``_resolve_executor``)."""
    return _resolve_executor(executor_config)


# Executor defaults for ``run_python``: the strictest the Docker backend offers.
# A fresh container per call (no state leaks between runs), a read-only workspace
# with only a size-capped, non-executable scratch tmpfs writable. The spec's
# ``python.executor`` block overrides key by key.
_PYTHON_EXECUTOR_DEFAULTS: dict = {
    "backend": "docker",
    "mode": "run",
    "workspace_readonly": True,
    "tmpfs": ["/tmp:rw,size=64m,noexec,nosuid"],
    # Ceiling on what programs may leave in the host-visible `writable` paths
    # (enforced while the program runs, then cleaned up). Inert until the spec
    # opens a `writable` path.
    "max_writable_size": "100m",
}


def _build_python(store, python_config: dict | None = None) -> _Python:
    """Sandboxed ``run_python`` rooted at the project root (paf-owned).

    Secure by default and stricter than ``Shell``: the executor defaults to
    Docker in per-call ``run`` mode with a read-only workspace
    (:data:`_PYTHON_EXECUTOR_DEFAULTS`), and the capability refuses the local
    backend unless the spec sets ``allow_unsandboxed: true``.

    `python_config` is the spec's top-level ``python:`` block (see
    ``specs._spec_python_config``): ``cwd`` moves the workspace, ``executor``
    overrides the defaults above, and ``allow_unsandboxed`` /
    ``default_timeout`` / ``max_timeout`` tune the tool. `store` is unused.
    """
    config = python_config if isinstance(python_config, dict) else {}
    cwd = PROJECT_ROOT
    if isinstance(config.get("cwd"), str) and config["cwd"]:
        cwd = Path(os.path.expanduser(config["cwd"]))
    executor_config = {**_PYTHON_EXECUTOR_DEFAULTS, **(config.get("executor") or {})}
    kwargs: dict = {}
    for key in ("default_timeout", "max_timeout"):
        if isinstance(config.get(key), int | float):
            kwargs[key] = float(config[key])
    return _Python(
        cwd=cwd,
        executor=_resolve_executor(executor_config),
        allow_unsandboxed=config.get("allow_unsandboxed") is True,
        **kwargs,
    )


def _build_shell(
    store,
    shell_config: dict | None = None,
    agent_name: str | None = None,
    on_complete=None,
) -> _Shell:
    """Shell command execution rooted at the project root (paf-owned copy).

    `store` is accepted for a uniform builder signature but unused — the shell
    is self-contained over its own cwd, not the persistence store. Rooting at
    PROJECT_ROOT lets the model run commands against the repo; the harness
    toolset's deny-list (rm/dd/shutdown/...), interactive-command blocking, and
    output truncation keep it bounded. Environment defaults to inheriting the
    parent, so host secrets (incl. LLM API keys) remain present unless a caller
    opts into `denied_env_patterns=LLM_API_KEY_ENV_PATTERNS`.

    `shell_config` is the spec's top-level ``shell:`` block (see
    ``specs._spec_shell_config``): ``cwd`` overrides the default ``PROJECT_ROOT``
    and ``executor`` selects the execution backend (local default, docker for
    the sandbox). Absent/``None`` -> ``PROJECT_ROOT`` + local executor. The
    built-in deny-list and interactive-command block always apply; these knobs
    only move the cwd and the backend.

    `agent_name` scopes shell's DURABLE background stores to
    ``data/processes/<agent>/`` (same root the coordinator's runs feed uses), so
    ``start_command`` processes are persisted and re-attached by pid across
    turns/restarts via ``ShellCommandStore`` + the ``ShellBackgroundQueue``.
    Background durability is always ON for agent-built shells; the toolset's
    ``processes_root=None`` legacy path is only for direct unit-test use.
    """
    cwd = PROJECT_ROOT
    if (
        isinstance(shell_config, dict)
        and isinstance(shell_config.get("cwd"), str)
        and shell_config.get("cwd")
    ):
        cwd = Path(os.path.expanduser(shell_config["cwd"]))
    executor = None
    if isinstance(shell_config, dict):
        executor = _resolve_shell_executor(shell_config.get("executor"))
    command_policy = None
    if isinstance(shell_config, dict) and shell_config.get("command_policy"):
        command_policy = parse_command_policy(shell_config["command_policy"])
    return _Shell(
        cwd=cwd,
        executor=executor,
        command_policy=command_policy,
        processes_root=str(PROCESSES_DIR),
        agent=agent_name or "demo_assistant",
        on_complete=on_complete,
    )


def _build_git(store, git_config: dict | None = None) -> _Git:
    """Safe git inspection + checkpoint tools rooted at the project root.

    `store` is accepted for a uniform builder signature but unused — git
    operates over the repo at PROJECT_ROOT, not the persistence store. The
    toolset confines every user-supplied path to this root; the two write tools
    (git_checkpoint / git_restore_paths) refuse empty path lists so the agent
    can never broad-`add -A` or wipe more than it named.

    `git_config` is the spec's top-level ``git:`` block (see
    ``specs._spec_git_config``): ``root`` overrides the default ``PROJECT_ROOT``
    so the capability can operate over another repo. Absent/``None`` ->
    ``PROJECT_ROOT``. Path-confinement to the root always applies; `root` never
    weakens it.
    """
    root = PROJECT_ROOT
    if (
        isinstance(git_config, dict)
        and isinstance(git_config.get("root"), str)
        and git_config.get("root")
    ):
        root = Path(os.path.expanduser(git_config["root"]))
    return _Git(root=root)


def _build_codebase_map(store, codebase_map_config: dict | None = None) -> _CodebaseMap:
    """Navigation-first codebase browser rooted at the project root.

    `store` is accepted for a uniform builder signature but unused — the codebase
    map reads/writes its own artifacts through the ``StorageService`` seam
    (``FlatFileStorage`` default, co-located under ``.agent-codeindex/``), not the
    persistence store. Rooting at PROJECT_ROOT indexes the repo the agent works in.
    The index is **on-demand**: ``get_overview`` / the ``run_codebase_map`` tool
    (re)build artifacts if missing or stale, so the capability constructs and
    registers without a pre-scan.

    `codebase_map_config` is the spec's top-level ``codebase_map:`` block (see
    ``specs._spec_codebase_map_config``): ``scan_root`` overrides the default
    ``PROJECT_ROOT``. Absent/``None`` -> ``PROJECT_ROOT``. Index artifacts always
    land under ``.agent-codeindex/`` inside whichever root is in effect.
    """
    scan_root = PROJECT_ROOT
    if (
        isinstance(codebase_map_config, dict)
        and isinstance(codebase_map_config.get("scan_root"), str)
        and codebase_map_config.get("scan_root")
    ):
        scan_root = Path(os.path.expanduser(codebase_map_config["scan_root"]))
    return _CodebaseMap(scan_root=scan_root)


def _build_web(store, search_domains: list[str] | None = None) -> _Web:
    """Tavily-backed web search + page extraction tools.

    `store` is accepted for a uniform builder signature but unused — web tool
    state is just the Tavily client, read lazily from `TAVILY_API_KEY` at call
    time (so the capability constructs and registers even before a key exists,
    and each tool reports the missing key/package rather than failing hard).

    `search_domains` is the extract trust boundary for `fetch_web_page`,
    declared per-spec (`web.search_domains` in the spec YAML); None -> the
    `PREFERRED_DOC_DOMAINS` default. Open web *search* is never restricted.
    """
    return _Web(search_domains=search_domains)


# Vault capability default root (no spec `vault:` block): a paf-owned playground
# under the repo's data/ tree — clean, deterministic, safe for tests. The real
# vault is ONLY reachable via an explicit `vault: root:` in an agent spec, never a
# hardcoded test/tool target.
_DEFAULT_VAULT_ROOT = DATA_DIR / "vault"


def _build_vault(store, vault_config: dict | None = None) -> _Vault:
    """Navigation-first note ladder over a markdown vault.

    `store` is accepted for a uniform builder signature but unused — the vault
    reads/writes its own flat-file index through the ``FlatFileVaultStorage``
    seam, stored repo-owned under ``data/.paf-vault-index/`` (keyed by the vault
    root's hash, NOT co-located inside the vault tree), not the persistence
    store.

    `vault_config` is the spec's top-level ``vault:`` block (see
    ``specs._spec_vault_config``): ``root`` (vault directory), optional
    ``allow``/``deny`` glob subtree patterns. Absent/empty block -> the paf-owned
    ``data/vault/`` playground default (never the user's real vault, which must be
    named explicitly in a spec).
    """
    is_dict = isinstance(vault_config, dict)
    root: Path | str = _DEFAULT_VAULT_ROOT
    if (
        is_dict
        and isinstance(vault_config.get("root"), str)
        and vault_config.get("root")
    ):
        root = vault_config["root"]
    allow = (
        [str(p) for p in vault_config.get("allow", [])]
        if is_dict and vault_config.get("allow")
        else []
    )
    deny = (
        [str(p) for p in vault_config.get("deny", [])]
        if is_dict and vault_config.get("deny")
        else []
    )
    return _Vault(vault_root=root, allow=allow, deny=deny)


def _build_visual(store, visual_config: dict | None = None) -> _Visual:
    """Vision routing for captured screenshots (a tool + a before_model_request hook).

    `store` is accepted for a uniform builder signature but unused — vision
    state is just the model slug, read from the spec's `visual:` block
    (`visual_config`) at build time. The capability provides two behaviors:

    - ``describe_screenshot`` tool — the model asks the vision model a TARGETED
      question ("is the padding OK?") about a captured screenshot ref.
    - ``before_model_request`` hook — when a vision model is configured
      (`visual.model` set), stashes raw capture pixels as `shot_...` refs and
      swaps them for text stubs so a text-only model never crashes on
      BinaryContent. When no vision model is configured the hook passes images
      through untouched.

    A contained image-in/text-out side-call, deliberately NOT a sub-agent: no
    child run, no delegation, no inherited tools.
    """
    if visual_config is None:
        # No `visual:` block — the capability is inert: no separate vision
        # model, images pass through to the main model untouched.
        return _Visual(model=None)
    model = visual_config.get("model")
    if not model:
        raise ValueError(
            "Spec declares a `visual:` block but no `model` — the Visual "
            "capability needs a vision model to route screenshots to."
        )
    return _Visual(model=model, model_settings=visual_config.get("model_settings"))


def _build_browser(store, browser_config: dict | None = None) -> _Browser:
    """Mobile-first, headless Chromium browser automation tools.

    `store` is accepted for a uniform builder signature but unused — browser
    state lives in the BrowserSession, not the persistence store. A paf-owned
    full copy of the harness `PlaywrightBrowser`, modified so the context is
    **mobile by default** (viewport 390×844, retina scale 3, real iPhone
    user-agent) and a `set_viewport` tool switches profiles mid-run. Defaults
    keep egress locked down: headless, private-address blocked (SSRF defense),
    an egress **allowlist** of local dev hosts, and no auto-install (the
    browser must be present via `playwright install chromium`).

    `browser_config` is the spec's top-level ``browser:`` block (see
    ``_spec_browser_config``). Two independent trust decisions, neither of
    which silently weakens the other:

    - ``allowed_domains`` — a non-empty list **wins** (overrides the
      capability's loopback-only default); absent/``None``/empty -> the
      capability's own default.
    - ``block_private_addresses`` — the **mechanical** SSRF guard (loopback /
      RFC-1918 / link-local / cloud-metadata). **Stays ``True`` (secure
      default) unless the spec explicitly declares ``false``.** Listing a
      private host in ``allowed_domains`` does NOT release this guard on its
      own — the guard outranks the allowlist, so both the allowlist entry
      AND ``block_private_addresses: false`` are required to actually reach a
      local host. Cloud-metadata (``169.254.169.254``) is never released by
      the guard either way.

    `chromium_sandbox` is deliberately False: Chromium's sandbox is fatal at
    launch on Ubuntu 23.10+ hosts where AppArmor restricts unprivileged user
    namespaces. The proper fix is enabling userns (sysctl + reboot) so sandbox
    can be held ON; until this host is configured that way, sandbox-off is
    required for the browser to launch at all. This is an environment-dependent
    escape hatch, not a recommendation to weaken isolation — revisit when userns
    is enabled.
    """
    allowed_domains = browser_config.get("allowed_domains") if browser_config else None
    # Secure by default: the private/loopback/link-local SSRF guard stays ON
    # unless the spec explicitly opts out (block_private_addresses: false).
    block_private = (
        browser_config.get("block_private_addresses") if browser_config else None
    )
    if block_private is None:
        block_private = True
    # Pass explicit kwargs (not a **-unpacked dict) so each field keeps its own
    # type — a heterogeneous dict makes pyright type every value as the union.
    if allowed_domains:
        return _Browser(
            headless=True,
            block_private_addresses=bool(block_private),
            auto_install_chromium=False,
            chromium_sandbox=False,
            viewport=(390, 844),
            mobile=True,
            is_mobile=True,
            has_touch=True,
            device_scale_factor=3,
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                "Mobile/15E148 Safari/604.1"
            ),
            allowed_domains=allowed_domains,
        )
    return _Browser(
        headless=True,
        block_private_addresses=bool(block_private),
        auto_install_chromium=False,
        chromium_sandbox=False,
        viewport=(390, 844),
        mobile=True,
        is_mobile=True,
        has_touch=True,
        device_scale_factor=3,
        user_agent=(
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
            "Mobile/15E148 Safari/604.1"
        ),
    )


def _build_skills(store, skills_config: dict | None = None) -> _Skills:
    """Skill library — list/load on demand.

    `store` is accepted for a uniform builder signature but unused — the skill
    library is self-contained over its configured library directories, not the
    persistence store. Indexes the configured SKILL.md skills + .md notes at
    construction; `list_skills`/`load_skill` expose the load mechanism (prompt
    stays thin, bodies pulled on demand). One paf-native capability for all skill
    functionality.

    `skills_config` is the spec's top-level ``skills:`` block (see
    ``specs._spec_skills_config``): ``libraries`` (one or more FULL absolute
    paths, `~`-expanded). Absent/``None`` -> the paf-owned
    ``data/skills/`` default — no implicit disk scanning beyond
    what a spec names explicitly (mirrors the vault capability's
    never-reach-the-real-thing-without-an-explicit-spec default).
    """
    libraries: tuple[str | Path, ...] = (DATA_DIR / "skills",)
    if isinstance(skills_config, dict):
        libs = skills_config.get("libraries")
        if isinstance(libs, (list, tuple)) and libs:
            libraries = tuple(Path(os.path.expanduser(str(v))) for v in libs if str(v))
    return _Skills(libraries=libraries)


def _build_tool_output_limits(store) -> ToolOutputLimits:
    """Framework-level tool-return reduction — a BUILTIN, not a paf icon-capability.

    Intercepts oversized tool returns in `after_tool_execute` and reduces them
    once, so a giant return doesn't dominate (and re-send on every later request
    of) the context window. Complements the eviction/compaction tier: a spilled
    payload is offloaded to the flat-file store instead of being retained in
    history, then paged back on demand via the builtin `read_tool_result`.

    Returns over 20_000 chars are LOSSLESSLY SPILLED to a `LocalFileStore` (a
    per-user temp directory) with a bounded truncation fallback — zero LLM
    cost, no silent drop. `read_tool_result` is registered by the capability's
    own toolset. The store is passed explicitly: the harness default is a store
    in the run's workspace, and most paf agents have none attached.

    `store` (the shared persistence store) is accepted for a uniform builder
    signature but unused.
    """
    return ToolOutputLimits(
        bands=[Band(over=20_000, action=Spill(then=Truncate()))],
        store=LocalFileStore(),
    )


def _build_warn_on_cache_busts(store) -> WarnOnCacheBusts:
    """Observational prompt-cache-collapse monitor — a BUILTIN, not a paf icon-capability.

    Attaches to the agent's model requests; after each response it reads
    ``usage.cache_read_tokens`` and warns (``CacheBustWarning``, stdlib ``warnings``)
    once when the established cacheable prefix collapses mid-run. A collapse means a
    moved prefix (reordered tools, injected timestamp, serialization-level block hop)
    or a provider-side expiry — either way it spikes billing by re-charging tokens a
    stable prefix would have served from cache. Purely observational: no tools,
    instructions, or model-settings, so it composes with every other capability.

    Thresholds are conservative (``min_missed_ratio`` 0.5, ``min_missed_tokens``
    1024): only a drop below half of an established >=1024-token prefix fires, so
    ordinary rounding or partial misses don't warn. The harness defaults (5% of
    2,000 tokens) are far more sensitive, and each warning becomes a red span in
    Langfuse. Whether a bust is "unexpected" is judged against the provider's own
    published cache window. Silent when caching is off or unreported
    (cache_read_tokens stays 0), so it never fires spuriously in runs that don't
    cache. Same drop-in pattern as ToolOutputLimits — consumed as a pydantic-ai
    AbstractCapability.

    BUSTS SURFACE AS LANGFUSE ERRORS (observability/instrumentation.py):
    a ``CacheBustWarning`` is a ``UserWarning`` (suppressed by default), so on its
    own it reaches neither stderr nor a trace. ``paf.observability._install_cache_bust_error_hook``
    installs a ``warnings.showwarning`` hook (at ``setup_observability()``, reached by
    both CLI + SSE) that flips the active model-request OTel span to ERROR and records
    the bust — it appears as a red error observation in Langfuse, and never aborts the turn.
    """

    return WarnOnCacheBusts(min_missed_ratio=0.5, min_missed_tokens=1024)


def _build_planning(store) -> Planning:
    """Model-owned task plan for long turns (paf-owned port).

    Gives the agent a structured, self-updating todo/plan it can create and edit
    (``write_plan`` / ``read_plan`` / ``add_task`` / ``update_task_status`` /
    ``remove_task``), with subtasks and dependencies (``add_subtask`` /
    ``set_dependency`` / ``get_available_tasks``), surfaced back to the model every request as a cache-safe
    tail reminder — so it never loses track of what it set out to do on a long
    run, and plan edits don't invalidate the prompt cache (the reminder rides
    behind a ``CachePoint`` breakpoint).

    Store: a per-conversation ``FilePlanStore`` (flat-file JSON under
    ``PROJECT_ROOT / "data/plans/<conversation_id>/plan.json"``). Plans persist
    across turns, runs, and process restarts — inspectable/diffable JSON (paf's
    flat-file-first principle) resolved per conversation via ``store_resolver``
    (falls back to a default-scope file when no ``conversation_id`` is set).
    This replaced the ephemeral ``InMemoryPlanStore`` so plans are durable
    and the web sidebar can render them on reload/restart. It is a thin
    wrapper over the shared ``StorageService`` seam — backend-swappable, never
    a bespoke path.

    `store` is accepted for a uniform builder signature but unused — the plan
    lives in the capability's per-conversation file store, resolved from the
    run context.
    """
    from paf.capabilities.planning import FilePlanStore, Planning
    from paf.capabilities.planning._registry import get_plan_emitter

    def _resolve_plan_store(ctx) -> FilePlanStore:
        cid = getattr(ctx, "conversation_id", None) or "default"
        # Attach the conversation's shared PlanEventEmitter so every store the
        # live agent writes to emits plan events. The SSE layer (ui/sse.py)
        # subscribes to the SAME emitter via the same scoped cid, bridging the
        # store <-> browser gap with zero client re-derivation.
        return FilePlanStore(
            root=_PLAN_ROOT,
            conversation_id=cid,
            event_emitter=get_plan_emitter(cid),
        )

    # Thin guidance: point out that write_plan returns the ids the model must
    # pass to a follow-up update_task_status(_es) — no read_plan round-trip
    # needed — and call out the one subtask behavior the model would not expect:
    # a parent completes on its own. The tool schemas document the rest.
    return Planning(
        store_resolver=_resolve_plan_store,
        enable_subtasks=True,
        guidance=(
            "For multi-step work, lay out the steps up front with `write_plan` and "
            "keep it current as you go. `write_plan` returns each step's [id] — "
            "use those ids directly in `update_task_status`/`update_task_statuses`; "
            "you do not need `read_plan` first. "
            "To clear the plan, call `write_plan` with `items: []`. "
            "Break a large step into subtasks with `add_subtask`, and use "
            "`set_dependency` when one step must wait for another (it stays `blocked` "
            "until that step is completed or cancelled). Work the subtasks, not the "
            "parent: a parent cannot be marked `completed` while any subtask is open, "
            "and it is completed automatically once its last subtask is done — do not "
            "complete it yourself. Reopening or adding a subtask reopens it."
        ),
    )


def _build_injection_guard(
    store, injection_config: dict | None = None
) -> _InjectionGuard:
    """paf-owned prompt-injection defense (``capabilities/injection``).

    A swappable scorer (stackone: patterns + local MiniLM classifier) plus a
    per-tool policy from the spec's ``guardrails.injection`` block (see
    ``specs._spec_injection_config``): ``off`` / ``observe`` / ``block`` per tool
    glob, a default action and an optional threshold. Absent block -> every tool
    ``block`` at the scorer's calibrated default, i.e. the behaviour of the
    harness ``PromptInjectionDefender`` it replaces. A malformed block raises at
    build time. Every noteworthy decision emits a ``paf.guardrail.
    prompt_injection`` span.

    `store` is accepted for a uniform builder signature but unused.
    """
    return _InjectionGuard(policy=parse_injection_policy(injection_config))


def _build_tool_gating(store, gating_config: dict | None = None) -> ToolGuardrail:
    """Per-call execution-time tool gate (BUILTIN shell, paf-owned policy).

    The harness ``ToolGuardrail`` is the shell; the policy is paf's own
    ``tool_gating_guard`` callable (see ``guardrails/tool_gating.py``), which
    refuses the highest-risk tools before they execute. The profile is the
    gatekeeper: this builder only runs when the profile allowlists
    ``ToolGating``.

    `gating_config` is the spec's top-level ``gating:`` block (see
    ``specs._spec_gating_config``): an ordered rule list compiled by
    ``guardrails.gating_policy`` into ``allow`` / ``block`` / ``approve``
    outcomes, optionally conditioned on a path argument. ``approve`` is enforced
    as a request to the person attached to the turn (``paf.approvals``), and as
    ``block`` when nobody is attached (background turns, piped runs). Absent block -> the legacy
    ``tool_gating_guard`` (``DEFAULT_GATED_TOOLS``). A malformed block raises at
    build time rather than loading half a policy.

    `store` is accepted for a uniform builder signature but unused — the
    policy is stateless per call.
    """
    policy = parse_policy(gating_config)
    if policy is not None:
        return ToolGuardrail(
            guard=make_gating_guard(policy, approval_available=approvals_available),
            id="tool_gating",
            description=(
                "Applies this agent's declarative gating policy to every tool "
                "call before it executes."
            ),
        )
    return ToolGuardrail(
        guard=tool_gating_guard,
        id="tool_gating",
        description=(
            "Refuses calls to the highest-risk tools (shell, git, filesystem "
            "writes) before they execute."
        ),
    )


def _build_secrets(store, secrets_config: dict | None = None) -> ToolGuardrail:
    """Secret-handle resolution + result redaction (BUILTIN shell, paf-owned policy).

    The harness ``ToolGuardrail`` is the shell; the policy is paf's own
    ``secret_handle_guard`` / ``secret_result_guard`` callables (see
    ``guardrails/secret_handles.py``). ``guard=`` resolves ``$SECRET:key``
    handles in tool args at execution time, so the real value never enters the
    window; ``result_guard=`` redacts any raw secret that leaks into a tool
    result. The profile is the gatekeeper: this builder only runs when the
    profile allowlists ``Secrets``.

    `secrets_config` is the spec's top-level ``secrets:`` block (see
    ``specs._spec_secrets_config``). Its ``allowed`` list names the only
    environment variables a handle may resolve; an empty list resolves nothing.
    Absent block (or no ``allowed`` key) keeps the original unrestricted
    behavior. The result-side redaction backstop scrubs credential-looking values
    (by variable name or value shape, never plain config like ``HOME``) from tool
    output; only allowlisted names get a reusable ``$SECRET:`` handle in place of
    the value, everything else becomes ``[redacted]``.

    `store` is accepted for a uniform builder signature but unused — the
    policy is stateless per call.
    """
    allowed = (secrets_config or {}).get("allowed")
    return ToolGuardrail(
        guard=(
            make_secret_handle_guard(allowed)
            if allowed is not None
            else secret_handle_guard
        ),
        result_guard=make_secret_result_guard(allowed),
        id="secrets",
        description=(
            "Resolves $SECRET:key handles in tool arguments at execution time "
            "and redacts raw secrets from tool results."
        ),
    )


def _build_subagents(
    store,
    filesystem_config: dict | None = None,
    agent_name: str | None = None,
    on_complete=None,
    subagents_config: dict | None = None,
) -> Subagents:
    """paf-owned subagent delegation over the spec-driven leaf roster.

    Each leaf is a spec under ``specs/sub_agents/`` (model, model_settings,
    description, instructions, plus a ``subagent:`` policy block) built by
    :func:`paf.capabilities.subagents.load_leaf`. Every child gets the SAME
    scoped FileSystem as the parent (root from ``filesystem_config`` / default
    PROJECT_ROOT) unless its spec sets ``filesystem.root_dir`` — the parent
    passes file paths and the child reads them in its own fresh run.

    ``agent_name`` scopes the durable task store to
    ``data/processes/<agent>/`` (the same root shell and the process feed use),
    where every delegation's record and ``.md`` output are persisted.

    ``on_complete`` is the embedding layer's completion hook (the web
    transport's notify, the same one shell gets): when a ``background=True``
    delegation settles it delivers the ``[system]`` result message to the
    originating conversation. ``None`` (CLI) means results are polled.

    ``subagents_config`` is the spec's ``subagents:`` block (see
    ``specs._spec_subagents_config``): ``agents`` replaces the default roster
    (:data:`_SUBAGENT_ROSTER`) and ``max_concurrent`` caps background tasks per
    conversation.

    NOT spec-serializable (holds live ``Agent`` instances), so it is built only
    here and supplied via the ``capabilities=`` seam, gated by the profile.
    """
    from paf.capabilities.subagents import load_leaf
    from paf.capabilities.subagents._background import DEFAULT_MAX_CONCURRENT

    config = subagents_config or {}
    roster = config.get("agents") or _SUBAGENT_ROSTER
    leaves = [
        load_leaf(
            name,
            store=store,
            filesystem_config=filesystem_config,
            agent_name=agent_name,
        )
        for name in roster
    ]
    return Subagents(
        leaves=leaves,
        processes_root=str(PROCESSES_DIR),
        agent=agent_name or "demo_assistant",
        on_complete=on_complete,
        max_concurrent=config.get("max_concurrent", DEFAULT_MAX_CONCURRENT),
    )


def _build_mcp(store, mcp_config: str | None = None) -> list:
    """Build builtin ``MCP`` capabilities from a ``mcpServers`` JSON path.

    ``mcp_config`` is a path (str) to a Claude-Desktop-style ``mcpServers`` JSON,
    declared by the spec's top-level ``mcp: {config: <path>}`` block.
    ``load_mcp_toolsets`` reads it with the builtin recursive ``${VAR}`` /
    ``${VAR:-default}`` env expander (so server creds come from .env, never the
    spec), returning one *prefixed* ``MCPToolset`` per server. Each is wrapped in
    the builtin ``MCP(local=toolset)`` capability so the profile gate + the
    ``capabilities=`` seam see it like any other capability (connection stays
    lazy — building spawns no subprocess / opens no socket until the agent's
    first turn needs the server's tools).

    Returns a list of ``MCP`` capabilities (one per server). ``None``/absent
    config -> ``[]`` (the profile-gated capability is simply absent). A
    declared-but-missing file raises ``FileNotFoundError`` loudly at build — a
    bad config should fail loudly, not silently no-op.
    """
    if not mcp_config:
        return []
    path = Path(os.path.expanduser(mcp_config))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return [MCP(local=cast(Any, toolset)) for toolset in load_mcp_toolsets(path)]


def _build_scheduled_tasks(
    store, scheduled_tasks_config=None, agent_name=None, queue=None
):
    """Scheduled-tasks capability: per-conversation CRUD + manual run.

    ``queue`` is the run-layer ``ScheduledTaskQueue`` (wired in paf-sse); when
    None the ``run`` tool reports the queue as unwired. The capability owns its
    own per-conversation store, so the shared ``store`` arg is ignored.
    """
    del store
    root = "data/scheduled_tasks"
    if isinstance(scheduled_tasks_config, dict):
        root = scheduled_tasks_config.get("root", root)
    return ScheduledTasks(root=root, agent=agent_name or "", queue=queue)


def _build_push(notifier=None):
    """Push capability: the ``send_push`` tool, present only when ``notifier`` is
    wired (paf-sse). Profile-gated like every capability: no ``"Push"`` in the
    profile's allowlist, no tool."""
    return Push(notifier=notifier)


def build_runtime_capabilities(
    profile: AgentProfile,
    compaction_config=None,
    web_search_domains: list[str] | None = None,
    browser_config: dict | None = None,
    vault_config: dict | None = None,
    memory_config: dict | None = None,
    filesystem_config: dict | None = None,
    shell_config: dict | None = None,
    python_config: dict | None = None,
    gating_config: dict | None = None,
    secrets_config: dict | None = None,
    injection_config: dict | None = None,
    git_config: dict | None = None,
    codebase_map_config: dict | None = None,
    skills_config: dict | None = None,
    keywords_config: dict | None = None,
    mcp_config: str | None = None,
    agent_name: str | None = None,
    on_complete=None,
    subagents_config: dict | None = None,
    visual_config: dict | None = None,
    search_config: dict | None = None,
    workspace_config: dict | None = None,
    scheduled_tasks_queue=None,
    push_notifier=None,
) -> list:
    """Build the capabilities the profile allowlists, over one shared store.

    Persistence + recall select per :data:`PERSISTENCE_BUILDERS`, all handed the
    SAME store (so write / search / read_turn provably resolve the same history).
    ``Compaction`` (the no-summary ladder) is appended iff the profile allowlists
    it. A profile may list fewer — e.g. a minimal agent with only
    ``StepPersistence`` — and that is exactly what it gets. Capability names
    without a Python builder are left for the spec-declared (YAML) path.
    """
    # Both builders are keyed by the SAME names a profile allowlists, so the
    # profile is the gatekeeper for each.
    builders = {
        **PERSISTENCE_BUILDERS,
        # Memory is its own journal-backed store (JournalViewStore), not the shared
        # run store, so its builder ignores the arg — but keeping it here means the
        # profile's `Memory` allowlist entry actually gets built.
        "Memory": _build_memory,
        # Per-conversation background thread summarizer. OFF by default — only
        # built when the profile allowlists it.
        "ThreadSummarizer": _build_thread_summarizer,
        # Per-turn relevance hints from the curated keyword store. OFF unless the
        # profile allowlists it; injects nothing until a store exists.
        "KeywordCurator": _build_keyword_curator,
        # Scoped filesystem (paf-owned harness copy) rooted at the project root.
        "FileSystem": _build_filesystem,
        # Shell command execution (paf-owned harness copy) rooted at the project root.
        "Shell": _build_shell,
        # Sandboxed Python execution (paf-owned): Docker-by-default, refuses the
        # unsandboxed local backend unless the spec opts in.
        "Python": _build_python,
        # Safe git inspection + checkpoint tools rooted at the repo root.
        "Git": _build_git,
        # Codebase-map: navigation-first codebase browser rooted at the repo root
        # (overview/lookup_file/read_block/search_codebase + run_codebase_map).
        "CodebaseMap": _build_codebase_map,
        # Vault: navigation-first note ladder over a markdown vault —
        # vault_overview/lookup_note/read_note + note_find/note_search.
        "Vault": _build_vault,
        # Tavily-backed web search + page extraction tools.
        "Web": _build_web,
        # Vision-model description of captured screenshots (describe_screenshot).
        "Visual": _build_visual,
        # Mobile-first headless browser automation (paf-owned copy, mobile default).
        "Browser": _build_browser,
        # Skill library — list/load on demand + deterministic pointer trigger.
        "Skills": _build_skills,
        "ScheduledTasks": _build_scheduled_tasks,
        "Push": _build_push,
        # Builtin MCP servers loaded from a mcpServers JSON (env-expanded).
        "MCP": _build_mcp,
        # Framework-level tool-return reduction (BUILTIN, paf-supplied config).
        "ToolOutputLimits": _build_tool_output_limits,
        # Observational prompt-cache-collapse monitor (BUILTIN, paf-supplied config).
        "WarnOnCacheBusts": _build_warn_on_cache_busts,
        # Sub-agent delegation (BUILTIN, not a paf port) — explicit child leaf agents.
        "Subagents": _build_subagents,
        # Model-owned task plan for long turns (paf-owned port).
        "Planning": _build_planning,
        # Prompt-injection defense (paf-owned): scores every tool result and
        # applies the spec's per-tool policy (off / observe / block).
        "InjectionGuard": _build_injection_guard,
        # Deprecated alias of InjectionGuard (profiles written before the rehome).
        "Guardrails": _build_injection_guard,
        # Per-call execution-time tool gate (BUILTIN shell, paf-owned policy) —
        # refuses the highest-risk tools before they execute. The profile is
        # only a build-time tool allowlist; this is the missing per-call gate.
        "ToolGating": _build_tool_gating,
        # Secret-handle resolution + result redaction (BUILTIN shell, paf-owned
        # policy) — the model proposes $SECRET:key, the runtime resolves it at
        # execution time so the real value never enters the window.
        "Secrets": _build_secrets,
    }
    store = _build_store()
    capabilities = []
    for name, builder in builders.items():
        if name not in profile.capability_names:
            continue
        if name == "Web":
            capabilities.append(_build_web(store, web_search_domains))
        elif name == "Browser":
            capabilities.append(_build_browser(store, browser_config))
        elif name == "Vault":
            capabilities.append(_build_vault(store, vault_config))
        elif name == "FileSystem":
            capabilities.append(
                _build_filesystem(
                    store,
                    filesystem_config,
                    # One spec switch (`workspace.expose_paths: false`) hides host
                    # paths from the workspace block AND from file-tool errors.
                    expose_root=(workspace_config or {}).get("expose_paths", True),
                )
            )
        elif name == "Shell":
            capabilities.append(
                _build_shell(store, shell_config, agent_name, on_complete)
            )
        elif name in ("InjectionGuard", "Guardrails"):
            if name == "Guardrails" and "InjectionGuard" in profile.capability_names:
                continue  # alias listed alongside the real name: build once
            capabilities.append(_build_injection_guard(store, injection_config))
        elif name == "ToolGating":
            capabilities.append(_build_tool_gating(store, gating_config))
        elif name == "Secrets":
            capabilities.append(_build_secrets(store, secrets_config))
        elif name == "Python":
            capabilities.append(_build_python(store, python_config))
        elif name == "Git":
            capabilities.append(_build_git(store, git_config))
        elif name == "CodebaseMap":
            capabilities.append(_build_codebase_map(store, codebase_map_config))
        elif name == "Skills":
            capabilities.append(_build_skills(store, skills_config))
        elif name == "MCP":
            # One MCP capability per server in the JSON; extends (not append) so
            # a profile allowlisting "MCP" gets every declared server.
            capabilities.extend(_build_mcp(store, mcp_config))
        elif name == "Memory":
            # Memory owns every memory config variable (e.g. `consolidate_every`
            # — the background-consolidation cadence) read from the spec's
            # top-level `memory:` block. The background queue is a separate
            # framework-level service the Memory capability hands values down to.
            capabilities.append(_build_memory(store, memory_config, agent_name))
        elif name == "ThreadSummarizer":
            # Keeps the (agent_name, future thread_config) wiring seam ready for
            # a capture hook. Gated by the allowlist via the loop's `name not in
            # capability_names` continue above.
            capabilities.append(_build_thread_summarizer(store, None, agent_name))
        elif name == "KeywordCurator":
            capabilities.append(
                _build_keyword_curator(store, keywords_config, agent_name)
            )
        elif name == "Subagents":
            # Each leaf gets the SAME scoped FileSystem as the main model (root
            # from `filesystem_config` / default PROJECT_ROOT), so the parent
            # passes file paths in `task` and the child reads the content in its
            # own fresh run. `agent_name` scopes the durable task store.
            capabilities.append(
                _build_subagents(
                    store, filesystem_config, agent_name, on_complete, subagents_config
                )
            )
        elif name == "Visual":
            # Vision model slug from the spec's `visual:` block. Absent -> the
            # capability builds inert (images pass through to the main model).
            capabilities.append(_build_visual(store, visual_config))
        elif name == "ConversationSearch":
            # Spec-driven recall knobs from the `search:` block (max_matches /
            # context_lines / bm25_k1 / bm25_b / scope). Absent -> the builder
            # applies its defaults (scope=conversation, 10/5/1.5/0.75).
            capabilities.append(builders["ConversationSearch"](store, search_config))
        elif name == "ScheduledTasks":
            # Per-conversation CRUD + manual run. `scheduled_tasks_queue` is the
            # run-layer queue (wired in paf-sse); None -> the `run` tool reports
            # the queue as unwired. The capability owns its own per-conversation
            # store, so the shared `store` arg is ignored.
            capabilities.append(
                _build_scheduled_tasks(store, None, agent_name, scheduled_tasks_queue)
            )
        elif name == "Push":
            capabilities.append(_build_push(push_notifier))
        else:
            capabilities.append(builder(store))
    if _COMPACTION_CAPABILITY in profile.capability_names:
        capabilities.append(build_compaction(compaction_config))
    # `read_turn("#N")` resolves memory ids when the agent has Memory: point
    # ConversationSearch at this agent's memory dir.
    if agent_name and "Memory" in profile.capability_names:
        from paf.capabilities.memory._window import memory_dir_for
        from paf.capabilities.search import ConversationSearch as _CS

        for cap in capabilities:
            if isinstance(cap, _CS):
                cap.memory_directory = str(memory_dir_for(agent_name))

    # Workspace context: where each built tool family's paths start, the user's
    # time zone, and spec `workspace.notes`. Built LAST, from the capabilities
    # above, so it lists only tools this agent actually has. Profile-gated like
    # everything else.
    if "Workspace" in profile.capability_names:
        from paf.capabilities.workspace import build_workspace

        capabilities.append(
            build_workspace(
                capabilities,
                (workspace_config or {}).get("notes"),
                (workspace_config or {}).get("expose_paths", True),
            )
        )
    # Approval plumbing, not a capability choice: the broker never approves
    # anything by itself. It answers pydantic-ai's deferred-tool hook by asking
    # the person attached to the turn, and denies at once when nobody is
    # (background turns, piped runs), so a guard that asks for approval can never
    # leave a run ending in an unhandled deferred request.
    capabilities.append(ApprovalBroker())
    # NOTE: the ThreadSummarizer no longer captures evictions — the
    # capture-WAL wiring (attach_to_window) was removed. The
    # runtime trigger resolves the unsummarized span directly from the step
    # store via resolve_unseen_span(conversation_id=...); nothing binds the
    # summarizer to the compaction window anymore, so eviction and
    # summarization are fully decoupled (their only shared trait is both being
    # background).
    return capabilities
