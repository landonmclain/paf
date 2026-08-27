"""Runtime capability builders — the ``capabilities=`` seam for Agent.from_spec.

Spec files (`specs/*.yaml`) go through pydantic-ai's ``AgentSpec``/``Agent.from_spec``,
which resolves each top-level ``capabilities:`` entry via that class's ``from_spec()`` and
— crucially — only ONE level deep (see ``_capabilities_from_spec`` in pydantic-ai's
``agent/__init__.py``). Capabilities that can't be expressed that way — because a class
doesn't override ``from_spec`` or needs real objects as arguments — are assembled here and
injected via the ``capabilities=`` kwarg, which ``Agent.from_spec`` MERGES with any
spec-declared capabilities.

**The profile is the gatekeeper.** ``build_runtime_capabilities(profile)`` builds exactly
the capabilities the profile's ``capability_names`` allowlists, over one shared store.
We no longer declare capabilities in YAML, so this Python ``capabilities=`` seam is the
ONLY thing that reaches the agent — which means the profile's allowlist is what decides
"what this agent gets." No hardcoded list here; it is the resolution of the profile.

Persistent capabilities (StepPersistence, ConversationSearch) select their
builders from :data:`paf.capabilities.persistence.PERSISTENCE_BUILDERS`. Context
compaction builds via ``build_compaction()`` **iff the profile allowlists ``Compaction``**
— the ladder WITHOUT the summarizing tier (the LLM rewrite is mutually exclusive with the
recall-ladder injection; see ``compaction.py``). A profile simply omits a name from
its ``capability_names`` and that agent won't get it.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits
from pydantic_ai_harness.warn_on_cache_busts import WarnOnCacheBusts

from paf.capabilities.browser import Browser as _Browser
from paf.capabilities.codebase_map import CodebaseMap as _CodebaseMap
from paf.capabilities.compaction import build_compaction
from paf.capabilities.fs import FileSystem as _FileSystem
from paf.capabilities.git import Git as _Git
from paf.capabilities.memory import Memory
from paf.capabilities.persistence import PERSISTENCE_BUILDERS, _build_store
from paf.capabilities.profiles import AgentProfile
from paf.capabilities.search import ConversationSearch, SnapshotHistorySource
from paf.capabilities.shell import Shell as _Shell
from paf.capabilities.skills import Skills as _Skills
from paf.capabilities.vault import Vault as _Vault
from paf.capabilities.visual import Visual as _Visual
from paf.capabilities.web import Web as _Web
from paf.config import PROJECT_ROOT, settings

# The profile name that gates context compaction in build_runtime_capabilities.
_COMPACTION_CAPABILITY = "Compaction"

# Where the Memory capability writes its readable `.md` bodies + `.memory-state.json`
# CAS sidecar, anchored to the repo root (same convention as persistence/_STORE_DIR).
# Directory comes from the shared Settings object (MEMORY_DIR / .env), so the
# consolidation owner and the assistant always target the same store.
_MEMORY_DIR = PROJECT_ROOT / Path(settings.memory_dir)


def _build_memory(store) -> Memory:
    """M2 journal-view durable memory, INJECT-ONLY for the assistant.

    Mirrors the benchmark assembly in `tests/framework.py`: the model carries
    the rendered journal (MEMORY.md) injected into every request via
    before_model_request, and is exposed ZERO memory tools (expose_tools=False)
    -- not even read/search -- because a separate consolidation owner sub-agent
    is the sole writer. This converges prod's memory surface on the M2 contract
    (JournalViewStore, read-only, inject-only), the same shape the LongMemEval
    harness measures.

    `store` (the shared persistence store) is accepted for a uniform builder
    signature but unused. inject_memory=True, writable=False, and an empty
    guidance string mirror _build_agent; max_lines=1_000_000 / max_tokens=65_536
    are high, non-evicting caps so the full rendered journal always shows.
    """
    from paf.capabilities.memory._view import JournalViewStore

    return Memory(
        store=JournalViewStore(_MEMORY_DIR),
        inject_memory=True,
        writable=False,
        expose_tools=False,
        max_lines=1_000_000,
        max_tokens=65_536,
        guidance="",
    )


def _conversation_search_suppressed(store):
    """ConversationSearch with its own instruction scrap disabled.

    Used when Memory owns the composite recall ladder: Memory's composite
    `_DEFAULT_GUIDANCE` already names `search_conversation_history`'s slot, so this
    capability's standalone "a search_conversation_history tool can retrieve
    exact details…" fragment would be duplicative and potentially conflicting.
    Same shared store either way — this only toggles instruction emission.
    """
    return ConversationSearch(
        source=SnapshotHistorySource(store),
        scope="conversation",
        add_instructions=False,
    )


def _build_filesystem(store, filesystem_config: dict | None = None) -> _FileSystem:
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
    gates always apply, whichever root is in effect.
    """
    root_dir = PROJECT_ROOT
    if isinstance(filesystem_config, dict) and isinstance(
        filesystem_config.get("root_dir"), str
    ) and filesystem_config.get("root_dir"):
        root_dir = Path(os.path.expanduser(filesystem_config["root_dir"]))
    return _FileSystem(root_dir=root_dir)


def _build_shell(store, shell_config: dict | None = None) -> _Shell:
    """Shell command execution rooted at the project root (paf-owned copy).

    `store` is accepted for a uniform builder signature but unused — the shell
    is self-contained over its own cwd, not the persistence store. Rooting at
    PROJECT_ROOT lets the model run commands against the repo; the harness
    toolset's deny-list (rm/dd/shutdown/...), interactive-command blocking, and
    output truncation keep it bounded. Environment defaults to inheriting the
    parent, so host secrets (incl. LLM API keys) remain present unless a caller
    opts into `denied_env_patterns=LLM_API_KEY_ENV_PATTERNS`.

    `shell_config` is the spec's top-level ``shell:`` block (see
    ``specs._spec_shell_config``): ``cwd`` overrides the default ``PROJECT_ROOT``.
    Absent/``None`` -> ``PROJECT_ROOT``. The built-in deny-list and
    interactive-command block always apply; this knob only moves the cwd.
    """
    cwd = PROJECT_ROOT
    if isinstance(shell_config, dict) and isinstance(shell_config.get("cwd"), str) and shell_config.get("cwd"):
        cwd = Path(os.path.expanduser(shell_config["cwd"]))
    return _Shell(cwd=cwd)


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
    if isinstance(git_config, dict) and isinstance(git_config.get("root"), str) and git_config.get("root"):
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
    if isinstance(codebase_map_config, dict) and isinstance(
        codebase_map_config.get("scan_root"), str
    ) and codebase_map_config.get("scan_root"):
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
_DEFAULT_VAULT_ROOT = PROJECT_ROOT / "data" / "vault"


def _build_vault(store, vault_config: dict | None = None) -> _Vault:
    """Navigation-first note ladder over a markdown vault (Tier A read tools).

    `store` is accepted for a uniform builder signature but unused — the vault
    reads/writes its own flat-file index through the ``FlatFileVaultStorage``
    seam (co-located under ``.paf-vault/``), not the persistence store.

    `vault_config` is the spec's top-level ``vault:`` block (see
    ``specs._spec_vault_config``): ``root`` (vault directory), optional
    ``allow``/``deny`` glob subtree patterns. Absent/empty block -> the paf-owned
    ``data/vault/`` playground default (never the user's real vault, which must be
    named explicitly in a spec).
    """
    is_dict = isinstance(vault_config, dict)
    root: Path | str = _DEFAULT_VAULT_ROOT
    if is_dict and isinstance(vault_config.get("root"), str) and vault_config.get("root"):
        root = vault_config["root"]
    allow = [str(p) for p in vault_config.get("allow", [])] if is_dict and vault_config.get("allow") else []
    deny = [str(p) for p in vault_config.get("deny", [])] if is_dict and vault_config.get("deny") else []
    return _Vault(vault_root=root, allow=allow, deny=deny)


def _build_visual(store) -> _Visual:
    """Vision routing for captured screenshots (a tool + a before_model_request hook).

    `store` is accepted for a uniform builder signature but unused — vision
    state is just the model slug, read lazily from `VISION_MODEL` at call time
    (default `openrouter:xiaomi/mimo-v2.5`), so the capability constructs and
    registers even before a key is configured and the tool reports a clear
    message rather than failing hard. The capability provides two behaviors:

    - ``describe_screenshot`` tool — the model asks the vision model a TARGETED
      question ("is the padding OK?") about a captured screenshot ref.
    - ``before_model_request`` hook — when `VISION_MODEL` is set (text-only main
      model), stashes raw capture pixels as `shot_...` refs and swaps them for
      text stubs so the text-only model never crashes on BinaryContent. When
      `VISION_MODEL` is unset the hook passes images through untouched.

    A contained image-in/text-out side-call, deliberately NOT a sub-agent (per
    Landon): no child run, no delegation, no inherited tools.
    """
    return _Visual()


def _build_browser(store, browser_config: dict | None = None) -> _Browser:
    """Mobile-first, headless Chromium browser automation tools.

    `store` is accepted for a uniform builder signature but unused — browser
    state lives in the BrowserSession, not the persistence store. A paf-owned
    full copy of the harness `PlaywrightBrowser` (same copy-and-own pattern as
    Memory / FileSystem / Shell), modified so the context is **mobile by default**
    and a `set_viewport` tool switches profiles mid-run. Defaults keep egress
    locked down for a demo agent: headless, private-address blocked (SSRF
    defense), an egress **allowlist** of local dev hosts so the browser can reach
    sites being built on this machine, and no auto-install (the browser must be
    present via `playwright install chromium`). `headless` stays True so a
    chat-spawned CLI never pops a window; `auto_install_chromium` stays False so
    a missing browser raises a clear `playwright install chromium` hint instead
    of silently fetching a ~150MB binary mid-session.

    `browser_config` is the spec's top-level ``browser:`` block (see
    ``_spec_browser_config``). Two independent trust decisions, neither of
    which silently weakens the other:

    - ``allowed_domains`` — a non-empty list **wins** (overrides the
      env-driven ``settings.playwright_allowed_domains`` /
      PLAYWRIGHT_ALLOWED_DOMAINS default); absent/``None``/empty -> the env
      default.
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
    namespaces (`kernel.apparmor_restrict_unprivileged_userns=1`, the default).
    The proper fix is enabling userns (sysctl + reboot) so sandbox can be held
    ON; until this host is configured that way, sandbox-off is required for the
    browser to launch at all. This is an environment-dependent escape hatch, not
    a recommendation to weaken isolation — revisit when userns is enabled.

    `block_private_addresses=True` (the default) STAYS on: the hosts in the
    resolved allowlist are the ONLY addresses reachable. The cloud-metadata
    endpoint and LAN RFC-1918 ranges remain refused, so SSRF defense is
    preserved while still browsing the allowlisted local/staging sites.

    Mobile is the default (viewport 390×844, `mobile=True`, retina scale 3, a
    real iPhone user-agent so responsive sites serve their mobile layout) —
    mobile is the only layout that matters — and is configurable on the
    capability: flip `mobile=False` for desktop or change `viewport` for a
    tablet, and the agent can switch these at runtime via the `set_viewport`
    tool.
    """
    allowed_domains = (
        browser_config.get("allowed_domains")
        if browser_config
        else None
    )
    if not allowed_domains:
        allowed_domains = settings.playwright_allowed_domains
    # Secure by default: the private/loopback/link-local SSRF guard stays ON
    # unless the spec explicitly opts out (block_private_addresses: false).
    block_private = (
        browser_config.get("block_private_addresses")
        if browser_config
        else None
    )
    if block_private is None:
        block_private = True
    return _Browser(
        headless=True,
        block_private_addresses=bool(block_private),
        allowed_domains=allowed_domains,
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


def _build_skills(store) -> _Skills:
    """Skill library — list/load on demand + deterministic pointer trigger.

    `store` is accepted for a uniform builder signature but unused — the skill
    library is self-contained over its demo_skills/ directory, not the
    persistence store. Indexes the repo's `demo_skills/` (SKILL.md skills +
    .md notes) at construction; `list_skills`/`load_skill` expose the load
    mechanism (prompt stays thin, bodies pulled on demand) and `point_to_skill`
    pre-arms which skill a message needs via zero-LLM BM25 over the declared
    frontmatter vocab (the skill-note-hint pointer engine re-homed into paf).
    One paf-native capability for all skill functionality (Landon 2026-08-19).
    """
    return _Skills(libraries=(PROJECT_ROOT / "demo_skills",), top_k=3)


def _build_tool_output_limits(store) -> ToolOutputLimits:
    """Framework-level tool-return reduction — a BUILTIN, not a paf icon-capability.

    Intercepts oversized tool returns in `after_tool_execute` and reduces them
    once, so a giant return doesn't dominate (and re-send on every later request
    of) the context window. Complements the eviction/compaction tier: a spilled
    payload is offloaded to the flat-file store instead of being retained in
    history, then paged back on demand via the builtin `read_tool_result`.

    Config (per Landon: "plug-and-play with some configuration"): by default,
    returns over `_DEFAULT_THRESHOLD` (10_000 chars) are LOSS LESSLY SPILLED to a
    `LocalFileStore` with a bounded truncation fallback — zero LLM cost, no silent
    drop. `read_tool_result` is registered by the capability's own toolset.

    `store` (the shared persistence store) is accepted for a uniform builder
    signature but unused — ToolOutputLimits uses its own `LocalFileStore` spill
    backend by default (swap via `store=` for a custom OverflowStore).
    """
    return ToolOutputLimits()


def _build_warn_on_cache_busts(store) -> WarnOnCacheBusts:
    """Observational prompt-cache-collapse monitor — a BUILTIN, not a paf icon-capability.

    Attaches to the agent's model requests; after each response it reads
    ``usage.cache_read_tokens`` and warns (``CacheBustWarning``, stdlib ``warnings``)
    once when the established cacheable prefix collapses mid-run. A collapse means a
    moved prefix (reordered tools, injected timestamp, serialization-level block hop)
    or a provider-side expiry — either way it spikes billing by re-charging tokens a
    stable prefix would have served from cache. Purely observational: no tools,
    instructions, or model-settings, so it composes with every other capability.

    Defaults are conservative (collapse_ratio 0.5, min_prefix_tokens 1024,
    cache_ttl_seconds 300): only a drop below half of an established >=1024-token
    prefix fires, so ordinary rounding or partial misses don't warn. Silent when
    caching is off or unreported (cache_read_tokens stays 0), so it never fires
    spuriously in runs that don't cache. Same drop-in pattern as ToolOutputLimits
    — consumed as a pydantic-ai AbstractCapability, per Landon.

    To surface busts in Langfuse traces, route stdlib warnings into logging once at
    startup: ``logging.captureWarnings(True)`` (Logfire bridges logging, not warnings;
    they reach the trace via the ``py.warnings`` logger).
    """

    return WarnOnCacheBusts()


def _build_planning(store) -> Planning:
    """Model-owned task plan for long turns (BUILTIN, not a paf port).

    Gives the agent a structured, self-updating todo/plan it can create and edit
    (``write_plan`` / ``read_plan`` / ``add_task`` / ``update_task_status`` /
    ``remove_task``), surfaced back to the model every request as a cache-safe
    tail reminder — so it never loses track of what it set out to do on a long
    run, and plan edits don't invalidate the prompt cache (the reminder rides
    behind a ``CachePoint`` breakpoint).

    Store: a shared ``InMemoryPlanStore`` (not the persistence store). The plan
    persists across turns/runs within this process via the single shared store
    instance, which is what the demo needs to actually exercise long-turn
    tracking. It is deliberately NOT wired to the flat-file ``FileStepStore``
    (that's run-history JSONL, not a plan store) and NOT to Sqlite/Postgres for
    now — durable cross-session plans are a future option
    (``SqlitePlanStore('plan.db', session=...)``) if the demo proves out.

    `store` is accepted for a uniform builder signature but unused — the plan
    lives in the capability's own in-memory store, not the persistence store.
    Consumed as a pydantic-ai AbstractCapability from the harness, same drop-in
    pattern as ToolOutputLimits / WarnOnCacheBusts / Subagents per Landon.
    """
    from pydantic_ai_harness.planning import InMemoryPlanStore

    # Thin guidance: the write_plan/read_plan/add_task/... tool schemas already
    # document themselves, so just cue the model to plan multi-step work rather
    # than repeat the harness's longer usage blurb in the instruction channel.
    return Planning(
        store=InMemoryPlanStore(),
        guidance=(
            "For multi-step work, lay out the steps up front with `write_plan` and "
            "keep it current as you go."
        ),
    )


# Default model for the demo sub-agents. Bare deepseek slug (NOT the `:nitro`
# variant) for determinism — :nitro's `provider.sort: throughput` routing bounces
# across upstream providers (fp8 vs fp4) and gives run-to-run nondeterminism (see
# specs.py `_openrouter_model_settings`). Override via SUBAGENT_MODEL / .env.
_SUBAGENT_MODEL = os.environ.get(
    "SUBAGENT_MODEL", "openrouter:deepseek/deepseek-v4-flash-0731"
)


def _build_subagents(store) -> SubAgents:
    """Explicit sub-agent delegation (BUILTIN, not a paf port) — demo leaf agents.

    Lets the demo agent delegate a self-contained task to a named child agent via
    one `delegate_task(agent_name, task)` tool. Each child runs in its own fresh,
    isolated run and never sees the parent conversation (the parent must pass
    everything the task needs). Two demo children, both on the demo model:

    - ``researcher`` — gathers/verifies facts and returns a tight evidence-backed
      brief (no tools, prompt-only; uses the same model as the demo subject).
    - ``writer`` — drafts prose/format from material the parent passes in (no
      tools; deterministic same-model leaf).

    This is a pydantic-ai AbstractCapability consumed from the harness per Landon
    (same drop-in pattern as ToolOutputLimits / WarnOnCacheBusts). It is NOT
    spec-serializable (holds live `Agent` instances; get_serialization_name ->
    None), so it is built ONLY here in Python by passing explicit `SubAgent`
    entries — it can never resolve from a YAML spec. `agent_folders=None` disables
    disk-agent loading entirely so the demo is purely explicit and deterministic
    (it does not vary with whatever markdown agent files happen to exist in
    `~/.agents/` or the project). `inherit_tools` is off (children don't get the
    parent's toolset), keeping delegation a clean leaf call. `store` unused
    (uniform builder signature).
    """
    from pydantic_ai import Agent

    researcher = Agent(
        _SUBAGENT_MODEL,
        name="researcher",
        description="Gathers and verifies facts, returns a tight evidence-backed brief",
        system_prompt=(
            "You are a research sub-agent. You do not see the parent conversation. "
            "Given a research task, return a concise, factual, evidence-backed brief. "
            "State what is certain vs. uncertain. Do not fabricate; say 'unknown' when you "
            "do not have evidence for something."
        ),
    )
    writer = Agent(
        _SUBAGENT_MODEL,
        name="writer",
        description="Drafts clear prose from material the delegating agent provides",
        system_prompt=(
            "You are a writing sub-agent. You do not see the parent conversation. "
            "Given source material and a writing task, produce clean, clear prose. "
            "Rewrite for clarity and structure, preserve all facts, add nothing."
        ),
    )
    return SubAgents(
        agents=[SubAgent(researcher), SubAgent(writer)],
        agent_folders=None,  # purely explicit, deterministic demo (no disk loading)
    )


def build_runtime_capabilities(
    profile: AgentProfile,
    compaction_config=None,
    web_search_domains: list[str] | None = None,
    browser_config: dict | None = None,
    vault_config: dict | None = None,
    filesystem_config: dict | None = None,
    shell_config: dict | None = None,
    git_config: dict | None = None,
    codebase_map_config: dict | None = None,
) -> list:
    """Build the capabilities the profile allowlists, over one shared store.

    Persistence + recall select per :data:`PERSISTENCE_BUILDERS`, all handed the
    SAME store (so write / search / read_turn provably resolve the same history).
    ``Compaction`` (the no-summary ladder) is appended iff the profile allowlists
    it. A profile may list fewer — e.g. a minimal agent with only
    ``StepPersistence`` — and that is exactly what it gets. Capability names
    without a Python builder are left for the spec-declared (YAML) path.

    **Recall-instruction ownership:** exactly ONE capability emits the composite
    recall ladder. When ``Memory`` is allowlisted it owns it (durable-fact +
    past-turn + last-rung history, in ``_DEFAULT_GUIDANCE``), so
    ``ConversationSearch`` gets ``add_instructions=False`` here to keep that single
    ladder the ONLY recall guidance the model sees. The suppression gate lives
    here (build time) because capabilities cannot see each other;
    ``add_instructions=False`` never disables tool registration.
    """
    # Both builders are keyed by the SAME names a profile allowlists, so the
    # profile is the gatekeeper for each.
    builders = {
        **PERSISTENCE_BUILDERS,
        # Memory is its own journal-backed store (JournalViewStore), not the shared
        # run store, so its builder ignores the arg — but keeping it here means the
        # profile's `Memory` allowlist entry actually gets built (un-gated, live-test beat).
        "Memory": _build_memory,
        # Scoped filesystem (paf-owned harness copy) rooted at the project root.
        "FileSystem": _build_filesystem,
        # Shell command execution (paf-owned harness copy) rooted at the project root.
        "Shell": _build_shell,
        # Safe git inspection + checkpoint tools rooted at the repo root.
        "Git": _build_git,
        # Codebase-map: navigation-first codebase browser rooted at the repo root
        # (overview/lookup_file/read_block/search_codebase + run_codebase_map).
        "CodebaseMap": _build_codebase_map,
        # Vault: navigation-first note ladder over a markdown vault (Tier A read
        # tools) — vault_overview/lookup_note/read_note + note_find/note_search.
        "Vault": _build_vault,
        # Tavily-backed web search + page extraction tools.
        "Web": _build_web,
        # Vision-model description of captured screenshots (describe_screenshot).
        "Visual": _build_visual,
        # Mobile-first headless browser automation (paf-owned copy, mobile default).
        "Browser": _build_browser,
        # Skill library — list/load on demand + deterministic pointer trigger.
        "Skills": _build_skills,
        # Framework-level tool-return reduction (BUILTIN, paf-supplied config).
        "ToolOutputLimits": _build_tool_output_limits,
        # Observational prompt-cache-collapse monitor (BUILTIN, paf-supplied config).
        "WarnOnCacheBusts": _build_warn_on_cache_busts,
        # Sub-agent delegation (BUILTIN, not a paf port) — explicit child leaf agents.
        "Subagents": _build_subagents,
        # Model-owned task plan for long turns (BUILTIN, not a paf port).
        "Planning": _build_planning,
    }
    store = _build_store()
    # Exactly ONE capability owns the composite recall ladder: when ``Memory`` is
    # allowlisted it owns it (durable-fact + past-turn + last-rung history, in
    # ``_DEFAULT_GUIDANCE``), so ``ConversationSearch``'s instruction scrap is
    # suppressed so the model sees one ladder, not two. This gate lives HERE
    # (build time) because capabilities can't see each other;
    # ``add_instructions=False`` only toggles instruction TEXT, never the tool
    # registration (search_conversation_history is still attached).
    has_memory = "Memory" in profile.capability_names
    capabilities = []
    for name, builder in builders.items():
        if name not in profile.capability_names:
            continue
        if name == "ConversationSearch" and has_memory:
            capabilities.append(_conversation_search_suppressed(store))
        elif name == "Web":
            capabilities.append(_build_web(store, web_search_domains))
        elif name == "Browser":
            capabilities.append(_build_browser(store, browser_config))
        elif name == "Vault":
            capabilities.append(_build_vault(store, vault_config))
        elif name == "FileSystem":
            capabilities.append(_build_filesystem(store, filesystem_config))
        elif name == "Shell":
            capabilities.append(_build_shell(store, shell_config))
        elif name == "Git":
            capabilities.append(_build_git(store, git_config))
        elif name == "CodebaseMap":
            capabilities.append(_build_codebase_map(store, codebase_map_config))
        else:
            capabilities.append(builder(store))
    if _COMPACTION_CAPABILITY in profile.capability_names:
        capabilities.append(build_compaction(compaction_config))
    return capabilities
