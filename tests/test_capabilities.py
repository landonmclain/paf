"""Tests for the profile↔catalog allowlist integrity and behavior."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    TextContent,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    TieredCompaction,
)
from pydantic_ai_harness.step_persistence import StepPersistence
from pydantic_ai_harness.subagents import SubAgent, SubAgents
from pydantic_ai_harness.warn_on_cache_busts import WarnOnCacheBusts

from paf.capabilities import ASSISTANT_PROFILE, AgentProfile
from paf.capabilities.browser import Browser
from paf.capabilities.browser._toolset import (
    EgressPolicy,
    EgressRequest,
    is_always_blocked_address,
    is_blocked_address,
)
from paf.capabilities.compaction import build_compaction
from paf.capabilities.memory._window import (
    MemoryConsolidationWindow,
    TurnBoundSlidingWindow,
)
from paf.capabilities.profiles import CONTEXTBENCH_PROFILE, DEMO_PROFILE, PROFILES
from paf.capabilities.registry import CAPABILITY_TYPES
from paf.capabilities.visual import Visual


def test_assistant_profile_names_all_exist_in_catalog() -> None:
    """Every allowlisted name must resolve in the catalog (no dead refs)."""
    missing = [
        name
        for name in ASSISTANT_PROFILE.capability_names
        if name not in CAPABILITY_TYPES
    ]
    assert missing == []


def test_catalog_covers_everything_build_uses() -> None:
    """The catalog is complete enough to resolve the built capabilities.

    build_compaction emits a TieredCompaction whose tier types are all in the
    catalog, and the spec-declared StepPersistence is cataloged too.
    """
    compaction = build_compaction()
    tier_names = {type(tier).__name__ for tier in compaction.tiers}
    assert tier_names <= set(CAPABILITY_TYPES)
    assert "StepPersistence" in CAPABILITY_TYPES
    assert CAPABILITY_TYPES["StepPersistence"] is StepPersistence
    # "Compaction" is the profile-gated name that resolves the no-summary ladder.
    assert "Compaction" in CAPABILITY_TYPES
    assert CAPABILITY_TYPES["Compaction"] is TieredCompaction


def test_capability_types_resolves_and_raises_on_missing() -> None:
    """A valid name resolves to its type; an unknown one fails loudly,
    naming the offending entry (profiles.py contract: fail at build, not run)."""
    profile = AgentProfile(name="probe", capability_names=("StepPersistence",))
    types = profile.capability_types()
    assert types == (StepPersistence,)

    bad = AgentProfile(name="probe", capability_names=("NoSuchCapability",))
    with pytest.raises(KeyError) as excinfo:
        bad.capability_types()
    assert "NoSuchCapability" in str(excinfo.value)


def test_warn_on_cache_busts_cataloged_and_wired() -> None:
    """WarnOnCacheBusts is a cataloged BUILTIN and resolves on the demo profile.

    Same drop-in pattern as ToolOutputLimits: cataloged (so the allowlist + schema
    resolve) and built in Python (so the profile gate controls it with real config
    objects). Observer-only, conservative defaults — no tools/instructions/settings.
    """
    assert CAPABILITY_TYPES["WarnOnCacheBusts"] is WarnOnCacheBusts

    from paf.capabilities.build import build_runtime_capabilities

    built = build_runtime_capabilities(DEMO_PROFILE)
    monitor = next(
        (c for c in built if type(c).__name__ == "WarnOnCacheBusts"), None
    )
    assert monitor is not None
    assert isinstance(monitor, WarnOnCacheBusts)
    assert monitor.collapse_ratio == 0.5
    assert monitor.min_prefix_tokens == 1024
    assert monitor.cache_ttl_seconds == 300.0


def test_subagents_cataloged_and_wired() -> None:
    """Subagents is a cataloged BUILTIN and resolves on the demo profile with child agents.

    Same BUILTIN drop-in pattern as WarnOnCacheBusts, but NOT spec-serializable
    (holds live `Agent` instances; get_serialization_name -> None), so it must be
    built in Python (`_build_subagents`) with explicit child entries. Assert the
    demo build attaches a SubAgents capability whose toolset exposes exactly the
    two named demo leaf agents.
    """
    assert CAPABILITY_TYPES["Subagents"] is SubAgents

    from paf.capabilities.build import build_runtime_capabilities

    built = build_runtime_capabilities(DEMO_PROFILE)
    subagents = next((c for c in built if type(c).__name__ == "SubAgents"), None)
    assert subagents is not None
    assert isinstance(subagents, SubAgents)
    # _by_name is built in __post_init__ from the two explicit SubAgent entries.
    assert set(subagents._by_name) == {"researcher", "writer"}
    assert all(isinstance(sa, SubAgent) for sa in subagents._by_name.values())
    # A live toolset is produced so the delegate_task tool is actually exposed.
    toolset = subagents.get_toolset()
    assert toolset is not None


def test_visual_cataloged_and_wired() -> None:
    """Visual is a cataloged paf-native capability wired on the demo profile only.

    Same icon-capability shape as Web/FileSystem/Shell/Git: cataloged (so the
    allowlist + schema resolve) and built in Python (`_build_visual`) on the
    demo profile, deliberately ABSENT from the clean assistant profile. It is a
    contained image-in/text-out side-call, NOT a sub-agent -- a single
    `describe_screenshot` tool (ask the vision model a targeted question about
    a captured screenshot ref), no child Agent, no delegation, no inherited
    tools. The tool must degrade to a clear message on an unknown ref (not abort
    the run).
    """
    assert CAPABILITY_TYPES["Visual"] is Visual
    assert "Visual" in DEMO_PROFILE.capability_names
    assert "Visual" not in ASSISTANT_PROFILE.capability_names

    from paf.capabilities.build import build_runtime_capabilities

    built = build_runtime_capabilities(DEMO_PROFILE)
    vis = next((c for c in built if isinstance(c, Visual)), None)
    assert vis is not None

    # Exactly one tool, named describe_screenshot.
    toolset = vis.get_toolset()
    assert toolset is not None
    assert set(toolset.tools) == {"describe_screenshot"}

    # An unknown ref degrades to a clear message, not a crash.
    async def _run() -> str:
        return await toolset.describe_screenshot("shot_deadbeef", "what is this?")

    import asyncio

    msg = asyncio.run(_run())
    assert "no screenshot found for ref" in msg


def test_browser_cataloged_and_wired() -> None:
    """Browser is a cataloged paf-owned capability on the demo profile only, mobile by default.

    It replaces the harness `PlaywrightBrowser` on the demo agent (mobile is the
    only layout that matters). Verify the catalog + profile wiring, that
    `build_runtime_capabilities` constructs a real `Browser`, that its toolset
    exposes the new `set_viewport` tool alongside the 18 standard browser tools,
    and that the default session is a 390×844 touch mobile profile.
    """
    assert CAPABILITY_TYPES["Browser"] is Browser
    assert "Browser" in DEMO_PROFILE.capability_names
    assert "Browser" not in ASSISTANT_PROFILE.capability_names
    assert "PlaywrightBrowser" not in CAPABILITY_TYPES

    from paf.capabilities.build import build_runtime_capabilities

    built = build_runtime_capabilities(DEMO_PROFILE)
    browser = next((c for c in built if isinstance(c, Browser)), None)
    assert browser is not None

    # Mobile-first default: 390×844 touch viewport, retina scale 3.
    assert browser.viewport == (390, 844)
    assert browser.mobile is True
    assert browser.is_mobile is True
    assert browser.has_touch is True
    assert browser.device_scale_factor == 3

    # The toolset exposes the 18 standard browser tools plus set_viewport, and
    # the session carries the mobile defaults.
    toolset = browser.get_toolset()
    assert toolset is not None
    assert "set_viewport" in toolset.tools
    assert len(toolset.tools) == 19
    assert browser._session._mobile is True
    assert browser._session._viewport == (390, 844)


def test_browser_spec_config_wins_over_env_default() -> None:
    """A spec `browser.allowed_domains` overrides the env-driven default.

    Verifies the `browser:` block plumbing: `build_runtime_capabilities` reads
    the spec's browser_config dict and `_build_browser` honors its
    `allowed_domains` (spec wins). Without a config the env
    `settings.playwright_allowed_domains` default is used.
    """
    from paf.capabilities.build import build_runtime_capabilities

    # Spec wins: a declared list replaces the env default entirely.
    built = build_runtime_capabilities(
        DEMO_PROFILE, None, None, {"allowed_domains": ["example.com"]}
    )
    browser = next((c for c in built if isinstance(c, Browser)), None)
    assert browser is not None
    assert browser.allowed_domains == ["example.com"]

    # No browser_config -> env-driven default (fall through unchanged).
    built_default = build_runtime_capabilities(DEMO_PROFILE)
    browser_default = next(
        (c for c in built_default if isinstance(c, Browser)), None
    )
    assert browser_default is not None
    assert browser_default.allowed_domains == [
        "localhost",
        "127.0.0.1",
        "::1",
        "staging.example.com",
    ]


def test_browser_private_guard_default_and_override() -> None:
    """`block_private_addresses` stays true (secure) unless the spec opts out.

    Build-level check that the browser's policy reflects the safe default and
    the explicit override. The real enforcement is exercised by the
    EgressPolicy tests below; here we only assert the capability is wired to
    the right flag.
    """
    from paf.capabilities.build import build_runtime_capabilities

    # Secure default: no browser_config -> guard stays ON.
    b_default = next(
        c
        for c in build_runtime_capabilities(DEMO_PROFILE)
        if isinstance(c, Browser)
    )
    assert b_default.block_private_addresses is True

    # Explicit override -> guard turns OFF (reaches loopback/LAN dev hosts).
    b_open = next(
        c
        for c in build_runtime_capabilities(
            DEMO_PROFILE, None, None, {"block_private_addresses": False}
        )
        if isinstance(c, Browser)
    )
    assert b_open.block_private_addresses is False

    # Explicit true is honored but is the same as the default.
    b_strict = next(
        c
        for c in build_runtime_capabilities(
            DEMO_PROFILE, None, None, {"block_private_addresses": True}
        )
        if isinstance(c, Browser)
    )
    assert b_strict.block_private_addresses is True


def test_visual_hook_swaps_capture_pixels_for_stub() -> None:
    """When VISION_MODEL is set, a screenshot BinaryContent becomes a text stub.

    A text-only main model must never receive raw pixels (it 404s on image
    input). The hook stashes the bytes and replaces the BinaryContent with a
    `[screenshot captured: shot_...]` stub naming the ref, so the model can call
    `describe_screenshot` with a targeted question.
    """
    import asyncio
    import os

    from paf.capabilities.visual._capability import Visual
    from paf.capabilities.visual._store import _store

    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    vis = Visual()

    def _ctx(messages, *, vision):
        if vision:
            os.environ["VISION_MODEL"] = "openrouter:xiaomi/mimo-v2.5"
        else:
            os.environ.pop("VISION_MODEL", None)
        return ModelRequestContext(
            model="openrouter:deepseek/deepseek-v4-flash-0731",
            messages=messages,
            model_settings={},
            model_request_parameters={},
        )

    tool_return = ToolReturnPart(
        tool_name="screenshot",
        tool_call_id="call_1",
        content=[
            TextContent("Screenshot captured. URL: http://example.com"),
            BinaryContent(data=png, media_type="image/png"),
        ],
    )

    async def _run() -> None:
        # VISION_MODEL set -> pixels stashed, stub left behind, ref resolvable.
        ctx = _ctx([ModelRequest(parts=[tool_return])], vision=True)
        out = await vis.before_model_request(None, ctx)
        content = out.messages[0].parts[0].content
        assert not any(isinstance(p, BinaryContent) for p in content)
        stub = next(
            (p for p in content if isinstance(p, TextContent) and "screenshot captured" in p.content),
            None,
        )
        assert stub is not None
        ref = stub.content.split("shot_")[1].split("]")[0].strip()
        assert _store.get(f"shot_{ref}") == png

        # VISION_MODEL unset -> passthrough (image preserved for a multimodal main model).
        ctx2 = _ctx([ModelRequest(parts=[tool_return])], vision=False)
        out2 = await vis.before_model_request(None, ctx2)
        content2 = out2.messages[0].parts[0].content
        assert any(isinstance(p, BinaryContent) for p in content2)

        # A screenshot ALSO surfaces as a UserPromptPart with multimodal content;
        # the hook must strip it there too (the run-verified root cause of the
        # original 404 -- deepseek still saw the image in the UserPromptPart).
        ascii_png = b"\x89PNG\r\n\x1a\n" + b"\xaa" * 16
        user_part = UserPromptPart(content=[BinaryContent(data=ascii_png, media_type="image/png")])
        ctx3 = _ctx([ModelRequest(parts=[user_part])], vision=True)
        out3 = await vis.before_model_request(None, ctx3)
        new_parts = out3.messages[0].parts
        assert len(new_parts) == 1
        assert not any(
            isinstance(p, BinaryContent)
            for p in new_parts[0].content
            if isinstance(new_parts[0], UserPromptPart)
        )
        kinds = [type(p).__name__ for p in new_parts[0].content]
        assert all(k == "TextContent" for k in kinds)

    asyncio.run(_run())



def test_only_one_real_profile_for_m1() -> None:
    """Known profiles: assistant (clean prod), demo (the everything sandbox),
    contextbench (the benchmark-runner arm), and runtime2_test (the LEAN
    next-gen-runtime test helper).

    ``demo_assistant`` is the surface where we build out and tune the full stack
    (persistence, recall, memory, compaction) via the ``demo`` profile, which
    allowlists the same harness capabilities as the production assistant. Once the
    experiment proves out, demo's tuned values promote to the clean ``assistant``
    spec and the sandbox resets. ``contextbench`` is the deliberate benchmark arm
    (CodebaseMap + Shell + FileSystem + Git, NO Memory). ``runtime2_test`` is the
    deliberate dev/test arm (Shell only) for deterministically exercising the
    Session stream/event contract — a test profile, not a product surface. This
    is the complete, intentional set — no further blank profile scaffolding.
    """
    assert set(PROFILES) == {
        "assistant",
        "demo",
        "contextbench",
        "runtime2_test",
    }
    assert PROFILES["assistant"] is ASSISTANT_PROFILE
    assert PROFILES["contextbench"] is CONTEXTBENCH_PROFILE


def test_compaction_builds_real_strategy_objects() -> None:
    """TieredCompaction must hold real strategy objects — the exact regression
    that bit us when tiers were YAML dicts (AttributeError 'model')."""
    compaction = build_compaction()
    assert isinstance(compaction, TieredCompaction)

    # `CompactionStrategy` is a bare Protocol (not runtime_checkable), so we can't
    # isinstance against it. Assert each tier is a real concrete strategy (a class,
    # not a raw dict) drawn from the no-summary ladder's known types. The sliding
    # window is always the turn-bounded trim-only `TurnBoundSlidingWindow` (no
    # consolidate toggle, retired 2026-09-02).
    strategy_types = (
        ClampOversizedMessages,
        ClearToolResults,
        TurnBoundSlidingWindow,
    )
    assert len(compaction.tiers) == 3
    assert all(type(tier) in strategy_types for tier in compaction.tiers)


def test_build_compaction_window_is_always_turn_bound_trim_only() -> None:
    """`build_compaction` always uses the turn-bounded trim-only window.

    Regression for the 2026-09-02 decision: there is NO `consolidate` toggle.
    Eviction is always trim-only `TurnBoundSlidingWindow` (never evicts the
    active user turn, never calls memory) — the background watermark worker is
    the sole memory writer. This pins that the two concerns are structurally
    disjoint: the background trigger is gated separately (session
    `memory_consolidation` flag + `memory.consolidate_every`), never by a
    `compact.consolidate` flag.
    """
    from paf.capabilities.compaction import CompactionConfig

    # Both an empty config and a config without any consolidate key yield the
    # same turn-bounded window.
    for cfg in (None, CompactionConfig(), CompactionConfig.from_mapping({})):
        comp = build_compaction(cfg)
        assert type(comp.tiers[-1]) is TurnBoundSlidingWindow
        assert not isinstance(comp.tiers[-1], MemoryConsolidationWindow)

    # A stale `compact: {consolidate: false}` in a spec is a no-op (unknown key
    # dropped by from_mapping) and does not change the window type.
    from_spec = build_compaction(CompactionConfig.from_mapping({"consolidate": False}))
    assert type(from_spec.tiers[-1]) is TurnBoundSlidingWindow


# --- EgressPolicy: secure default, explicit override, metadata firebreak ----

def _policy(*, block_private: bool, allowed: list[str] | None) -> EgressPolicy:
    """A browser egress policy with the given guard + allowlist."""
    return EgressPolicy(
        allowed_domains=allowed, block_private_addresses=block_private
    )


def _req(url: str, kind: str = "navigation") -> EgressRequest:
    """A minimal egress request, like the session's pre-navigation check."""
    return EgressRequest(url=url, kind=kind)  # type: ignore[arg-type]


def _refuse(policy: EgressPolicy, url: str) -> str | None:
    """Whether the policy refuses `url` (kind navigation)."""
    return policy.refuse(_req(url))


def test_egress_default_blocks_private_and_metadata() -> None:
    """Secure default: loopback, RFC-1918, and link-local/metadata are refused —
    even when the allowlist names the loopback host. The guard outranks the
    allowlist, so listing `localhost` does NOT make it reachable on its own.
    """
    # The guard's address classifier sees all three classes.
    assert is_blocked_address("127.0.0.1")
    assert is_blocked_address("localhost")
    assert is_blocked_address("169.254.169.254")  # cloud metadata
    assert is_blocked_address("192.168.1.50")

    # refuse() refuses them under the default, even when allowlisted.
    p = _policy(block_private=True, allowed=["localhost", "127.0.0.1"])
    assert _refuse(p, "http://localhost:4321/") is not None
    assert _refuse(p, "http://127.0.0.1:4321/") is not None
    assert _refuse(p, "http://169.254.169.254/latest/meta-data/") is not None
    assert _refuse(p, "http://192.168.1.50/admin") is not None

    # A public host is allowed when named in the allowlist.
    p_public = _policy(block_private=True, allowed=["staging.example.com"])
    assert _refuse(p_public, "http://staging.example.com/") is None


def test_egress_override_reaches_loopback_but_not_metadata() -> None:
    """`block_private_addresses: false` (override) releases loopback/RFC-1918
    for allowlisted dev hosts — but the cloud-metadata endpoint is STILL refused
    (the firebreak can never be released by the override).
    """
    # Loopback and LAN are reachable once the override + allowlist both apply.
    p_open = _policy(
        block_private=False, allowed=["localhost", "127.0.0.1", "192.168.1.50"]
    )
    assert _refuse(p_open, "http://localhost:4321/") is None
    assert _refuse(p_open, "http://127.0.0.1:4321/") is None
    assert _refuse(p_open, "http://192.168.1.50/admin") is None

    # But metadata is always blocked — link-local, literal and aliases.
    assert is_always_blocked_address("169.254.169.254")
    assert is_always_blocked_address("169.254.0.1")
    assert is_always_blocked_address("metadata.google.internal")
    assert is_always_blocked_address("metadata.azure.internal")
    assert not is_always_blocked_address("localhost")  # override may release it

    # ...and the policy refuses it even with the override + allowlist naming it.
    p_meta = _policy(
        block_private=False,
        allowed=["localhost", "169.254.169.254", "metadata.google.internal"],
    )
    assert _refuse(p_meta, "http://169.254.169.254/latest/meta-data/iam/") is not None
    assert _refuse(p_meta, "http://metadata.google.internal/computeMetadata/v1/") is not None


def test_egress_metadata_always_blocked_even_when_allowlisted_default() -> None:
    """Even under the SECURE default, refusing the metadata endpoint does not
    depend on it being allowlisted — it is refused regardless (literal and the
    known aliases), as the firebreak that SSRF worst-case cannot clear.
    """
    assert is_always_blocked_address("169.254.169.254")
    assert is_always_blocked_address("metadata.ec2.internal")
    assert is_always_blocked_address("metadata")

    p = _policy(block_private=True, allowed=["169.254.169.254"])
    assert _refuse(p, "http://169.254.169.254/latest/meta-data/") is not None
