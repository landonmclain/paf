"""Tests for the profile↔catalog allowlist integrity and behavior."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    TieredCompaction,
    compact_now,
)
from pydantic_ai_harness.step_persistence import StepPersistence
from pydantic_ai_harness.warn_on_cache_busts import WarnOnCacheBusts

from paf.capabilities import ASSISTANT_PROFILE, AgentProfile
from paf.capabilities._window import TurnBoundSlidingWindow
from paf.capabilities.browser import Browser
from paf.capabilities.browser._toolset import (
    EgressPolicy,
    EgressRequest,
    is_always_blocked_address,
    is_blocked_address,
)
from paf.capabilities.compaction import (
    build_boundary_compaction,
    build_boundary_compaction_for_agent,
    build_compaction,
)
from paf.capabilities.profiles import (
    LONGMEMEVAL_PROFILE,
    PROFILES,
)
from paf.capabilities.registry import CAPABILITY_TYPES
from paf.capabilities.subagents import Subagents
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

    from paf.capabilities import AgentProfile
    from paf.capabilities.build import build_runtime_capabilities

    profile = AgentProfile(name="warn-on", capability_names=("WarnOnCacheBusts",))
    built = build_runtime_capabilities(profile)
    monitor = next((c for c in built if type(c).__name__ == "WarnOnCacheBusts"), None)
    assert monitor is not None
    assert isinstance(monitor, WarnOnCacheBusts)
    assert monitor.min_missed_ratio == 0.5
    assert monitor.min_missed_tokens == 1024


def test_cache_bust_warning_flips_active_span_to_error() -> None:
    """A CacheBustWarning must surface as an ERROR on the active OTel span.

    ``WarnOnCacheBusts`` emits ``CacheBustWarning`` (a UserWarning) via stdlib
    ``warnings``, which the runtime suppresses by default. ``_install_cache_bust_error_hook``
    wraps ``warnings.showwarning`` so a bust flips the current span to ERROR +
    records it — so Langfuse shows it red — and never aborts the turn.
    """
    import warnings

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from pydantic_ai_harness.warn_on_cache_busts import CacheBustWarning

    from paf.observability import instrumentation as inst

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Fresh process guard: the hook is idempotent, but force-reinstall so the
    # test is self-contained regardless of call order.
    inst._cache_bust_hook_installed = False
    inst._install_cache_bust_error_hook()

    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("model-req") as _:
        warnings.warn(  # noqa: B028 - deliberate: firing the signal under test
            CacheBustWarning(
                "cache hit collapsed at model request 3",
                reason="unexpected",
                established_tokens=8000,
                cache_read_tokens=500,
                missed_tokens=7500,
            )
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code.name == "ERROR", spans[0].status
    events = [e for e in spans[0].events]
    assert any(e.name == "exception" for e in events)
    # The turn continues: the `with` block above completed without raising.


@pytest.mark.live
def test_subagents_cataloged_and_wired() -> None:
    """Subagents is the paf-owned capability and resolves on a profile with its roster.

    NOT spec-serializable (holds live `Agent` instances; get_serialization_name ->
    None), so it must be built in Python (`_build_subagents`) from the leaf-spec
    roster. Assert the build attaches a `Subagents` whose roster is the demo
    `writer` leaf and whose toolset exposes the delegation tools.
    """
    assert CAPABILITY_TYPES["Subagents"] is Subagents
    assert Subagents.get_serialization_name() is None

    from paf.capabilities import AgentProfile
    from paf.capabilities.build import build_runtime_capabilities

    profile = AgentProfile(name="subagents-on", capability_names=("Subagents",))
    built = build_runtime_capabilities(profile)
    subagents = next((c for c in built if isinstance(c, Subagents)), None)
    assert subagents is not None
    assert [leaf.name for leaf in subagents.leaves] == ["writer"]
    toolset = subagents.get_toolset()
    assert toolset is not None
    assert {"delegate_task", "check_task"} <= set(toolset.tools)


def test_visual_cataloged_and_wired() -> None:
    """Visual is a cataloged paf-native capability and builds when allowlisted.

    Same icon-capability shape as Web/FileSystem/Shell/Git: cataloged (so the
    allowlist + schema resolve) and built in Python (`_build_visual`). It is a
    contained image-in/text-out side-call, NOT a sub-agent -- a single
    `describe_screenshot` tool (ask the vision model a targeted question about
    a captured screenshot ref), no child Agent, no delegation, no inherited
    tools. The tool must degrade to a clear message on an unknown ref (not abort
    the run).
    """
    assert CAPABILITY_TYPES["Visual"] is Visual

    from paf.capabilities import AgentProfile
    from paf.capabilities.build import build_runtime_capabilities

    profile = AgentProfile(name="visual-on", capability_names=("Visual",))
    built = build_runtime_capabilities(profile)
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
    """Browser is a cataloged paf-owned capability, mobile by default.

    It replaces the harness `PlaywrightBrowser` (mobile is the only layout that
    matters). Verify the catalog + profile wiring, that
    `build_runtime_capabilities` constructs a real `Browser`, that its toolset
    exposes the new `set_viewport` tool alongside the 18 standard browser tools,
    and that the default session is a 390×844 touch mobile profile.
    """
    assert CAPABILITY_TYPES["Browser"] is Browser
    assert "PlaywrightBrowser" not in CAPABILITY_TYPES

    from paf.capabilities import AgentProfile
    from paf.capabilities.build import build_runtime_capabilities

    profile = AgentProfile(name="browser-on", capability_names=("Browser",))
    built = build_runtime_capabilities(profile)
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


def test_browser_spec_config_wins_over_capability_default() -> None:
    """A spec `browser.allowed_domains` overrides the capability's default.

    Verifies the `browser:` block plumbing: `build_runtime_capabilities` reads
    the spec's browser_config dict and `_build_browser` honors its
    `allowed_domains` (spec wins). Without a config the capability's own
    loopback-only default is used.
    """
    from paf.capabilities import AgentProfile
    from paf.capabilities.build import build_runtime_capabilities

    profile = AgentProfile(name="browser-on", capability_names=("Browser",))

    # Spec wins: a declared list replaces the capability default entirely.
    built = build_runtime_capabilities(
        profile, None, None, {"allowed_domains": ["example.com"]}
    )
    browser = next((c for c in built if isinstance(c, Browser)), None)
    assert browser is not None
    assert browser.allowed_domains == ["example.com"]

    # No browser_config -> the capability's loopback-only default.
    built_default = build_runtime_capabilities(profile)
    browser_default = next((c for c in built_default if isinstance(c, Browser)), None)
    assert browser_default is not None
    assert browser_default.allowed_domains == [
        "localhost",
        "127.0.0.1",
        "::1",
    ]


def test_browser_private_guard_default_and_override() -> None:
    """`block_private_addresses` stays true (secure) unless the spec opts out.

    Build-level check that the browser's policy reflects the safe default and
    the explicit override. The real enforcement is exercised by the
    EgressPolicy tests below; here we only assert the capability is wired to
    the right flag.
    """
    from paf.capabilities import AgentProfile
    from paf.capabilities.build import build_runtime_capabilities

    profile = AgentProfile(name="browser-on", capability_names=("Browser",))

    # Secure default: no browser_config -> guard stays ON.
    b_default = next(
        c for c in build_runtime_capabilities(profile) if isinstance(c, Browser)
    )
    assert b_default.block_private_addresses is True

    # Explicit override -> guard turns OFF (reaches loopback/LAN dev hosts).
    b_open = next(
        c
        for c in build_runtime_capabilities(
            profile, None, None, {"block_private_addresses": False}
        )
        if isinstance(c, Browser)
    )
    assert b_open.block_private_addresses is False

    # Explicit true is honored but is the same as the default.
    b_strict = next(
        c
        for c in build_runtime_capabilities(
            profile, None, None, {"block_private_addresses": True}
        )
        if isinstance(c, Browser)
    )
    assert b_strict.block_private_addresses is True


def test_visual_hook_swaps_capture_pixels_for_stub() -> None:
    """When a vision model is configured, a screenshot BinaryContent becomes a text stub.

    A text-only main model must never receive raw pixels (it 404s on image
    input). The hook stashes the bytes and replaces the BinaryContent with a
    `[screenshot captured: shot_...]` stub naming the ref, so the model can call
    `describe_screenshot` with a targeted question. The hook is keyed on the
    capability's `model` (from the spec `visual:` block), not the environment.
    """
    import asyncio

    from paf.capabilities.visual._capability import Visual
    from paf.capabilities.visual._store import _store

    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    vis_on = Visual(model="openrouter:xiaomi/mimo-v2.5")
    vis_off = Visual()

    def _ctx(messages):
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
        # Vision model configured -> pixels stashed, stub left behind, ref resolvable.
        ctx = _ctx([ModelRequest(parts=[tool_return])])
        out = await vis_on.before_model_request(None, ctx)
        content = out.messages[0].parts[0].content
        assert not any(isinstance(p, BinaryContent) for p in content)
        stub = next(
            (
                p
                for p in content
                if isinstance(p, TextContent) and "screenshot captured" in p.content
            ),
            None,
        )
        assert stub is not None
        ref = stub.content.split("shot_")[1].split("]")[0].strip()
        assert _store.get(f"shot_{ref}") == png

        # No vision model -> passthrough (image preserved for a multimodal main model).
        ctx2 = _ctx([ModelRequest(parts=[tool_return])])
        out2 = await vis_off.before_model_request(None, ctx2)
        content2 = out2.messages[0].parts[0].content
        assert any(isinstance(p, BinaryContent) for p in content2)

        # A screenshot ALSO surfaces as a UserPromptPart with multimodal content;
        # the hook must strip it there too (the run-verified root cause of the
        # original 404 -- deepseek still saw the image in the UserPromptPart).
        ascii_png = b"\x89PNG\r\n\x1a\n" + b"\xaa" * 16
        user_part = UserPromptPart(
            content=[BinaryContent(data=ascii_png, media_type="image/png")]
        )
        ctx3 = _ctx([ModelRequest(parts=[user_part])])
        out3 = await vis_on.before_model_request(None, ctx3)
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


def test_keyword_curator_is_cataloged_and_allowlisted_where_promoted() -> None:
    """Per-turn relevance hints: tried in demo, promoted to the assistant profile."""
    from paf.capabilities.keyword_curator import KeywordCurator
    from paf.capabilities.profiles import DEMO_PROFILE

    assert CAPABILITY_TYPES["KeywordCurator"] is KeywordCurator
    assert "KeywordCurator" in DEMO_PROFILE.capability_names
    assert "KeywordCurator" in ASSISTANT_PROFILE.capability_names


def test_profiles_are_the_complete_intentional_set() -> None:
    """Known profiles: assistant (clean prod), demo (the everything sandbox),
    longmemeval (the memory benchmark arm), and secure (the locked-down
    high-stakes agent).

    ``demo`` is the surface where new/experimental capabilities are tried before
    being promoted to the production ``assistant`` spec. ``longmemeval`` is the memory benchmark arm (StepPersistence
    + ConversationSearch + Memory + Compaction). This is the complete,
    intentional set — no further blank profile scaffolding.
    """
    assert set(PROFILES) == {
        "assistant",
        "demo",
        "longmemeval",
        "secure",
    }
    assert PROFILES["assistant"] is ASSISTANT_PROFILE
    assert PROFILES["longmemeval"] is LONGMEMEVAL_PROFILE


def test_compaction_builds_real_strategy_objects() -> None:
    """TieredCompaction must hold real strategy objects — the exact regression
    that bit us when tiers were YAML dicts (AttributeError 'model')."""
    compaction = build_compaction()
    assert isinstance(compaction, TieredCompaction)

    # `CompactionStrategy` is a bare Protocol (not runtime_checkable), so we can't
    # isinstance against it. Assert each tier is a real concrete strategy (a class,
    # not a raw dict) drawn from the no-summary ladder's known types. The sliding
    # window is always the turn-bounded trim-only `TurnBoundSlidingWindow` (no
    # consolidate toggle).
    strategy_types = (
        ClampOversizedMessages,
        ClearToolResults,
        TurnBoundSlidingWindow,
    )
    assert len(compaction.tiers) == 3
    assert all(type(tier) in strategy_types for tier in compaction.tiers)


def test_build_compaction_window_is_always_turn_bound_trim_only() -> None:
    """`build_compaction` always uses the turn-bounded trim-only window.

    Regression: there is NO `consolidate` toggle.
    Eviction is always trim-only `TurnBoundSlidingWindow` (never evicts the
    active user turn, never calls memory) — the background watermark worker is
    the sole memory writer. This pins that the two concerns are structurally
    disjoint: the background trigger is gated by `memory.consolidate_every`
    (per-agent spec), never by a `compact.consolidate` flag or a session flag.
    """
    from paf.capabilities.compaction import CompactionConfig

    # Both an empty config and a config without any consolidate key yield the
    # same turn-bounded window.
    for cfg in (None, CompactionConfig(), CompactionConfig.from_mapping({})):
        comp = build_compaction(cfg)
        assert type(comp.tiers[-1]) is TurnBoundSlidingWindow

    # A stale `compact: {consolidate: false}` in a spec is a no-op (unknown key
    # dropped by from_mapping) and does not change the window type.
    from_spec = build_compaction(CompactionConfig.from_mapping({"consolidate": False}))
    assert type(from_spec.tiers[-1]) is TurnBoundSlidingWindow


def test_build_boundary_compaction_targets_turn_boundary_target() -> None:
    """The Tier A boundary strategy trims to `turn_boundary_target`, not the
    emergency `target_tokens`. When unset it falls back to `target_tokens`."""
    from paf.capabilities.compaction import CompactionConfig

    # Explicit turn_boundary_target wins over target_tokens.
    cfg = CompactionConfig(target_tokens=300000, turn_boundary_target=150000)
    boundary = build_boundary_compaction(cfg)
    assert isinstance(boundary, TieredCompaction)
    assert boundary.target_tokens == 150000

    # Unset turn_boundary_target -> falls back to target_tokens.
    fallback = build_boundary_compaction(CompactionConfig(target_tokens=300000))
    assert fallback.target_tokens == 300000

    # Neither set -> the ladder default (4000), same as build_compaction's bare default.
    default = build_boundary_compaction()
    assert default.target_tokens == 4000

    # Boundary uses the same turn-bounded trim-only window, never evicts active turn.
    assert type(boundary.tiers[-1]) is TurnBoundSlidingWindow


def test_build_boundary_compaction_for_agent_obeys_turn_boundary_flag() -> None:
    """`build_boundary_compaction_for_agent` returns a strategy when
    `turn_boundary` is absent/true, and None when explicitly false (control arm)."""
    from paf.capabilities.compaction import CompactionConfig

    # ON (absent / true) -> strategy built.
    on = build_boundary_compaction_for_agent("demo_assistant")
    assert isinstance(on, TieredCompaction)

    # OFF via an explicit spec override -> None (control arm, today's behavior).
    off_cfg = CompactionConfig(turn_boundary=False)
    assert off_cfg.turn_boundary is False


@pytest.mark.live
def test_turn_boundary_compact_now_trims_large_history_noop_small() -> None:
    """Driving the Tier A strategy via `compact_now` at the turn boundary trims a
    large accumulated prior-turn history (well over `turn_boundary_target`) and is a
    no-op on a small history (zero overhead on short sessions)."""
    boundary = build_boundary_compaction_for_agent("demo_assistant")
    model = "openrouter:deepseek/deepseek-flash-v4"

    # A large cross-turn history, well over the 150k boundary target.
    large = []
    for i in range(60):
        large.append(
            ModelRequest(parts=[UserPromptPart(content=f"user {i} " + "x" * 12000)])
        )
        large.append(
            ModelResponse(parts=[TextPart(content=f"reply {i} " + "y" * 10000)])
        )

    def _run() -> None:
        compacted = asyncio.run(
            compact_now(boundary, list(large), model=model, deps=None)
        )
        assert len(compacted) < len(large)
        assert compacted != large

        # A small history is returned unchanged — the boundary trim must not burn work
        # (or rewrite history / bust the cache) on ordinary small sessions.
        small = [
            ModelRequest(parts=[UserPromptPart(content="hi")]),
            ModelResponse(parts=[TextPart(content="hello")]),
        ]
        out = asyncio.run(compact_now(boundary, list(small), model=model, deps=None))
        assert out == small

    _run()


# --- EgressPolicy: secure default, explicit override, metadata firebreak ----


def _policy(*, block_private: bool, allowed: list[str] | None) -> EgressPolicy:
    """A browser egress policy with the given guard + allowlist."""
    return EgressPolicy(allowed_domains=allowed, block_private_addresses=block_private)


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
    assert (
        _refuse(p_meta, "http://metadata.google.internal/computeMetadata/v1/")
        is not None
    )


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
