"""Visual capability spec-config tests (Phase 2).

The vision model comes from the spec's `visual:` block, not env. Covers the
three states the builder must distinguish:

- `visual:` block with a `model`  -> the Visual capability carries the model.
- no `visual:` block             -> the capability is inert (model None; images
                                    pass through to the main model untouched).
- `visual:` block without `model` -> build-time hard error (a declared vision
                                    route needs a model -- no silent default).
"""

from __future__ import annotations

import pytest

from paf.capabilities import AgentProfile
from paf.capabilities.build import build_runtime_capabilities
from paf.capabilities.visual import Visual
from paf.specs import _spec_visual_config


def _build_visual(visual_config=None) -> Visual:
    profile = AgentProfile(name="visual-on", capability_names=("Visual",))
    built = build_runtime_capabilities(profile, visual_config=visual_config)
    vis = next((c for c in built if isinstance(c, Visual)), None)
    assert vis is not None
    return vis


def test_visual_spec_model_drives_build() -> None:
    """A spec `visual.model` sets the capability's vision model."""
    vis = _build_visual({"model": "openrouter:acme/vision-1"})
    assert vis.model == "openrouter:acme/vision-1"


def test_visual_absent_block_is_inert() -> None:
    """No `visual:` block -> the capability builds inert (model None)."""
    vis = _build_visual(None)
    assert vis.model is None


def test_visual_present_without_model_hard_errors() -> None:
    """A `visual:` block without a model is a misconfiguration, not a default."""
    with pytest.raises(ValueError, match="visual.*model"):
        _build_visual({"model": None})


def test_spec_visual_config_distinguishes_states(tmp_path) -> None:
    """The accessor tells absent from present-without-model apart."""
    absent = tmp_path / "absent.yaml"
    absent.write_text("name: x\nmodel: openrouter:deepseek\n")
    assert _spec_visual_config(absent) is None

    with_model = tmp_path / "with.yaml"
    with_model.write_text("visual:\n  model: openrouter:acme/vision-1\n")
    assert _spec_visual_config(with_model) == {"model": "openrouter:acme/vision-1"}

    no_model = tmp_path / "no_model.yaml"
    no_model.write_text("visual:\n  color: red\n")
    assert _spec_visual_config(no_model) == {"model": None}

    with_settings = tmp_path / "settings.yaml"
    with_settings.write_text(
        "visual:\n  model: ollama:qwen3.5:4b\n"
        "  model_settings:\n    openai_reasoning_effort: none\n"
    )
    assert _spec_visual_config(with_settings) == {
        "model": "ollama:qwen3.5:4b",
        "model_settings": {"openai_reasoning_effort": "none"},
    }


def test_visual_model_settings_reach_the_toolset() -> None:
    """`visual.model_settings` is carried by the capability into its tool."""
    visual = _build_visual(
        {"model": "ollama:x", "model_settings": {"openai_reasoning_effort": "none"}}
    )
    assert visual.model_settings == {"openai_reasoning_effort": "none"}
    assert visual.get_toolset()._model_settings == {"openai_reasoning_effort": "none"}
