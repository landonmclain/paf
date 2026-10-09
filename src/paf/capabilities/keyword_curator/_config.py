"""Where the keyword store lives and what the curator reads, per agent.

One place, shared by the CLI, the hot-path capability, and the background
trigger, so they cannot disagree about which store or which sources an agent has.
The sources are derived from the agent's own spec (its ``skills:`` libraries and
``vault:`` root), so the curator indexes exactly what that agent can read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paf.config import DATA_DIR


def keyword_dir_for(agent_name: str | None) -> Path:
    """The per-agent keyword store: ``data/keywords/<agent>/``."""
    return DATA_DIR / "keywords" / (agent_name or "agent")


def sources_config_for(agent_name: str) -> dict[str, Any]:
    """``scan_sources`` kwargs for an agent, read from its spec.

    Skill libraries default to the paf-owned ``data/skills/`` when the spec has no
    ``skills:`` block; the vault is included only when the spec names a root.
    """
    from paf.specs import _spec_skills_config, _spec_vault_config, get_spec_path

    spec_path = get_spec_path(agent_name)
    skills = _spec_skills_config(spec_path) or {}
    vault = _spec_vault_config(spec_path) or {}
    return {
        "skill_libraries": skills.get("libraries") or [DATA_DIR / "skills"],
        "vault_root": vault.get("root"),
        "vault_allow": vault.get("allow") or [],
        "vault_deny": vault.get("deny") or [],
    }
