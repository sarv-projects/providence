"""
Mode system — loads mode configurations and quality dials from config/modes.yaml.

Each mode defines budgets (tokens, cost, time, tool_calls) that are enforced
during research runs. Quality dials overlay mode settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml  # type: ignore


@dataclass
class ModeBudgets:
    """Budget limits for a research mode."""
    max_tokens: int = 100000
    max_cost_usd: float = 0.50
    max_time_s: int = 600
    max_tool_calls: int = 20
    max_iterations: int = 6


@dataclass
class QualityDial:
    """Quality dial settings that overlay mode budgets."""
    description: str = "Default"
    max_tokens_per_call: int = 8000
    max_search_results: int = 10
    max_extract_pages: int = 5
    thinker_enabled: bool = False
    triangulation_enabled: bool = False
    factoid_enabled: bool = False


@dataclass
class Mode:
    """A configured research mode."""
    name: str = "standard"
    description: str = ""
    budgets: ModeBudgets = field(default_factory=ModeBudgets)
    quality_dial: str = "balanced"
    quality: QualityDial = field(default_factory=QualityDial)
    vault_rag: bool = True
    allow_tools: bool = True
    escalate_to_research: bool = False
    recency_bias: bool = False
    academic_bias: bool = False
    structured_output: bool = False
    requires_temporal: bool = False


@dataclass
class ModeRegistry:
    """Registry of all available modes and quality dials."""
    modes: dict[str, Mode] = field(default_factory=dict)
    quality_dials: dict[str, QualityDial] = field(default_factory=dict)
    default_mode: str = "standard"


def _resolve_config_path() -> Path:
    """Find modes.yaml."""
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "config" / "modes.yaml"


def _load_raw(path: Path) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_modes(config_path: Optional[str] = None) -> ModeRegistry:
    """Load mode configurations from YAML.

    Returns a ModeRegistry with all modes and quality dials. If the config
    file is missing, returns hardcoded defaults (standard/balanced only).
    """
    path = Path(config_path) if config_path else _resolve_config_path()
    raw = _load_raw(path)

    registry = ModeRegistry()

    # Parse quality dials
    dials_raw = raw.get("quality_dials", {})
    for name, cfg in dials_raw.items():
        if not isinstance(cfg, dict):
            continue
        registry.quality_dials[name] = QualityDial(
            description=cfg.get("description", name),
            max_tokens_per_call=cfg.get("max_tokens_per_call", 8000),
            max_search_results=cfg.get("max_search_results", 10),
            max_extract_pages=cfg.get("max_extract_pages", 5),
            thinker_enabled=cfg.get("thinker_enabled", False),
            triangulation_enabled=cfg.get("triangulation_enabled", False),
            factoid_enabled=cfg.get("factoid_enabled", False),
        )

    # Parse modes
    modes_raw = raw.get("modes", {})
    for name, cfg in modes_raw.items():
        if not isinstance(cfg, dict):
            continue
        budgets_raw = cfg.get("budgets", {})
        dial_name = cfg.get("quality_dial", "balanced")
        dial = registry.quality_dials.get(dial_name, QualityDial())

        registry.modes[name] = Mode(
            name=name,
            description=cfg.get("description", ""),
            budgets=ModeBudgets(
                max_tokens=budgets_raw.get("max_tokens", 100000),
                max_cost_usd=budgets_raw.get("max_cost_usd", 0.50),
                max_time_s=budgets_raw.get("max_time_s", 600),
                max_tool_calls=budgets_raw.get("max_tool_calls", 20),
                max_iterations=budgets_raw.get("max_iterations", 6),
            ),
            quality_dial=dial_name,
            quality=dial,
            vault_rag=cfg.get("vault_rag", True),
            allow_tools=cfg.get("allow_tools", True),
            escalate_to_research=cfg.get("escalate_to_research", False),
            recency_bias=cfg.get("recency_bias", False),
            academic_bias=cfg.get("academic_bias", False),
            structured_output=cfg.get("structured_output", False),
            requires_temporal=cfg.get("requires_temporal", False),
        )

    # Ensure defaults
    if "standard" not in registry.modes:
        registry.modes["standard"] = Mode(
            name="standard",
            description="Default research",
            quality_dial="balanced",
            quality=registry.quality_dials.get("balanced", QualityDial()),
        )

    if "balanced" not in registry.quality_dials:
        registry.quality_dials["balanced"] = QualityDial()

    return registry


def get_mode(registry: ModeRegistry, name: str) -> Mode:
    """Get a mode by name, falling back to standard."""
    return registry.modes.get(name, registry.modes.get("standard", Mode()))


# ── Research lenses (combinable toggles) ─────────────────────────────
# The UI offers Chat / Research, with Research depth = standard | deep and
# three orthogonal toggles: recency, academic, compare. Each lens maps to
# the mode_flags the agents already consume (researcher query rewriting,
# planner outline hints, synthesizer structure).
LENSES = ("recency", "academic", "compare")

# Legacy mode names kept for CLI/API back-compat: each maps to a base depth
# mode plus the lens flag it used to imply. Academic maps to deep (both use
# the accurate dial with thinker enabled); recency/compare map to standard
# (both used the balanced dial).
LEGACY_LENS_MODES: dict[str, tuple[str, dict[str, bool]]] = {
    "recency": ("standard", {"recency": True}),
    "academic": ("deep", {"academic": True}),
    "compare": ("standard", {"compare": True}),
}


def normalize_lenses(lenses: dict | None) -> dict[str, bool]:
    """Coerce a lens mapping to {recency, academic, compare} bools."""
    lenses = lenses or {}
    return {name: bool(lenses.get(name, False)) for name in LENSES}


def resolve_mode(name: str) -> tuple[str, dict[str, bool]]:
    """Split a mode name into (base_depth_mode, implied_lenses).

    Legacy lens-modes map to standard + their lens; everything else passes
    through with no implied lenses. Unknown names still fall back to
    standard downstream via get_mode().
    """
    base, implied = LEGACY_LENS_MODES.get(name or "", (name or "standard", {}))
    return base, normalize_lenses(implied)


def merge_lenses(*lens_dicts: dict | None) -> dict[str, bool]:
    """OR-merge lens mappings (request > settings > legacy-implied)."""
    merged = normalize_lenses(None)
    for d in lens_dicts:
        for name in LENSES:
            if d and d.get(name):
                merged[name] = True
    return merged
