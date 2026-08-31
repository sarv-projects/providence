"""
Provider catalog — loads presets from config/providers.yaml and manages
provider slots for the LLM gateway.

Convention:
- Empty base_url → https://opencode.ai/zen/v1 (OpenCode Zen free)
- Empty api_key → no Authorization header (free tier)

Providers can be added/removed at runtime via the catalog API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


@dataclass
class ProviderSlot:
    """A configured LLM provider with models and auth."""
    name: str                                   # internal key: "opencode_free", "groq", etc.
    display_name: str                           # human-readable: "OpenCode Zen (Free)"
    base_url: str                               # empty → Zen free; otherwise full URL
    api_key: str = ""                           # empty → no auth
    protocol: str = "openai_chat"               # "openai_chat" | "anthropic_messages" | "cohere_v2_chat"
    models: list[str] = field(default_factory=list)
    is_default: bool = False
    env_key_name: str = ""                      # e.g. "GROQ_API_KEY"

    @property
    def effective_base_url(self) -> str:
        """Resolve empty base_url to Zen free endpoint."""
        if not self.base_url:
            return "https://opencode.ai/zen/v1"
        base = self.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        return base

    @property
    def has_auth(self) -> bool:
        return bool(self.api_key)


@dataclass
class TierRoute:
    """A route in a tier's failover chain."""
    provider_name: str
    model: str
    priority: int = 0


@dataclass
class TierConfig:
    """A tier (fast/strong/thinker) with ordered routes."""
    name: str
    routes: list[TierRoute] = field(default_factory=list)


@dataclass
class CatalogConfig:
    """Full catalog: providers + tiers."""
    providers: dict[str, ProviderSlot] = field(default_factory=dict)
    tiers: dict[str, TierConfig] = field(default_factory=dict)


def _resolve_config_path() -> Path:
    """Find providers.yaml — user override or bundled example."""
    project_root = Path(__file__).resolve().parent.parent.parent
    user_path = project_root / "config" / "providers.yaml"
    example_path = project_root / "config" / "providers.example.yaml"
    if user_path.exists():
        return user_path
    return example_path


def _load_raw(path: Path) -> dict:
    """Load the YAML config file."""
    if yaml is None:
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _resolve_api_keys(env_key: str) -> list[str]:
    """Look up ALL provider API keys from environment variables.

    Supports multi-key pools: GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ...
    Returns an empty list if no keys are configured (free tier).
    """
    if not env_key:
        return []
    keys = []
    val = os.getenv(env_key, "")
    if val:
        keys.append(val)
    for i in range(2, 10):
        v = os.getenv(f"{env_key}_{i}", "")
        if not v:
            break
        keys.append(v)
    return keys


def load_catalog(config_path: Optional[str] = None) -> CatalogConfig:
    """Load the provider catalog from YAML config.

    Args:
        config_path: Optional path to providers.yaml. Falls back to
                     config/providers.yaml → config/providers.example.yaml.

    Returns:
        CatalogConfig with all enabled providers and tier routes.
    """
    path = Path(config_path) if config_path else _resolve_config_path()
    raw = _load_raw(path)

    cat = CatalogConfig()

    # Parse providers
    providers_raw = raw.get("providers", {})
    for key, cfg in providers_raw.items():
        if not isinstance(cfg, dict):
            continue

        env_key = cfg.get("api_key_env", "")
        api_keys = _resolve_api_keys(env_key)

        # A provider is enabled IF:
        # - It has at least one API key, OR
        # - It's the Zen free default (empty URL = no auth needed)
        base_url = cfg.get("base_url", "")
        is_free = (not base_url)  # empty URL means Zen free

        if not api_keys and not is_free:
            continue  # skip providers that need a key but don't have one

        slot = ProviderSlot(
            name=key,
            display_name=cfg.get("name", key),
            base_url=base_url,
            api_key=api_keys[0] if api_keys else "",
            protocol=cfg.get("protocol", "openai_chat"),
            models=list(cfg.get("models", [])),
            is_default=cfg.get("default", False),
            env_key_name=env_key,
        )
        # Attach the full key pool for gateway use
        slot._all_keys = api_keys
        cat.providers[key] = slot

    # Parse tiers
    tiers_raw = raw.get("tiers", {})
    for tier_name, routes in tiers_raw.items():
        if not isinstance(routes, list):
            continue
        tier = TierConfig(name=tier_name)
        for i, route in enumerate(routes):
            if not isinstance(route, dict):
                continue
            provider_name = route.get("provider", "")
            if provider_name in cat.providers:
                tier.routes.append(TierRoute(
                    provider_name=provider_name,
                    model=route.get("model", ""),
                    priority=i + 1,
                ))
        if tier.routes:
            cat.tiers[tier_name] = tier

    # Ensure at least free tier routes if nothing is configured
    _ensure_fallback(cat)

    return cat


def _ensure_fallback(cat: CatalogConfig) -> None:
    """If no providers are available, register Zen free as ultimate fallback.

    Note: The gateway (build_gateway_from_env) also registers Zen free.
    This catalog-level fallback ensures tests and tools can work standalone.
    """
    if cat.providers:
        return
    if yaml is None:
        return  # can't load anything without yaml

    # Register OpenCode Zen free with known working models
    zen = ProviderSlot(
        name="opencode_free",
        display_name="OpenCode Zen (Free)",
        base_url="",  # empty → Zen
        api_key="",
        protocol="openai_chat",
        models=["nemotron-3-ultra-free", "hy3-free", "deepseek-v4-flash-free", "big-pickle"],
        is_default=True,
    )
    cat.providers["opencode_free"] = zen

    # Only add basic fast/strong tiers for standalone catalog use.
    # The gateway's build_gateway_from_env adds its own routes.
    if "fast" not in cat.tiers:
        cat.tiers["fast"] = TierConfig(name="fast", routes=[
            TierRoute(provider_name="opencode_free", model="nemotron-3-ultra-free", priority=1),
        ])
    if "strong" not in cat.tiers:
        cat.tiers["strong"] = TierConfig(name="strong", routes=[
            TierRoute(provider_name="opencode_free", model="nemotron-3-ultra-free", priority=1),
        ])


def list_provider_presets(config_path: Optional[str] = None) -> list[dict]:
    """Return provider definitions for the BYOK connection picker.

    Presets are metadata only; they are never treated as connected until a
    credential is present. This mirrors provider pickers that separate the
    provider directory from authenticated provider instances.
    """
    raw = _load_raw(Path(config_path) if config_path else _resolve_config_path())
    result = []
    for key, cfg in (raw.get("providers") or {}).items():
        if not isinstance(cfg, dict):
            continue
        base_url = str(cfg.get("base_url", "") or "")
        result.append({
            "id": key,
            "name": cfg.get("name", key),
            "base_url": base_url or "https://opencode.ai/zen/v1",
            "protocol": cfg.get("protocol", "openai_chat"),
            "env_key": cfg.get("api_key_env", ""),
            "free": not base_url,
            "requires_key": bool(base_url),
        })
    return result


def get_default_provider(cat: CatalogConfig) -> Optional[ProviderSlot]:
    """Get the default provider (first with is_default=True)."""
    for slot in cat.providers.values():
        if slot.is_default:
            return slot
    # Fallback: first available provider
    for slot in cat.providers.values():
        return slot
    return None
