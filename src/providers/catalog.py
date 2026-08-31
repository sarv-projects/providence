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

# ---------------------------------------------------------------------------
# Live model discovery (de-hardcoding model IDs)
#
# Model IDs rotate frequently (OpenCode Zen deprecates free IDs without
# notice — e.g. ``hy3-free`` started returning 401 "not supported").  The
# YAML ``models:`` lists are treated as an OFFLINE FALLBACK only: at load
# time we query the provider's OpenAI-compatible GET /models endpoint and
# merge live IDs into each slot, then validate tier routes against the
# merged list so stale IDs never enter the gateway's failover chain.
# ---------------------------------------------------------------------------

_MODELS_ENDPOINTS = {
    "opencode_free": "https://opencode.ai/zen/v1/models",
    "groq": "https://api.groq.com/openai/v1/models",
    "openrouter": "https://openrouter.ai/api/v1/models",
    "nvidia_nim": "https://integrate.api.nvidia.com/v1/models",
    "openai": "https://api.openai.com/v1/models",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/models",
    "deepseek": "https://api.deepseek.com/v1/models",
}

_DISCOVERY_TTL_S = 1800.0  # 30 min
_DISCOVERY_TIMEOUT_S = 10.0
_MAX_DISCOVERED_MODELS = 80

import json as _json
import threading as _threading
import time as _time
import urllib.request as _urlreq

_disc_cache: dict[str, tuple[float, list[str]]] = {}
_disc_lock = _threading.Lock()


def _fetch_model_ids(provider_key: str, api_key: str) -> list[str]:
    """GET {provider}/models and return the live model IDs ([] on any failure)."""
    url = _MODELS_ENDPOINTS.get(provider_key)
    if not url:
        return []
    headers = {"User-Agent": "AutonomousResearchAgent/1.0", "Accept": "application/json"}
    # Zen free needs no key; everything else requires one
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif provider_key != "opencode_free":
        return []
    try:
        req = _urlreq.Request(url, headers=headers)
        with _urlreq.urlopen(req, timeout=_DISCOVERY_TIMEOUT_S) as resp:
            payload = _json.loads(resp.read().decode("utf-8", errors="ignore"))
        if isinstance(payload, dict) and "data" in payload:
            return [str(m.get("id", "")) for m in payload["data"] if m.get("id")]
        if isinstance(payload, list):
            return [str(m.get("id", m) if isinstance(m, dict) else m) for m in payload]
    except Exception:
        pass
    return []


def _discovered_models(provider_key: str, api_key: str) -> list[str]:
    """Live model IDs with a TTL cache; last-known-good survives outages."""
    now = _time.time()
    with _disc_lock:
        cached = _disc_cache.get(provider_key)
        if cached and now - cached[0] < _DISCOVERY_TTL_S:
            return cached[1]
    ids = _fetch_model_ids(provider_key, api_key)
    with _disc_lock:
        if ids:
            _disc_cache[provider_key] = (now, ids)
        elif cached:
            return cached[1]  # keep last-known-good on discovery failure
    return ids



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

    # Live model discovery — merge remote IDs into each slot (config first).
    # Failure-tolerant: on network error the YAML lists stand unchanged.
    if os.getenv("PROVIDENCE_DISABLE_MODEL_DISCOVERY", "") != "1":
        for key, slot in cat.providers.items():
            try:
                remote = _discovered_models(key, slot.api_key)[:_MAX_DISCOVERED_MODELS]
            except Exception:
                remote = []
            if remote:
                seen: set[str] = set()
                merged: list[str] = []
                for m in list(slot.models) + remote:
                    if m and m not in seen:
                        seen.add(m)
                        merged.append(m)
                slot.models = merged
                slot._discovered_ok = True

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

    # Validate tier routes against live discovery: when a provider's /models
    # responded, drop routes referencing IDs the provider no longer serves
    # (e.g. a deprecated Zen free model that would 401 mid-run), and append
    # newly discovered models that the static tiers do not know about yet.
    for tier_name, tier in list(cat.tiers.items()):
        if tier_name not in ("fast", "strong", "compare", "recency", "academic"):
            continue  # thinker is Gemini-only and curated by design
        validated: list[TierRoute] = []
        seen_pairs: set[tuple[str, str]] = set()
        for route in tier.routes:
            slot = cat.providers.get(route.provider_name)
            if slot is not None and getattr(slot, "_discovered_ok", False) and slot.models \
                    and route.model not in slot.models:
                continue  # stale ID — provider no longer serves it
            pair = (route.provider_name, route.model)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                validated.append(route)
        # Surface newly discovered IDs (config may lag the provider catalog)
        for provider_name in dict.fromkeys(r.provider_name for r in validated):
            slot = cat.providers.get(provider_name)
            if slot is None or not getattr(slot, "_discovered_ok", False):
                continue
            for model in slot.models:
                pair = (provider_name, model)
                if pair in seen_pairs:
                    continue
                if "free" in model.lower() or model == "big-pickle":
                    seen_pairs.add(pair)
                    validated.append(TierRoute(provider_name, model, len(validated) + 1))
        if validated:
            tier.routes = validated[:12]
            cat.tiers[tier_name] = tier
        else:
            cat.tiers.pop(tier_name, None)

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
        # Offline fallback only — load_catalog() refreshes these live via
        # GET /zen/v1/models when the network is available.
        models=["nemotron-3-ultra-free", "nemotron-3.5-lightning-free",
                "deepseek-v4-flash-free", "big-pickle"],
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
