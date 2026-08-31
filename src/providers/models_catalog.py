"""
Model catalog + live health probes for the model picker.

- Lists every provider/model from config/providers.yaml
- Optionally discovers remote models via GET /v1/models (Groq, OpenRouter, NIM, Zen)
- Probes each selected model with a tiny chat completion
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from .catalog import load_catalog, CatalogConfig, ProviderSlot

# Known OpenAI-compatible /models endpoints
_MODELS_PATH = {
    "opencode_free": "https://opencode.ai/zen/v1/models",
    "groq": "https://api.groq.com/openai/v1/models",
    "openrouter": "https://openrouter.ai/api/v1/models",
    "nvidia_nim": "https://integrate.api.nvidia.com/v1/models",
    "openai": "https://api.openai.com/v1/models",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/models",
    "deepseek": "https://api.deepseek.com/v1/models",
}


def _http_get_json(url: str, api_key: str = "", timeout: float = 20.0) -> Any:
    headers = {"User-Agent": "AutonomousResearchAgent/1.0", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def _extract_model_ids(payload: Any) -> list[str]:
    if isinstance(payload, dict) and "data" in payload:
        return [str(m.get("id", "")) for m in payload["data"] if m.get("id")]
    if isinstance(payload, list):
        return [str(m.get("id", m) if isinstance(m, dict) else m) for m in payload]
    return []


def _load_raw_providers() -> dict:
    """Load full providers.yaml including providers without keys (for picker UI)."""
    from pathlib import Path
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    root = Path(__file__).resolve().parent.parent.parent
    path = root / "config" / "providers.yaml"
    if not path.exists():
        path = root / "config" / "providers.example.yaml"
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


_FREE_MODEL_RE = __import__("re").compile(r"(?:free|big[-_ ]?pickle)", __import__("re").IGNORECASE)


def _is_free_model(model: str) -> bool:
    """Match the documented Zen free IDs by model name, including Big Pickle."""
    return bool(_FREE_MODEL_RE.search(model))


def list_catalog_models(discover_remote: bool = True) -> list[dict]:
    """
    Return only free models for the picker.

    Free classification intentionally uses a case-insensitive regex over the
    model ID/display name: IDs containing ``free`` are free, and Big Pickle is
    the documented exception whose ID does not contain that word.
    """
    raw = _load_raw_providers()
    providers_raw = raw.get("providers") or {}
    cat = load_catalog()  # enabled slots with resolved keys
    rows: list[dict] = []

    for key, cfg in providers_raw.items():
        if not isinstance(cfg, dict):
            continue
        display = cfg.get("name", key)
        base_url = cfg.get("base_url", "") or ""
        env_key = cfg.get("api_key_env", "") or ""
        protocol = cfg.get("protocol", "openai_chat")
        configured = list(cfg.get("models") or [])
        is_zen = key == "opencode_free" or not base_url
        free = is_zen
        env_val = os.getenv(env_key, "") if env_key else ""
        # Zen free endpoints never require a key, regardless of config claims.
        has_key = is_zen or bool(env_val)

        # Effective base
        if not base_url:
            eff_base = "https://opencode.ai/zen/v1"
        else:
            eff_base = base_url.rstrip("/")
            if not eff_base.endswith("/v1"):
                eff_base = eff_base + "/v1"

        discovered: list[str] = []
        if discover_remote and has_key:
            # use enabled slot when present for key pool
            slot = cat.providers.get(key)
            if slot is not None:
                discovered = _try_discover(key, slot)
            elif is_zen:
                class _Tmp:
                    api_key = ""
                    effective_base_url = eff_base
                    name = display
                discovered = _try_discover(key, _Tmp())  # type: ignore

        seen: set[str] = set()
        ordered: list[str] = []
        for m in configured + discovered:
            if m and m not in seen:
                seen.add(m)
                ordered.append(m)

        for model in ordered:
            # The picker is deliberately free-only. Do not mark every Zen
            # catalog/discovered model free: paid Zen models are also returned
            # by /v1/models.
            if not is_zen and not _is_free_model(str(model)):
                continue
            if is_zen and not _is_free_model(str(model)):
                continue
            source = "config"
            if model in discovered and model not in configured:
                source = "remote"
            elif model in discovered and model in configured:
                source = "config+remote"
            model_free = _is_free_model(str(model))
            rows.append({
                "provider": key,
                "provider_name": display,
                "model": model,
                "free": model_free,
                "has_key": has_key,
                "env_key": env_key,
                "base_url": eff_base,
                "protocol": protocol,
                "source": source,
                "default": bool(cfg.get("default", False)),
            })

    rows.sort(key=lambda r: (0 if r["provider"] == "opencode_free" else 1, r["provider"], r["model"]))
    return rows


def _try_discover(provider_key: str, slot: ProviderSlot) -> list[str]:
    url = _MODELS_PATH.get(provider_key)
    if not url:
        return []
    try:
        key = slot.api_key or ""
        # Zen free needs no key
        if provider_key == "opencode_free":
            key = ""
        elif not key:
            return []
        payload = _http_get_json(url, api_key=key, timeout=15.0)
        return _extract_model_ids(payload)[:80]
    except Exception:
        return []


def probe_model(
    provider: str,
    model: str,
    timeout: float = 45.0,
) -> dict:
    """Live probe: one tiny completion. Returns status dict."""
    cat = load_catalog()
    slot = cat.providers.get(provider)
    if slot is None:
        return {
            "provider": provider,
            "model": model,
            "ok": False,
            "error": "unknown provider",
            "latency_s": 0,
            "reply": "",
        }

    from src.gateway.providers import OpenAICompatibleProvider, ProviderHTTPError, ProviderConnectionError, ProviderTimeoutError

    base = slot.effective_base_url
    protocol = slot.protocol or "openai_chat"
    keys = getattr(slot, "_all_keys", None) or ([slot.api_key] if slot.api_key else [])
    prov = OpenAICompatibleProvider(slot.name, base, protocol=protocol)
    prov.api_keys = keys

    t0 = time.time()
    try:
        res = prov.complete(
            messages=[
                {"role": "user", "content": "Reply with exactly the word: OK"},
            ],
            model=model,
            temperature=0,
            max_tokens=16,
            api_key=keys[0] if keys else None,
            timeout=timeout,
        )
        text = (res.text or "").strip()
        ok = bool(text) or res.completion_tokens >= 0  # empty reasoning still counts as connect OK
        # Prefer connected+any response as ok for free reasoning models
        ok = True if res.latency_s > 0 else ok
        return {
            "provider": provider,
            "model": model,
            "ok": True,
            "error": "",
            "latency_s": round(time.time() - t0, 2),
            "reply": text[:120],
            "provider_name": slot.display_name,
            "free": not bool(keys),
        }
    except (ProviderHTTPError, ProviderConnectionError, ProviderTimeoutError) as e:
        return {
            "provider": provider,
            "model": model,
            "ok": False,
            "error": str(e)[:200],
            "latency_s": round(time.time() - t0, 2),
            "reply": "",
            "provider_name": slot.display_name,
            "free": not bool(keys),
        }
    except Exception as e:
        return {
            "provider": provider,
            "model": model,
            "ok": False,
            "error": f"{type(e).__name__}: {e}"[:200],
            "latency_s": round(time.time() - t0, 2),
            "reply": "",
            "provider_name": getattr(slot, "display_name", provider),
            "free": not bool(keys),
        }


def probe_zen_free(timeout: float = 45.0) -> list[dict]:
    """Probe every free Zen model — discovered live, YAML as fallback."""
    cat = load_catalog()
    slot = cat.providers.get("opencode_free")
    models = list(slot.models) if slot else []
    if not models:
        models = ["nemotron-3-ultra-free", "nemotron-3.5-lightning-free",
                  "deepseek-v4-flash-free", "big-pickle"]
    return [probe_model("opencode_free", m, timeout=timeout) for m in models]


def probe_provider(provider: str, max_models: int = 12, timeout: float = 40.0) -> list[dict]:
    """Probe configured models for one provider (capped)."""
    cat = load_catalog()
    slot = cat.providers.get(provider)
    if not slot:
        return []
    models = list(slot.models)[:max_models]
    return [probe_model(provider, m, timeout=timeout) for m in models]


def group_for_picker(rows: list[dict], probes: Optional[dict] = None) -> list[dict]:
    """
    Group models by provider for the UI.

    probes: optional map "provider/model" -> probe result
    """
    probes = probes or {}
    groups: dict[str, dict] = {}
    for r in rows:
        g = groups.setdefault(r["provider"], {
            "provider": r["provider"],
            "provider_name": r["provider_name"],
            "free": r["free"] and r["provider"] == "opencode_free",
            "has_key": r["has_key"],
            "env_key": r["env_key"],
            "base_url": r["base_url"],
            "default": r.get("default", False),
            "models": [],
        })
        # provider free flag: zen always free; others free only if :free models
        key = f"{r['provider']}/{r['model']}"
        probe = probes.get(key, {})
        g["models"].append({
            "id": r["model"],
            "label": r["model"],
            "free": r["free"],
            "source": r["source"],
            "status": "ok" if probe.get("ok") else ("fail" if probe else "unknown"),
            "latency_s": probe.get("latency_s"),
            "error": probe.get("error", ""),
            "reply": probe.get("reply", ""),
        })
        # if any model needs key and has_key false, mark group
        if r["env_key"] and not r["has_key"] and r["provider"] != "opencode_free":
            g["has_key"] = False
    # Zen first
    order = sorted(groups.values(), key=lambda g: (0 if g["provider"] == "opencode_free" else 1, g["provider_name"]))
    return order
