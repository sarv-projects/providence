"""
Phase A tests — provider catalog, modes, gateway Zen free integration.

Offline except for one optional live probe. Run with:
    uv run python test_phase_a.py
"""

import sys
import os

# ── 1. Provider Catalog ────────────────────────────────────────────────
def test_catalog_loads_example():
    from src.providers.catalog import load_catalog
    cat = load_catalog()
    assert len(cat.providers) > 0, "No providers loaded"
    print("1/8 catalog loads from example YAML OK")


def test_zen_free_is_available():
    from src.providers.catalog import load_catalog
    cat = load_catalog()
    has_zen = any(p.base_url == "" for p in cat.providers.values())
    assert has_zen, "Zen free provider not found"
    print("2/8 Zen free provider available OK")


def test_provider_slot_resolution():
    from src.providers.catalog import ProviderSlot
    slot = ProviderSlot(name="test", display_name="Test", base_url="", api_key="")
    assert slot.effective_base_url == "https://opencode.ai/zen/v1"
    assert not slot.has_auth
    slot2 = ProviderSlot(name="test2", display_name="Test2", base_url="https://api.openai.com", api_key="sk-abc")
    assert slot2.effective_base_url == "https://api.openai.com/v1"
    assert slot2.has_auth
    print("3/8 provider slot resolution OK")


# ── 2. Mode System ─────────────────────────────────────────────────────
def test_modes_load():
    from src.engine.modes import load_modes, get_mode
    registry = load_modes()
    assert "standard" in registry.modes
    assert "balanced" in registry.quality_dials
    mode = get_mode(registry, "standard")
    assert mode.budgets.max_tokens == 100000
    assert mode.quality_dial == "balanced"
    print("4/8 modes load correctly OK")


def test_mode_fallback():
    from src.engine.modes import load_modes, get_mode
    registry = load_modes()
    mode = get_mode(registry, "nonexistent")
    assert mode.name == "standard"  # falls back to standard
    print("5/8 mode fallback OK")


# ── 3. Gateway Zen Free Integration ────────────────────────────────────
def test_gateway_has_zen_free_route():
    from src.llm import gateway_info, reset_gateway
    reset_gateway()
    info = gateway_info()
    has_zen = any("opencode_free" in r["provider"] for r in info["routes"])
    assert has_zen, "Zen free route not registered in gateway"
    print("6/8 gateway Zen free route registered OK")


def test_no_auth_header_for_empty_key():
    from src.gateway.providers import OpenAICompatibleProvider
    import urllib.request
    p = OpenAICompatibleProvider("test", "https://opencode.ai/zen/v1")
    p.api_keys = []

    # Construct the request to check headers
    import json
    payload = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    body = json.dumps(payload).encode("utf-8")

    # We verify the logic: no key → no Authorization
    from src.gateway.providers import OpenAICompatibleProvider as P
    headers_used = []
    original_urlopen = urllib.request.urlopen
    def capture(url, **kw):
        # Don't actually call; just captures the request
        headers_used.append(kw.get("data", b""))
        raise Exception("captured")

    # The provider must omit Authorization when api_keys is empty.
    # Capture the real urllib Request via a stubbed urlopen (no network).
    captured = {}

    class _FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"choices": [{"message": {"content": "ok"}}], "usage": {}}'

    def _fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.header_items())
        return _FakeResp()

    urllib.request.urlopen = _fake_urlopen
    try:
        p.complete([{"role": "user", "content": "hi"}], model="test")
    finally:
        urllib.request.urlopen = original_urlopen
    assert "Authorization" not in (captured.get("headers") or {}), \
        f"empty key must not send Authorization, got: {captured.get('headers')}"
    assert p.api_keys == [] or p.api_keys == [""] or all(k == "" for k in p.api_keys)
    # Just verify the setting is correct
    print("7/8 empty key → no Authorization header OK")


# ── 4. Doctor Command ──────────────────────────────────────────────────
def test_doctor_runs():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", "from main import doctor; doctor()"],
        capture_output=True, text=True, timeout=60,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    assert "SYSTEM DOCTOR" in result.stdout
    assert "LLM Gateway" in result.stdout
    print("8/8 doctor command runs OK")


TESTS = [
    test_catalog_loads_example,
    test_zen_free_is_available,
    test_provider_slot_resolution,
    test_modes_load,
    test_mode_fallback,
    test_gateway_has_zen_free_route,
    test_no_auth_header_for_empty_key,
    test_doctor_runs,
]

if __name__ == "__main__":
    passed = 0
    for t in TESTS:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__} -> {e}")
    print(f"\n{passed}/{len(TESTS)} tests passed")
    sys.exit(0 if passed == len(TESTS) else 1)
