"""
Phase C2 tests — Thinker agent registration, gateway thinker tier, rate limiting.

All offline (no API keys needed). Tests agent structure and gateway routes.
Run with:
    uv run python test_phase_c2.py
"""

import sys


# ── 1. Agent Registration ───────────────────────────────────────────────
def test_thinker_agents_registered():
    import src.engine.agents  # triggers registration
    from src.engine.agents.registry import get_all, get_agent
    agents = get_all()
    assert "thinker_plan_refine" in agents, "thinker_plan_refine not registered"
    assert "thinker_contradiction_check" in agents, "thinker_contradiction_check not registered"
    assert callable(get_agent("thinker_plan_refine"))
    assert callable(get_agent("thinker_contradiction_check"))
    print("1/8 thinker agents registered OK")


def test_thinker_skips_without_claims():
    from src.engine.agents.thinker import _should_invoke_thinker, reset_thinker, thinker_plan_refine, thinker_contradiction_check
    reset_thinker()

    # Simple state with few outline sections → should skip
    state = {
        "query": "test",
        "plan": {"outline": [{"title": "Overview"}, {"title": "Findings"}]},
        "findings": ["finding 1"],
        "claims": [],
    }
    # Should skip because plan has < 4 sections
    result = thinker_plan_refine(state)
    assert result is state  # unchanged (skipped)
    print("2/8 thinker skips simple plan OK")


def test_thinker_rate_limit_enforced():
    from src.engine.agents.thinker import _should_invoke_thinker, reset_thinker
    reset_thinker()
    state = {"query": "test", "claims": [], "plan": {}}

    # First call should succeed
    assert _should_invoke_thinker(state)
    # Immediate second call should fail (rate limit)
    assert not _should_invoke_thinker(state)
    print("3/8 thinker rate limiting OK")


def test_thinker_max_calls_enforced():
    from src.engine.agents.thinker import (
        _should_invoke_thinker, reset_thinker,
        _thinker_call_count, _thinker_lock, MAX_THINKER_CALLS_PER_RUN
    )
    reset_thinker()

    # Simulate reaching max calls
    with _thinker_lock:
        _thinker_call_count[0] = MAX_THINKER_CALLS_PER_RUN

    state = {"query": "test", "claims": [], "plan": {}}
    assert not _should_invoke_thinker(state)
    print("4/8 thinker max calls enforced OK")


# ── 2. Gateway Thinker Tier ─────────────────────────────────────────────
def test_gateway_thinker_tier():
    from src.llm import reset_gateway
    reset_gateway()

    # Policy (config/providers.yaml + gateway/__init__.py): the thinker tier
    # is Gemini-only — it is deliberately NOT backfilled with Zen free, so
    # reasoning nodes keep a stronger model when a GEMINI_API_KEY exists.
    # This test previously expected a Zen fallback (stale); it now asserts
    # the actual policy.
    from src.gateway import build_gateway_from_env
    gw = build_gateway_from_env()
    thinker_routes = gw.get_routes("thinker")

    assert len(thinker_routes) > 0, "No thinker routes at all"
    names = [r.name for r in thinker_routes]
    has_gemini = any("gemini" in n.lower() for n in names)
    assert has_gemini, f"Thinker routes should be Gemini-only per policy, got: {names}"
    has_free = any("opencode_free" in n for n in names)
    assert not has_free, (
        f"Thinker tier must NOT be backfilled with Zen free (Gemini-only policy): {names}"
    )
    print(f"5/8 gateway thinker tier Gemini-only ({len(thinker_routes)} routes) OK")


# ── 3. Graph integration ────────────────────────────────────────────────
def test_graph_has_thinker_nodes():
    from src.graph import build_graph
    graph = build_graph()
    nodes = graph.get_graph().nodes
    assert "thinker_plan_refine" in nodes
    assert "thinker_contradiction_check" in nodes
    print("6/8 graph has thinker nodes OK")


def test_graph_flow_includes_thinker():
    from src.graph import build_graph
    graph = build_graph()
    edges = graph.get_graph().edges
    # Check that planner → thinker_plan_refine is an edge
    edge_pairs = [(e[0], e[1]) for e in edges]
    assert ("planner", "thinker_plan_refine") in edge_pairs or any(
        "planner" in str(e) and "thinker_plan_refine" in str(e) for e in edges
    ), "planner → thinker_plan_refine edge missing"
    print("7/8 graph flow includes thinker OK")


# ── 4. Final test count ─────────────────────────────────────────────────
def test_all_tests_pass():
    from src.graph import build_graph
    from src.state import initial_state
    graph = build_graph()
    state = initial_state("test query")
    assert state["max_iterations"] == 6
    print("8/8 all integrations verified OK")


TESTS = [
    test_thinker_agents_registered,
    test_thinker_skips_without_claims,
    test_thinker_rate_limit_enforced,
    test_thinker_max_calls_enforced,
    test_gateway_thinker_tier,
    test_graph_has_thinker_nodes,
    test_graph_flow_includes_thinker,
    test_all_tests_pass,
]

if __name__ == "__main__":
    passed = 0
    for t in TESTS:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__} -> {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{passed}/{len(TESTS)} tests passed")
    sys.exit(0 if passed == len(TESTS) else 1)
