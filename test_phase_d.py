"""
Phase D tests — tool registry, Wikipedia, built-in scraper, executor, integration.

Zero config: Wikipedia + built-in scraper always work. Tavily/Firecrawl optional.
Run with:
    uv run python test_phase_d.py
"""

import sys


# ── 1. Tool Registry ────────────────────────────────────────────────────
def test_registry_has_tools():
    from src.tools import get_registry
    registry = get_registry()
    tools = registry.list_all()
    names = [t.name for t in tools]
    assert "wikipedia" in names, f"Wikipedia not registered: {names}"
    assert "builtin" in names, f"Built-in scraper not registered: {names}"
    print(f"1/9 registry has {len(tools)} tools: {names} OK")


def test_builtin_scraper_registered():
    from src.tools import get_registry
    registry = get_registry()
    builtin = registry.get("builtin")
    assert builtin is not None, "Built-in scraper not registered!"
    assert builtin.has_capability("free")
    assert builtin.has_capability("extract")
    assert builtin.has_capability("always")
    print("2/9 built-in scraper always registered (zero config) OK")


def test_three_tools_minimum():
    from src.tools import get_registry
    registry = get_registry()
    tools = registry.list_all()
    names = [t.name for t in tools]
    free_count = sum(1 for t in tools if "free" in t.capabilities)
    assert free_count >= 2, f"Expected 2+ free tools, got {free_count}: {names}"

    from src.tools.adapters.builtin_scraper import scrape_url
    result = scrape_url("https://example.com")
    assert result.get("content"), "Built-in scraper failed on example.com"
    print(f"3/9 {len(tools)} tools ({free_count} free, zero config): {names} OK")


def test_registry_capability_filter():
    from src.tools import get_registry
    registry = get_registry()
    free_tools = registry.list_by_capability("free")
    assert len(free_tools) >= 2, f"Need 2+ free tools, got {len(free_tools)}"
    print("4/9 capability filtering OK")


def test_registry_priority_order():
    from src.tools import get_registry
    registry = get_registry()
    tools = registry.list_by_capability("web_search")
    priorities = [t.priority for t in tools]
    assert priorities == sorted(priorities, reverse=True), f"Not sorted: {priorities}"
    print(f"5/9 priority ordering OK: {[(t.name, t.priority) for t in tools]}")


# ── 2. Wikipedia ────────────────────────────────────────────────────────
def test_wikipedia_search():
    from src.tools.adapters.wikipedia import wiki_search
    results = wiki_search("artificial intelligence", max_results=3)
    assert len(results) >= 1, f"No Wikipedia results: {results}"
    assert results[0].get("title")
    assert results[0].get("url", "").startswith("https://en.wikipedia.org")
    print(f"6/9 Wikipedia search OK ({len(results)} results)")


def test_wikipedia_extract():
    from src.tools.adapters.wikipedia import wiki_extract
    results = wiki_extract(["https://en.wikipedia.org/wiki/Python_(programming_language)"])
    assert len(results) >= 1
    assert "Python" in results[0].get("content", "")
    print(f"7/9 Wikipedia extract OK ({len(results[0]['content'])} chars)")


# ── 3. Executor + Integration ──────────────────────────────────────────
def test_executor_fuses_wikipedia():
    from src.tools import execute_searches
    results = execute_searches(["quantum computing"], max_results=3)
    assert len(results) >= 1
    sources = set(r.get("source", "") for r in results)
    assert "wikipedia" in sources, f"Wikipedia not fused: {sources}"
    print(f"8/9 executor fuses Wikipedia OK ({len(results)} results, sources: {sources})")


def test_researcher_uses_tool_bus():
    from src.engine.agents import researcher
    import inspect
    source = inspect.getsource(researcher.researcher_gather)
    assert "execute_searches" in source, "Researcher not using tool bus"
    assert "get_registry" in source, "Researcher not checking registry"
    print("9/9 researcher uses tool bus OK")


TESTS = [
    test_registry_has_tools,
    test_builtin_scraper_registered,
    test_three_tools_minimum,
    test_registry_capability_filter,
    test_registry_priority_order,
    test_wikipedia_search,
    test_wikipedia_extract,
    test_executor_fuses_wikipedia,
    test_researcher_uses_tool_bus,
]

if __name__ == "__main__":
    # Phase D exercises the live tool bus (Wikipedia/Exa HTTP calls) — skip
    # cleanly in offline environments instead of hanging on DNS.
    from tests_common import require_live_network
    require_live_network("test_phase_d")

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
