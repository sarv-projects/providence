"""
Tool registry — capability-based tool discovery and selection.

Tools are registered with:
  - name: unique identifier (e.g. "tavily", "wikipedia")
  - capabilities: set of tags (e.g. {"web_search", "factual", "free"})
  - search_fn: callable(query, max_results) -> list[dict]
  - extract_fn: callable(urls) -> list[dict] (optional)
  - priority: int, higher = preferred when multiple tools match

Performance upgrades:
  - TTL search cache: repeated queries within a run (or across overlapping
    runs) hit the cache instead of the provider — big wall-time win in the
    research loop, which re-asks overlapping sub-queries each iteration.
  - Parallel extraction: URLs are fetched concurrently (not one-by-one),
    bounded by a worker cap so slow pages don't serialize the round.
  - Optional provider fusion: with TOOL_FUSE_SEARCH=1 the top web_search
    providers run CONCURRENTLY and results are merged by URL instead of the
    sequential fallback chain (broader coverage, one round-trip).
"""

from __future__ import annotations

import logging

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

SearchFunc = Callable[[str, int], list[dict]]
ExtractFunc = Callable[[list[str]], list[dict]]

# ── Search cache ────────────────────────────────────────────────────────
_SEARCH_CACHE_TTL_S = float(os.getenv("TOOL_SEARCH_CACHE_TTL_S", "600"))
_SEARCH_CACHE_MAX = int(os.getenv("TOOL_SEARCH_CACHE_MAX", "256"))


class _SearchCache:
    """Small thread-safe TTL cache keyed by (query, max_results).

    Only successful (non-empty) results are cached so a transient failure
    can retry the provider on the next call. Entries expire after TTL so
    recency-sensitive queries still see fresh results.
    """

    def __init__(self, ttl_s: float = _SEARCH_CACHE_TTL_S, max_entries: int = _SEARCH_CACHE_MAX):
        self._ttl_s = ttl_s
        self._max = max_entries
        self._data: dict[tuple[str, int], tuple[float, list[dict]]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: tuple[str, int]) -> Optional[list[dict]]:
        with self._lock:
            hit = self._data.get(key)
            if not hit:
                self._misses += 1
                return None
            ts, results = hit
            if time.time() - ts > self._ttl_s:
                self._data.pop(key, None)
                self._misses += 1
                return None
            self._hits += 1
            return results

    def put(self, key: tuple[str, int], results: list[dict]) -> None:
        if not results:
            return
        with self._lock:
            self._data[key] = (time.time(), results)
            if len(self._data) > self._max:
                # Evict oldest by insertion order (dicts preserve order in 3.7+)
                for k in list(self._data.keys())[: len(self._data) - self._max]:
                    self._data.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        """Snapshot for observability (cache size, TTL, hit/miss rate)."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._data),
                "max_entries": self._max,
                "ttl_s": self._ttl_s,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }


class Tool:
    """A registered tool with search and optional extract capability."""
    def __init__(
        self,
        name: str,
        capabilities: set[str],
        search_fn: SearchFunc,
        extract_fn: Optional[ExtractFunc] = None,
        priority: int = 0,
    ) -> None:
        self.name = name
        self.capabilities = capabilities
        self.search_fn = search_fn
        self.extract_fn = extract_fn
        self.priority = priority

    def has_capability(self, cap: str) -> bool:
        return cap in self.capabilities


class ToolRegistry:
    """Registry of all available research tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._cache = _SearchCache()

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_all(self) -> list[Tool]:
        return list(self._tools.values())

    def list_by_capability(self, capability: str) -> list[Tool]:
        """Return tools matching a capability, sorted by priority (desc)."""
        matching = [t for t in self._tools.values() if t.has_capability(capability)]
        matching.sort(key=lambda t: -t.priority)
        return matching

    def clear_cache(self) -> None:
        """Clear the TTL search cache (useful between runs/tests)."""
        self._cache.clear()

    def cache_stats(self) -> dict:
        """Expose search-cache observability metrics (size, TTL, hit rate)."""
        return self._cache.stats()

    def search(
        self,
        queries: list[str],
        max_results: int = 5,
        prefer_capability: str = "web_search",
    ) -> list[dict]:
        """Search using the best available tool, always fusing Wikipedia.

        1. Primary tool (highest priority, e.g. Exa/Firecrawl)
        2. Wikipedia (always queried in parallel as supplementary source)
        Results are merged: primary first (deduped by URL), then Wikipedia novel URLs.

        Caching: the per-query provider results are cached with a TTL, so
        overlapping sub-queries across research iterations reuse results.

        Fusion: when TOOL_FUSE_SEARCH=1, the top web_search providers run
        CONCURRENTLY and results are merged by URL (broader coverage in a
        single round-trip) instead of the sequential fallback chain.
        """
        tools = self.list_by_capability(prefer_capability)
        if not tools:
            tools = self.list_all()

        if not tools:
            return []

        primary_tools = [t for t in tools if not t.has_capability("always")]
        always_tools = [t for t in tools if t.has_capability("always")]
        fuse = os.getenv("TOOL_FUSE_SEARCH", "").lower() in ("1", "true", "yes", "on")

        all_results: list[dict] = []
        seen_urls: set[str] = set()

        def _merge(rs: list[dict]) -> None:
            for r in rs:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)

        # Start the always-tools (Wikipedia) in parallel with the primary chain.
        always_results: list[dict] = []
        always_done = threading.Event()

        def _run_always() -> None:
            try:
                for tool in always_tools:
                    try:
                        always_results.extend(_search_with_cache(tool, queries, max_results, self._cache))
                    except Exception as e:
                        print(f"  [tool:{tool.name}] search failed: {e}")
            finally:
                always_done.set()

        threading.Thread(target=_run_always, daemon=True).start()

        if fuse and len(primary_tools) >= 2:
            # ── Provider fusion (TOOL_FUSE_SEARCH=1) ──
            # Top providers run concurrently; results merged by URL. Falls back
            # to the sequential chain if every provider fails.
            fused: list[dict] = []
            fused_seen: set[str] = set()
            with ThreadPoolExecutor(max_workers=min(len(primary_tools), 3)) as executor:
                futures = {
                    executor.submit(_search_with_cache, tool, queries, max_results, self._cache): tool
                    for tool in primary_tools[:3]
                }
                for future in as_completed(futures):
                    tool = futures[future]
                    try:
                        rs = future.result()
                    except Exception as e:
                        print(f"  [tool:{tool.name}] fused search failed: {e}")
                        continue
                    if rs:
                        for r in rs:
                            url = r.get("url", "")
                            if url and url not in fused_seen:
                                fused_seen.add(url)
                                fused.append(r)
                        print(f"  [tool:{tool.name}] fused search OK ({len(rs)} raw results)")
            if fused:
                fused.sort(key=lambda x: x.get("score", 0), reverse=True)
                _merge(fused)
        else:
            # ── Sequential fallback chain (default) ──
            for tool in primary_tools:
                try:
                    results = _search_with_cache(tool, queries, max_results, self._cache)
                except Exception as e:
                    print(f"  [tool:{tool.name}] search failed: {e} — trying next provider")
                    continue
                if results:
                    _merge(results)
                    print(f"  [tool:{tool.name}] primary search OK ({len(results)} raw results)")
                    break
                print(f"  [tool:{tool.name}] returned 0 results — trying next provider")

        always_done.wait(timeout=60)
        _merge(always_results)

        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results

    def extract(self, urls: list[str]) -> list[dict]:
        """Extract content from URLs using the best available extract tool.

        Tools are tried in priority order (highest first); the first tool that
        returns a non-empty result wins. A tool returning [] (e.g. Wikipedia
        for a non-Wikipedia URL) falls through to the next candidate.

        URLs are fetched CONCURRENTLY (bounded worker pool) so a few slow
        pages don't serialize the whole round.
        """
        candidates = sorted(
            (t for t in self.list_all() if t.extract_fn),
            key=lambda t: -t.priority,
        )
        for tool in candidates:
            try:
                out = _parallel_extract_with_tool(tool, urls)
                if out:
                    return out
            except Exception:
                continue
        return []


def _search_with_cache(
    tool: Tool,
    queries: list[str],
    max_results: int,
    cache: _SearchCache,
) -> list[dict]:
    """Run searches with a tool, using the registry TTL cache per query."""
    all_results: list[dict] = []
    seen_urls: set[str] = set()

    with ThreadPoolExecutor(max_workers=min(len(queries), 8)) as executor:
        def _run(q: str) -> list[dict]:
            key = (tool.name, q, max_results)
            cached = cache.get(key)
            if cached is not None:
                return cached
            try:
                rs = tool.search_fn(q, max_results)
            except Exception as e:
                print(f"  [tool:{tool.name}] query failed: {e}")
                return []
            cache.put(key, rs)
            return rs

        futures = {executor.submit(_run, q): q for q in queries}
        for future in as_completed(futures):
            try:
                results = future.result()
                for r in results:
                    url = r.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(r)
            except Exception as e:
                print(f"  [tool:{tool.name}] query failed: {e}")

    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_results


def _parallel_extract_with_tool(tool: Tool, urls: list[str], max_workers: int = 6) -> list[dict]:
    """Extract URLs concurrently with a single tool.

    Some extract_fns are true batch APIs (Exa /contents, Tavily /extract) —
    those are called once with the full list. Others (firecrawl, builtin,
    mineru) are per-URL loops internally, so we shard URLs across a bounded
    worker pool and merge results instead of letting them run serially.
    """
    if not urls:
        return []
    urls = list(dict.fromkeys(urls))[:12]  # dedupe + bound

    # Tools whose extract_fn already accepts a batch and parallelizes/fast-paths
    # internally. Everything else is treated as a per-URL extractor.
    batch_tools = {"exa", "tavily", "wikipedia"}

    if tool.name in batch_tools:
        try:
            out = tool.extract_fn(urls)
            if out:
                return out
        except Exception:
            logging.getLogger(__name__).debug("ignored error", exc_info=True)

    # Per-URL concurrent extraction (firecrawl, builtin, mineru, nougat, …).
    results: list[dict] = []
    seen_urls: set[str] = set()
    with ThreadPoolExecutor(max_workers=min(len(urls), max_workers)) as executor:
        def _extract_one(url: str) -> dict:
            try:
                rs = tool.extract_fn([url])
                return rs[0] if rs else {}
            except Exception:
                return {}

        futures = {executor.submit(_extract_one, u): u for u in urls}
        for future in as_completed(futures):
            try:
                r = future.result()
            except Exception:
                continue
            url = r.get("url", "")
            if url and url not in seen_urls and r.get("content"):
                seen_urls.add(url)
                results.append(r)

    results.sort(key=lambda x: len(x.get("content", "") or ""), reverse=True)
    return results


# Module-level singleton
_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def register_tool(
    name: str,
    capabilities: set[str],
    search_fn: SearchFunc,
    extract_fn: Optional[ExtractFunc] = None,
    priority: int = 0,
) -> Tool:
    """Register a tool in the module-level registry."""
    tool = Tool(name, capabilities, search_fn, extract_fn, priority)
    get_registry().register(tool)
    return tool
