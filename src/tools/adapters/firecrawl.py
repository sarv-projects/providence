"""
Firecrawl Search & Scrape Adapter — Open Source Web Search, Crawling, Mapping & Scraping.

Based on https://github.com/firecrawl/firecrawl

Modes supported:
1. Self-Hosted Docker Container (Zero API Key):
   - URL: http://localhost:3002 or FIRECRAWL_BASE_URL
   - Docker command: `docker run -d -p 3002:3002 --name firecrawl ghcr.io/firecrawl/firecrawl:latest`
2. Firecrawl Cloud (Optional API key):
   - URL: https://api.firecrawl.dev (needs FIRECRAWL_API_KEY)
3. Native Embedded Firecrawl Engine (Zero Docker, Zero Key):
   - Runs Firecrawl's 2-stage search pipeline natively:
     Stage 1: Multi-engine search discovery (SearXNG / DuckDuckGo)
     Stage 2: Trafilatura / HTML DOM markdown extraction & cleaning
"""

from __future__ import annotations

import logging

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Dict, Optional

FIRE_CLOUD = "https://api.firecrawl.dev"
FIRE_SELF_DEFAULT = "http://localhost:3002"
FIRE_TIMEOUT = 25.0


def _get_base_and_key() -> tuple[str, str]:
    """Return (base_url, api_key). Cloud if key set, else self-hosted."""
    key = os.getenv("FIRECRAWL_API_KEY", "")
    if key:
        return FIRE_CLOUD, key
    base = os.getenv("FIRECRAWL_BASE_URL", FIRE_SELF_DEFAULT)
    return base.rstrip("/"), ""


def _is_self_hosted() -> bool:
    """Check if a self-hosted Firecrawl instance is reachable."""
    base, key = _get_base_and_key()
    if key:
        return False
    try:
        req = urllib.request.Request(
            f"{base}/v2/health",
            headers={"User-Agent": "AutonomousResearchAgent/1.0"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        try:
            req = urllib.request.Request(
                f"{base}/v1/health",
                headers={"User-Agent": "AutonomousResearchAgent/1.0"},
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            return False


def _request(base: str, key: str, endpoint: str, payload: dict) -> dict:
    """Make a Firecrawl API request (works for both v2 and v1, cloud and self-hosted)."""
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "AutonomousResearchAgent/1.0",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"

    req = urllib.request.Request(
        f"{base}{endpoint}", data=body, method="POST", headers=headers
    )
    with urllib.request.urlopen(req, timeout=FIRE_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def firecrawl_search(query: str, max_results: int = 5) -> List[Dict]:
    """Search the web via Firecrawl (self-hosted container, cloud API, or native Firecrawl pipeline)."""
    if not query.strip():
        return []

    base, key = _get_base_and_key()
    
    # 1. Try Firecrawl Container / Cloud API
    if key or _is_self_hosted():
        for version in ("/v2/search", "/v1/search"):
            try:
                data = _request(base, key, version, {
                    "query": query,
                    "limit": min(max_results, 10),
                    "scrapeOptions": {"formats": ["markdown"]}
                })
                web_results = data.get("data", {}).get("web", []) or data.get("data", [])
                if isinstance(web_results, list) and len(web_results) > 0:
                    results = []
                    source_tag = "firecrawl-cloud" if key else "firecrawl-self"
                    for item in web_results:
                        markdown = item.get("markdown", "") or item.get("description", "")
                        results.append({
                            "title": item.get("title", "") or item.get("metadata", {}).get("title", ""),
                            "url": item.get("url", ""),
                            "content": (item.get("description", "") or markdown)[:600],
                            "raw_content": markdown,
                            "score": 0.90,
                            "source": source_tag,
                        })
                    return results[:max_results]
            except Exception as e:
                pass

    # 2. Native Firecrawl Pipeline Fallback (Zero Docker, Zero Key)
    # Stage 1: SearXNG / DuckDuckGo Search Discovery
    # Stage 2: Trafilatura Clean Markdown Extraction
    from .builtin_scraper import builtin_search, builtin_extract
    search_hits = builtin_search(query, max_results=max_results)
    
    # Enrich search hits with full page markdown extraction
    extracted_urls = [h["url"] for h in search_hits if h.get("url")]
    extracted_pages = {p["url"]: p.get("content", "") for p in builtin_extract(extracted_urls)}

    results = []
    for hit in search_hits:
        url = hit.get("url", "")
        markdown = extracted_pages.get(url, hit.get("content", ""))
        results.append({
            "title": hit.get("title", ""),
            "url": url,
            "content": hit.get("content", "")[:500],
            "raw_content": markdown,
            "score": 0.85,
            "source": "firecrawl-native",
        })

    return results[:max_results]


def firecrawl_scrape(url: str) -> Dict:
    """Scrape a single URL via Firecrawl (self-hosted container, cloud, or native trafilatura)."""
    if not url:
        return {}
    try:
        from ..urlguard import is_safe_url
        if not is_safe_url(url):
            return {}
    except ImportError:
        pass

    base, key = _get_base_and_key()
    if key or _is_self_hosted():
        for version in ("/v2/scrape", "/v1/scrape"):
            try:
                data = _request(base, key, version, {
                    "url": url,
                    "formats": ["markdown"],
                    "onlyMainContent": True,
                })
                content = data.get("data", {}).get("markdown", "")
                title = data.get("data", {}).get("metadata", {}).get("title", "")
                if content:
                    return {"url": url, "content": content, "title": title}
            except Exception:
                logging.getLogger(__name__).debug("ignored error", exc_info=True)

    # Native fallback
    from .builtin_scraper import scrape_url
    return scrape_url(url)


def firecrawl_extract(urls: List[str]) -> List[Dict]:
    """Extract full markdown from multiple URLs via Firecrawl.

    URLs are scraped CONCURRENTLY (bounded worker pool) so a slow page
    doesn't serialize the whole batch.
    """
    if not urls:
        return []
    urls = list(dict.fromkeys(urls))[:8]
    results: List[Dict] = []
    seen: set = set()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(len(urls), 5)) as executor:
        futures = {executor.submit(firecrawl_scrape, url): url for url in urls}
        for future in as_completed(futures):
            try:
                r = future.result()
            except Exception:
                continue
            if r and r.get("content"):
                u = r.get("url", "")
                if u and u not in seen:
                    seen.add(u)
                    results.append(r)
    return results


def firecrawl_map(url: str) -> List[str]:
    """Map out domain sitemap / links via Firecrawl /v2/map."""
    if not url:
        return []
    try:
        from ..urlguard import is_safe_url
        if not is_safe_url(url):
            return []
    except ImportError:
        pass
    base, key = _get_base_and_key()
    try:
        data = _request(base, key, "/v2/map", {"url": url})
        return data.get("links", [])
    except Exception:
        return []
