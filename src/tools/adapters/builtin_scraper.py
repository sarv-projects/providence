"""
Built-in web scraper — zero config, no API key, no Docker, no setup.

Uses trafilatura to extract clean markdown from any URL. This is the
always-available fallback that makes the tool bus work out of the box.

Also provides a lightweight search capability using DuckDuckGo HTML scraping
(no API key required) as a free alternative to Tavily/Firecrawl.
"""

from __future__ import annotations

import logging

import json
import urllib.error
import urllib.parse
import urllib.request

HEADERS = {"User-Agent": "AutonomousResearchAgent/1.0 (+https://github.com)"}


def _unwrap_ddg_url(href: str) -> str:
    """Decode DuckDuckGo redirect links (//duckduckgo.com/l/?uddg=<target>).

    Without this the bus stores DDG redirect URLs and extraction fetches
    duckduckgo.com instead of the real article.
    """
    try:
        if "duckduckgo.com/l/" in href:
            qs = urllib.parse.parse_qs(urllib.parse.urlsplit(href).query)
            target = (qs.get("uddg") or [""])[0]
            if target:
                return urllib.parse.unquote(target)
    except Exception:
        pass
    return href


def _trafilatura():
    """Lazy trafilatura import — importing it at module level made the whole
    adapters package (and everything importing it) fail when the dependency
    was absent."""
    import trafilatura
    return trafilatura


def scrape_url(url: str) -> dict:
    """Extract clean markdown from a URL using trafilatura.

    Works on any HTML page — zero config, no API key, no Docker.
    URLs failing the SSRF guard are refused.
    """
    if not url:
        return {}
    try:
        from ..urlguard import is_safe_url
        if not is_safe_url(url):
            return {}
    except ImportError:
        pass
    try:
        t = _trafilatura()
        downloaded = t.fetch_url(url)
        if downloaded:
            markdown = t.extract(
                downloaded,
                output_format="markdown",
                include_links=True,
            )
            if markdown:
                title_line = markdown.split("\n")[0].strip("# ").strip()
                return {"url": url, "content": markdown, "title": title_line}
    except Exception:
        logging.getLogger(__name__).debug("ignored error", exc_info=True)
    return {}


def builtin_extract(urls: list[str]) -> list[dict]:
    """Extract content from multiple URLs using the built-in scraper.

    URLs are fetched CONCURRENTLY (bounded worker pool) — the always-on
    fallback path, so it should not serialize the research round.
    """
    if not urls:
        return []
    urls = list(dict.fromkeys(urls))[:8]
    results: list[dict] = []
    seen: set[str] = set()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(len(urls), 4)) as executor:
        futures = {executor.submit(scrape_url, url): url for url in urls}
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


def builtin_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using DuckDuckGo HTML (no API key needed).

    Falls back gracefully — if DuckDuckGo blocks, returns empty results.
    Wikipedia is the primary free search tool; this is supplementary.
    """
    if not query.strip():
        return []

    try:
        params = urllib.parse.urlencode({"q": query, "format": "json"})
        url = f"https://duckduckgo.com/html/?{params}"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Simple HTML parsing for DuckDuckGo results
        results = []
        from html.parser import HTMLParser

        class DDParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self._current = {}
                self._in_link = False
                self._in_snippet = False
                self._link_url = ""

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                cls = attrs_dict.get("class", "")
                if tag == "a" and "result__a" in cls:
                    self._in_link = True
                    self._current = {"title": "", "url": "", "content": "", "raw_content": "",
                                     "score": 0.5, "source": "ddg"}
                    href = attrs_dict.get("href", "")
                    if href.startswith("//"):
                        href = "https:" + href
                    self._link_url = href
                elif tag == "a" and "result__snippet" in cls:
                    self._in_snippet = True

            def handle_data(self, data):
                if self._in_link:
                    self._current["title"] = data.strip()
                elif self._in_snippet and self._current:
                    self._current["content"] = data.strip()
                    self._current["raw_content"] = data.strip()

            def handle_endtag(self, tag):
                if tag == "a" and self._in_link:
                    self._in_link = False
                    if self._current.get("title") and self._link_url:
                        self._current["url"] = _unwrap_ddg_url(self._link_url)
                        self.results.append(self._current)
                    self._current = {}
                elif tag == "a" and self._in_snippet:
                    self._in_snippet = False

        parser = DDParser()
        parser.feed(html)
        return parser.results[:max_results]
    except Exception:
        return []
