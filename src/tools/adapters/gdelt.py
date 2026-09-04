"""
GDELT DOC 2.0 News Adapter — zero-key, real-time global newswire.

GDELT (Global Database of Events, Language, and Tone) monitors broadcast,
print, and web news in 100+ languages across virtually every country.
The DOC 2.0 API needs NO API key and has NO daily quota — the only
always-available newswire source for a zero-config research agent.

Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
  - mode=artlist   → list of articles
  - format=json
  - sort=datedesc  → newest first (freshness matters for newswire)
  - timespan       → e.g. "18m", "1y" — recency window

Rate-limit resilience (learned in the 15-topic benchmark round 1→2):
GDELT is free and unlimited but NOT unbounded — 6 concurrent benchmark
processes each firing 2+ calls per iteration triggered HTTP 429 storms
and empty JSON bodies. This adapter therefore:
  1. Enforces a cross-process minimum interval between calls (flock on a
     lock file), so concurrent runs serialize instead of hammering.
  2. On 429: backs off and retries once, then returns [] gracefully.
  3. Validates the response is actually JSON; empty/HTML error bodies
     are treated as "no results", never a crash.

Note: GDELT indexes open-web journalism; it does not bypass paywalls for
Reuters/FT/Caixin, but their freely syndicated wire copy appears in the index.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from typing import List, Dict

try:
    import fcntl  # POSIX only; Windows falls back to in-process lock
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - windows
    _HAS_FCNTL = False

# Minimum wall-clock seconds between GDELT calls across ALL processes.
# GDELT's free API throttles hard under concurrency (seen in the 15-topic
# benchmark: 6 parallel runs triggered 429 storms at 1.5s spacing). We use a
# long skip-based cooldown instead of blocking: if another process called
# GDELT recently, we return [] and the researcher retries next iteration.
# Research iterations are minutes apart, so every run eventually gets a slot.
_MIN_CALL_INTERVAL_S = 45.0

# One retry with a short backoff on 429; after that, skip news this round.
_RETRY_DELAY_S = 4.0

_lock = threading.Lock()
_last_call = 0.0
_lock_path = os.path.join(tempfile.gettempdir(), "research_agent_gdelt.lock")


def _throttle() -> bool:
    """Cross-process cooldown. Returns True if the call may proceed.

    Non-blocking: if another GDELT call happened within the cooldown window
    (this process or any other via the lock file), returns False and the
    caller skips this round — the researcher will retry next iteration.

    The lock file stores WALL-CLOCK time (time.time): the in-process fast
    path uses monotonic, but monotonic resets on reboot while the file
    persists — mixing them bricked GDELT for hours after a restart.
    All file I/O goes through the flock'd fd (O_NOFOLLOW) so a symlinked
    lock path cannot truncate arbitrary files.
    """
    global _last_call
    now_mono = time.monotonic()
    with _lock:
        if now_mono - _last_call < _MIN_CALL_INTERVAL_S:
            return False
        # Cross-process check under flock
        if _HAS_FCNTL:
            flags = os.O_CREAT | os.O_RDWR
            nofollow = getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(_lock_path, flags | nofollow, 0o644)
            except OSError:
                # Symlink / unsafe lock path — fail closed for this round.
                return False
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    with os.fdopen(os.dup(fd), "r") as f:
                        last = float(f.read().strip() or "0")
                except (OSError, ValueError):
                    last = 0.0
                if time.time() - last < _MIN_CALL_INTERVAL_S:
                    return False
                os.lseek(fd, 0, os.SEEK_SET)
                os.ftruncate(fd, 0)
                os.write(fd, str(time.time()).encode("ascii"))
            except OSError:
                pass
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
        _last_call = time.monotonic()
    return True


def _sanitize_query(query: str) -> str:
    """GDELT has a query language; strip operators so plain words pass through.

    GDELT treats quotes specially and rejects some characters. We keep the
    alphanumeric core of the query (up to a handful of terms), which is enough
    to surface the current wire coverage of the topic.
    """
    import re
    words = re.findall(r"[A-Za-z0-9]{3,}", query or "")
    # Drop very generic terms that would match everything (and nothing useful)
    stop = {
        "the", "and", "for", "with", "what", "how", "does", "are", "was",
        "that", "this", "from", "into", "about", "2026", "2025", "vs",
    }
    kept = [w for w in words if w.lower() not in stop][:6]
    if not kept:
        kept = words[:4]
    return " ".join(kept)


def _fetch(url: str) -> str:
    """GET with retry on 429; returns response text or '' on failure."""
    for attempt in (0, 1):
        req = urllib.request.Request(
            url,
            headers={"accept": "application/json", "user-agent": "research-agent/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=20.0) as resp:
                return resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt == 0:
                time.sleep(_RETRY_DELAY_S)
                continue
            print(f"  [gdelt] HTTP {e.code}")
            return ""
        except Exception as e:
            print(f"  [gdelt] request failed ({e})")
            return ""
    return ""


def gdelt_search(query: str, max_results: int = 6, timespan: str = "1y") -> List[Dict]:
    """Real-time global news search via GDELT DOC 2.0 (no key required).

    Returns results shaped like other tool-bus adapters so the researcher
    treats them uniformly: {title, url, content, published_date, source}.
    """
    if not _throttle():
        return []  # another process called GDELT recently — retry next iteration
    try:
        q = urllib.parse.quote(_sanitize_query(query))
        # timespan is interpolated into the GDELT query string — validate it
        # against the documented shape (digits + unit) so a caller cannot
        # inject extra &-joined parameters.
        import re as _re
        span = timespan if _re.fullmatch(r"\d+[a-zA-Z]+", timespan or "") else "1y"
        params = (
            f"query={q}&mode=artlist&format=json&maxrecords={min(max(max_results, 1), 25)}"
            f"&sort=datedesc&timespan={urllib.parse.quote(span)}"
        )
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?{params}"
        text = _fetch(url)
        if not text.strip():
            return []
        if not text.strip().startswith(("{", "[")):
            # GDELT sometimes returns an HTML error page for odd queries
            return []
        body = json.loads(text)

        results: List[Dict] = []
        for a in body.get("articles", [])[:max_results]:
            title = a.get("title", "") or ""
            article_url = a.get("url", "") or ""
            if not article_url:
                continue
            # seendate format: YYYYMMDDHHMMSS (or YYYYMMDDTHHMMSSZ)
            seen = a.get("seendate", "") or ""
            date = ""
            if len(seen) >= 8:
                date = f"{seen[:4]}-{seen[4:6]}-{seen[6:8]}"
            snippet = a.get("snippet", "") or ""
            # Prepend date so the retrieval guard's freshness scorer sees it
            content = (f"({date}) {snippet}" if date else snippet)[:1200]
            results.append({
                "title": title,
                "url": article_url,
                "content": content,
                "raw_content": content,
                "score": 0.85,
                "source": "gdelt",
                "published_date": date,
                "source_name": a.get("domain", "") or "",
                "language": a.get("language", ""),
            })
        return results
    except Exception as e:
        print(f"  [gdelt] search failed ({e})")
        return []


def gdelt_extract(urls: List[str]) -> List[Dict]:
    """No dedicated extract endpoint — GDELT returns snippets only.

    Return [] so the registry falls through to the built-in scraper.
    """
    return []
