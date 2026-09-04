"""
SSRF guard for outbound fetching (tool adapters).

Centralizes URL safety checks so every adapter that downloads arbitrary
content (builtin scraper, firecrawl fallback, mineru/nougat/llamaparse PDF
fetch) enforces the same policy:

- only http/https schemes (no file://, ftp://, data:, …)
- no credentials embedded in the URL
- hostname must not resolve to a loopback / private / link-local /
  reserved address, and must not be a known cloud-metadata host
- literal IP hosts in those ranges are rejected without DNS

DNS itself is NOT resolved here (latency + failure modes); literal-IP
and hostname denylists cover the practical attack surface for a
research agent fetching search-result URLs.
"""

from __future__ import annotations

import ipaddress
import urllib.parse

# Cap for unbounded resp.read() calls on PDF / document downloads.
MAX_DOWNLOAD_BYTES = 8_000_000

_BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "metadata.goog",
}

# 169.254.169.254 (AWS/GCP/Azure metadata) + IPv6 equivalent
_METADATA_IPS = {"169.254.169.254", "fd00:ec2::254", "::ffff:169.254.169.254"}


def _host_is_blocked(host: str) -> bool:
    h = (host or "").strip().lower().rstrip(".")
    if not h:
        return True
    if h in _BLOCKED_HOSTNAMES or h in _METADATA_IPS:
        return True
    # Literal IP → reject non-global addresses (loopback, RFC1918,
    # link-local, multicast, reserved, unspecified).
    try:
        ip = ipaddress.ip_address(h)
        return not ip.is_global
    except ValueError:
        pass
    # Hostname heuristics: localhost variants and .local/.internal zones,
    # plus single-label names that can only resolve via search domains.
    if h == "localhost" or h.endswith((".localhost", ".local", ".internal", ".lan")):
        return True
    if "." not in h:
        return True
    return False


def is_safe_url(url: str) -> bool:
    """Return True if `url` is safe for server-side fetching."""
    if not url or not isinstance(url, str):
        return False
    try:
        parts = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    if parts.username or parts.password:
        return False
    try:
        host = parts.hostname or ""
    except ValueError:
        return False
    return not _host_is_blocked(host)


def bounded_read(resp, limit: int = MAX_DOWNLOAD_BYTES) -> bytes:
    """Read at most `limit` bytes from a response object."""
    chunks: list[bytes] = []
    remaining = limit
    while remaining > 0:
        data = resp.read(min(65536, remaining))
        if not data:
            break
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)
