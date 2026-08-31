"""Shared helpers for the ad-hoc phase test suites.

Offline/live strategy: suites that require live network (LLM APIs, web
search, DNS) declare it via ``require_live_network()``. CI / sandboxes set
``PROVIDENCE_OFFLINE=1`` (or simply have no DNS) and those suites skip
cleanly with exit code 0 instead of hanging or failing spuriously.
"""

import os
import socket
import sys

_PROBE_HOST = "api.groq.com"


def offline_mode() -> bool:
    """True when the environment asks for offline-only test runs."""
    return os.getenv("PROVIDENCE_OFFLINE", "").strip() not in ("", "0", "false")


def _has_dns() -> bool:
    try:
        socket.setdefaulttimeout(3.0)
        socket.gethostbyname(_PROBE_HOST)
        return True
    except OSError:
        return False


def require_live_network(suite: str) -> None:
    """Exit 0 (skip) when live network is unavailable or disabled."""
    if offline_mode():
        print(f"[{suite}] PROVIDENCE_OFFLINE=1 — skipping live-network suite")
        sys.exit(0)
    if not _has_dns():
        print(f"[{suite}] no DNS/network access — skipping live-network suite "
              f"(set PROVIDENCE_OFFLINE=1 to make this explicit)")
        sys.exit(0)
