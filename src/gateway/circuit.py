"""
Circuit breaker for LLM endpoints (Martin Fowler pattern).

An LLM provider/deployment can start failing (5xx, 429 exhaustion, timeouts).
If we keep hammering it, in-flight requests pile up and failures cascade. A
circuit breaker monitors failures and, once a threshold is reached, "opens"
so subsequent calls fast-fail without even touching the network. After a
cooldown it goes HALF-OPEN and lets a few trial requests through; if they
succeed it closes, otherwise it re-opens.

States:
  CLOSED    -> normal operation; failures counted
  OPEN      -> fast-fail for `cooldown` seconds
  HALF_OPEN -> trial window; a bounded number of probes allowed

Only *retriable* failures trip the breaker (client errors like 400/401 are
application bugs, not unhealthy-endpoint signals -- those must not open it).
"""

from __future__ import annotations

import logging

import threading
import time
from typing import Callable, Dict, Optional

CLOSED = "CLOSED"
OPEN = "OPEN"
HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        cooldown_s: float = 30.0,
        half_open_max: int = 2,
        on_state_change: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_s = max(0.1, cooldown_s)
        self.half_open_max = max(1, half_open_max)
        self._on_state_change = on_state_change

        self._lock = threading.RLock()
        self._state = CLOSED
        self._failures = 0
        self._last_failure_ts: Optional[float] = None
        self._half_open_inflight = 0

    # ---- public API -----------------------------------------------------
    def allow_request(self) -> bool:
        """Return True if a request may proceed, False if it must fast-fail."""
        with self._lock:
            if self._state == CLOSED:
                return True
            if self._state == OPEN:
                if time.time() - self._last_failure_ts >= self.cooldown_s:
                    self._transition(HALF_OPEN)
                    self._half_open_inflight = 0
                    return True
                return False
            # HALF_OPEN: allow a bounded number of concurrent probes
            if self._half_open_inflight < self.half_open_max:
                self._half_open_inflight += 1
                return True
            return False

    def on_success(self) -> None:
        """Report a successful call on this route."""
        with self._lock:
            self._failures = 0
            # Decrement the half-open probe counter BEFORE any state
            # transition — _transition(CLOSED) makes the HALF_OPEN branch
            # unreachable afterwards (previously a dead-code bug that leaked
            # probe slots).
            if self._state == HALF_OPEN:
                self._half_open_inflight = max(0, self._half_open_inflight - 1)
                self._transition(CLOSED)
            elif self._state == OPEN:
                self._transition(CLOSED)

    def on_failure(self, retriable: bool) -> None:
        """Report a failure. Only retriable failures should trip the breaker."""
        with self._lock:
            if self._state == HALF_OPEN:
                # a failed probe re-opens immediately
                self._half_open_inflight = max(0, self._half_open_inflight - 1)
                self._record_failure()
                self._transition(OPEN)
                return
            # CLOSED
            if not retriable:
                return
            self._record_failure()
            if self._failures >= self.failure_threshold:
                self._transition(OPEN)

    def _record_failure(self) -> None:
        self._failures += 1
        self._last_failure_ts = time.time()

    def _transition(self, new_state: str) -> None:
        if new_state == self._state:
            return
        old = self._state
        self._state = new_state
        if new_state == OPEN:
            self._failures = self.failure_threshold
        if self._on_state_change:
            try:
                self._on_state_change(self.name, new_state)
            except Exception:
                logging.getLogger(__name__).debug("ignored error", exc_info=True)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._half_open_inflight = 0
            self._transition(CLOSED)

    def open_now(self) -> None:
        """Operator tool: force the breaker open (maintenance)."""
        with self._lock:
            self._record_failure()
            self._transition(OPEN)


class CircuitRegistry:
    """Holds one breaker per route string and hands them out on demand."""

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_s: float = 30.0,
        half_open_max: int = 2,
        on_state_change: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        self._cfg = (failure_threshold, cooldown_s, half_open_max)
        self._on_state_change = on_state_change

    def get(self, route: str) -> CircuitBreaker:
        with self._lock:
            cb = self._breakers.get(route)
            if cb is None:
                cb = CircuitBreaker(
                    route,
                    failure_threshold=self._cfg[0],
                    cooldown_s=self._cfg[1],
                    half_open_max=self._cfg[2],
                    on_state_change=self._on_state_change,
                )
                self._breakers[route] = cb
            return cb

    def snapshot(self) -> Dict[str, str]:
        with self._lock:
            return {name: cb.state for name, cb in self._breakers.items()}
